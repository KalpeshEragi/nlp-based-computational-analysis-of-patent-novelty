"""
Fine-tune a patent-specific disclosure cross-encoder on PANORAMA PI4PC.

Pair: (claim's own text, candidate paragraph of a cited reference)
Label: 1 if the examiner cited that paragraph for the claim, else 0
       (silver = cited against another claim of the application: skipped)

Leakage control: every application appearing in NOC4PC validation/test or
PI4PC validation/test is removed from training. Model selection uses PI4PC
validation without NOC4PC-test applications.

    python -m disclosure_model.train_ce --backbone AI-Growth-Lab/PatentSBERTa --out out/ce_patentsberta
    python -m disclosure_model.train_ce --smoke      # tiny local run to test the code path
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.elements import parse_claims

REPO = "LG-AI-Research/PANORAMA"


def hf_parquet(path: str, local_dir="hf_data") -> pd.DataFrame:
    from huggingface_hub import hf_hub_download
    return pd.read_parquet(hf_hub_download(REPO, path, repo_type="dataset", local_dir=local_dir))


def excluded_apps(local_dir) -> set:
    apps = set()
    for p in ("NOC4PC/validation.parquet", "NOC4PC/test.parquet",
              "PI4PC/validation.parquet", "PI4PC/test.parquet"):
        apps |= set(hf_parquet(p, local_dir)["application_number"])
    return apps


def build_examples(df: pd.DataFrame) -> list[dict]:
    out = []
    for r in df.itertuples():
        ctx = json.loads(r.context)
        claims = {c.num: c for c in parse_claims(" ".join(ctx["claims"]))}
        c = claims.get(int(r.claim_number))
        if not c:
            continue
        opts = {int(k): v for k, v in json.loads(r.options).items()}
        out.append({"app": r.application_number, "query": " ".join(c.elements),
                    "options": opts, "gold": [int(x) for x in r.gold_answers],
                    "silver": [int(x) for x in r.silver_answers]})
    return out


def to_pairs(examples):
    q, p, y = [], [], []
    for e in examples:
        for k, text in e["options"].items():
            if k in e["silver"]:
                continue
            q.append(e["query"]); p.append(text); y.append(float(k in e["gold"]))
    return q, p, y


def top1(model, examples) -> dict:
    hits, rr = [], []
    for e in examples:
        keys = list(e["options"])
        s = model.predict([(e["query"], e["options"][k]) for k in keys], batch_size=32,
                          show_progress_bar=False)
        order = [keys[i] for i in np.argsort(-np.asarray(s))]
        hits.append(order[0] in e["gold"])
        rr.append(1 / (1 + min(order.index(g) for g in e["gold"])))
    return {"n": len(examples), "top1_acc": float(np.mean(hits)), "mrr": float(np.mean(rr))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="AI-Growth-Lab/PatentSBERTa")
    ap.add_argument("--out", default="out/disclosure_ce")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-length", type=int, default=384)
    ap.add_argument("--max-train", type=int, default=0)
    ap.add_argument("--smoke", action="store_true", help="train on a slice of PI4PC validation")
    ap.add_argument("--data-dir", default="hf_data")
    args = ap.parse_args()

    from datasets import Dataset
    from sentence_transformers.cross_encoder import (CrossEncoder, CrossEncoderTrainer,
                                                     CrossEncoderTrainingArguments)
    from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss

    val_df = hf_parquet("PI4PC/validation.parquet", args.data_dir)
    noc_test = set(hf_parquet("NOC4PC/test.parquet", args.data_dir)["application_number"])
    val_df = val_df[~val_df.application_number.isin(noc_test)]

    if args.smoke:
        apps = sorted(val_df.application_number.unique())
        random.Random(0).shuffle(apps)
        train_df = val_df[val_df.application_number.isin(apps[:20])]
        val_df = val_df[val_df.application_number.isin(apps[20:40])]
    else:
        train_df = hf_parquet("PI4PC/train.parquet", args.data_dir)
        before = train_df.application_number.nunique()
        train_df = train_df[~train_df.application_number.isin(excluded_apps(args.data_dir))]
        print(f"train apps {before} -> {train_df.application_number.nunique()} after leakage filter")

    train_ex, val_ex = build_examples(train_df), build_examples(val_df)
    if args.max_train:
        random.Random(0).shuffle(train_ex)
        train_ex = train_ex[:args.max_train]
    val_ex = val_ex[:1500]
    q, p, y = to_pairs(train_ex)
    print(f"train examples {len(train_ex)} -> {len(y)} pairs ({np.mean(y):.2%} positive); val {len(val_ex)}")

    model = CrossEncoder(args.backbone, num_labels=1, max_length=args.max_length)
    base = top1(model, val_ex[:300])
    print("before training (val subset):", base, flush=True)

    ds = Dataset.from_dict({"query": q, "passage": p, "label": y}).shuffle(seed=0)
    out = Path(args.out)
    import math
    import torch
    n_dev = max(torch.cuda.device_count(), 1)
    total_steps = math.ceil(len(ds) / (args.batch_size * n_dev)) * args.epochs
    # warmup_ratio was removed in newer transformers; give 10% as steps
    targs = CrossEncoderTrainingArguments(
        output_dir=str(out / "ckpt"), num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size, learning_rate=args.lr,
        warmup_steps=int(0.1 * total_steps),
        fp16=not args.smoke, logging_steps=200, save_strategy="no", report_to="none", seed=0)
    pos_weight = (1 - np.mean(y)) / max(np.mean(y), 1e-6)
    trainer = CrossEncoderTrainer(model=model, args=targs, train_dataset=ds,
                                  loss=BinaryCrossEntropyLoss(model, pos_weight=__import__("torch").tensor(pos_weight)))
    trainer.train()

    res = top1(model, val_ex)
    print("after training (PI4PC validation, NOC4PC-test apps excluded):", res, flush=True)
    model.save(str(out / "model"))
    (out / "metrics.json").write_text(json.dumps(
        {"backbone": args.backbone, "args": vars(args), "before_subset": base, "validation": res,
         "n_train_examples": len(train_ex), "n_pairs": len(y)}, indent=2))
    print("saved ->", out / "model")


if __name__ == "__main__":
    main()
