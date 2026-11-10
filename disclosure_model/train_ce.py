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


def clip(text: str, words: int) -> str:
    return " ".join(text.split()[:words])


# Some paragraphs are thousands of tokens; clip the text first so a pair stays
# near the 384-token limit (~96 + 220 words). The tokenizer also truncates.
Q_WORDS, P_WORDS = 96, 220


def to_pairs(examples):
    q, p, y = [], [], []
    for e in examples:
        for k, text in e["options"].items():
            if k in e["silver"]:
                continue
            q.append(clip(e["query"], Q_WORDS)); p.append(clip(text, P_WORDS))
            y.append(float(k in e["gold"]))
    return q, p, y


class Scorer:
    """Relevance scores for (query, paragraph) pairs from a 1-output classifier."""

    def __init__(self, model, tok, max_length, device):
        self.model, self.tok, self.max_length, self.device = model, tok, max_length, device

    def __call__(self, pairs, batch_size=32):
        import torch
        self.model.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(pairs), batch_size):
                q, p = zip(*pairs[i:i + batch_size])
                enc = self.tok(list(q), list(p), truncation="longest_first",
                               max_length=self.max_length, padding=True,
                               return_tensors="pt").to(self.device)
                with torch.autocast(self.device.type, enabled=self.device.type == "cuda"):
                    out.extend(self.model(**enc).logits[:, 0].float().cpu().tolist())
        return out


def top1(score, examples) -> dict:
    hits, rr = [], []
    for e in examples:
        keys = list(e["options"])
        s = score([(clip(e["query"], Q_WORDS), clip(e["options"][k], P_WORDS)) for k in keys])
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
    ap.add_argument("--max-steps", type=int, default=0, help="stop early (smoke tests)")
    ap.add_argument("--smoke", action="store_true", help="train on a slice of PI4PC validation")
    ap.add_argument("--data-dir", default="hf_data")
    args = ap.parse_args()

    import math
    import torch
    from torch import nn
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              get_linear_schedule_with_warmup)

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(args.backbone)
    # an NLI checkpoint has a 3-way head; replace it with a fresh 1-output relevance head
    model = AutoModelForSequenceClassification.from_pretrained(
        args.backbone, num_labels=1, ignore_mismatched_sizes=True).to(device)
    score = Scorer(model, tok, args.max_length, device)
    base = top1(score, val_ex[:300])
    print("before training (val subset):", base, flush=True)

    order = np.random.default_rng(0).permutation(len(y))
    bs = args.batch_size
    steps = math.ceil(len(y) / bs) * args.epochs
    if args.max_steps:
        steps = min(steps, args.max_steps)
    steps = int(steps)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(optim, int(0.1 * steps), steps)
    pos_weight = torch.tensor((1 - np.mean(y)) / max(np.mean(y), 1e-6), device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    print(f"batch {bs}, {steps} steps on {device}", flush=True)

    model.train()
    running, t0 = 0.0, __import__("time").time()
    for step in range(steps):
        idx = order[(step * bs) % len(y):(step * bs) % len(y) + bs]
        enc = tok([q[i] for i in idx], [p[i] for i in idx], truncation="longest_first",
                  max_length=args.max_length, padding=True, return_tensors="pt").to(device)
        target = torch.tensor([y[i] for i in idx], device=device)
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            logits = model(**enc).logits[:, 0]
        loss = loss_fn(logits.float(), target)
        scaler.scale(loss).backward()
        scaler.unscale_(optim)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optim)
        scaler.update()
        optim.zero_grad(set_to_none=True)
        sched.step()
        running += loss.item()
        if (step + 1) % 500 == 0:
            print(f"step {step + 1}/{steps} loss {running / 500:.4f} "
                  f"({__import__('time').time() - t0:.0f}s)", flush=True)
            running = 0.0
    if device.type == "cuda":
        print(f"peak GPU memory {torch.cuda.max_memory_allocated() / 1e9:.2f} GB", flush=True)

    res = top1(score, val_ex)
    print("after training (PI4PC validation, NOC4PC-test apps excluded):", res, flush=True)
    out = Path(args.out)
    (out / "model").mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out / "model")
    tok.save_pretrained(out / "model")
    (out / "metrics.json").write_text(json.dumps(
        {"backbone": args.backbone, "args": vars(args), "before_subset": base, "validation": res,
         "n_train_examples": len(train_ex), "n_pairs": len(y), "steps": steps}, indent=2))
    print("saved ->", out / "model")


if __name__ == "__main__":
    main()
