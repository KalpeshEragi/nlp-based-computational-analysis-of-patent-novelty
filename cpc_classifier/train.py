"""
Step 3 - Fine-tune PatentSBERTa as a multi-label CPC subclass classifier.

Input text : title + abstract + first claim (+ summary, if tokens remain)
Output     : one probability per CPC subclass (sigmoid, multi-label)

Usage:
    python -m cpc_classifier.train --quick                # tiny local test
    python -m cpc_classifier.train --epochs 3             # full run (GPU)
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

DATA_DIR = Path(__file__).parent / "data"
MODEL_NAME = "AI-Growth-Lab/PatentSBERTa"


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def build_text(row) -> str:
    parts = [row.title, row.abstract, row.first_claim, row.summary]
    return " . ".join(p for p in parts if isinstance(p, str) and p)


def build_label_space(train: pd.DataFrame, min_count: int) -> list[str]:
    counts = train["subclasses"].explode().value_counts()
    return sorted(counts[counts >= min_count].index.tolist())


class PatentDataset(Dataset):
    def __init__(self, df, labels, tokenizer, max_len):
        self.texts = [build_text(r) for r in df.itertuples()]
        self.main = df["main_subclass"].tolist()
        self.subs = df["subclasses"].tolist()
        self.idx = {c: i for i, c in enumerate(labels)}
        self.n = len(labels)
        self.tok, self.max_len = tokenizer, max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        y = torch.zeros(self.n)
        for c in self.subs[i]:
            if c in self.idx:
                y[self.idx[c]] = 1.0
        return self.texts[i], y, self.idx.get(self.main[i], -1)

    def collate(self, batch):
        texts, ys, mains = zip(*batch)
        enc = self.tok(list(texts), truncation=True, max_length=self.max_len,
                       padding=True, return_tensors="pt")
        return enc, torch.stack(ys), torch.tensor(mains)


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
class CPCClassifier(nn.Module):
    """Transformer encoder + mean pooling + linear multi-label head."""

    def __init__(self, n_labels, model_name=MODEL_NAME, dropout=0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, n_labels))

    def forward(self, input_ids, attention_mask, **_):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return self.head(pooled)


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
def compute_metrics(logits: np.ndarray, y: np.ndarray, mains: np.ndarray) -> dict:
    """
    top{k}_acc : main subclass is in the top-k predictions (the headline number)
    recall@5   : share of ALL true subclasses found in the top 5 (matters for Layer 1)
    micro_f1   : threshold 0.5 on sigmoid probabilities
    Patents whose main subclass is outside the label space count as misses.
    """
    order = np.argsort(-logits, axis=1)
    m = {}
    for k in (1, 3, 5):
        topk = order[:, :k]
        m[f"top{k}_acc"] = float(np.mean([mn >= 0 and mn in row for mn, row in zip(mains, topk)]))
    top5 = order[:, :5]
    hits = np.take_along_axis(y, top5, axis=1).sum(1)
    m["recall@5"] = float(np.sum(hits) / max(np.sum(y), 1))
    pred = (1 / (1 + np.exp(-logits))) >= 0.5
    tp = np.sum(pred & (y == 1))
    m["micro_f1"] = float(2 * tp / max(pred.sum() + y.sum(), 1))
    return m


@torch.no_grad()
def evaluate(model, loader, device, amp):
    model.eval()
    all_logits, all_y, all_main = [], [], []
    for enc, y, mains in loader:
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.autocast(device_type=device.type, enabled=amp):
            logits = model(**enc)
        all_logits.append(logits.float().cpu().numpy())
        all_y.append(y.numpy())
        all_main.append(mains.numpy())
    return compute_metrics(np.concatenate(all_logits), np.concatenate(all_y),
                           np.concatenate(all_main))


# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="use *_quick.parquet files")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--max-len", type=int, default=384)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--min-count", type=int, default=50,
                    help="drop subclasses with fewer train examples than this")
    ap.add_argument("--max-train", type=int, default=0, help="cap train rows (0 = all)")
    ap.add_argument("--out", default=str(Path(__file__).parent / "model"))
    args = ap.parse_args()

    suffix = "_quick" if args.quick else ""
    if args.quick:
        args.min_count = min(args.min_count, 3)
    train_df = pd.read_parquet(DATA_DIR / f"train{suffix}.parquet")
    val_df = pd.read_parquet(DATA_DIR / f"val{suffix}.parquet")
    test_df = pd.read_parquet(DATA_DIR / f"test{suffix}.parquet")
    if args.max_train:
        train_df = train_df.sample(min(args.max_train, len(train_df)), random_state=0)

    labels = build_label_space(train_df, args.min_count)
    print(f"train={len(train_df)} val={len(val_df)} test={len(test_df)} labels={len(labels)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)

    def loader(df, shuffle):
        ds = PatentDataset(df, labels, tok, args.max_len)
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle,
                          collate_fn=ds.collate,
                          num_workers=0 if os.name == "nt" else 2)

    train_dl, val_dl, test_dl = loader(train_df, True), loader(val_df, False), loader(test_df, False)

    model = CPCClassifier(len(labels)).to(device)
    optim = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": args.lr},
        {"params": model.head.parameters(), "lr": args.head_lr},
    ], weight_decay=0.01)
    steps = math.ceil(len(train_dl) / args.grad_accum) * args.epochs
    sched = get_linear_schedule_with_warmup(optim, int(0.06 * steps), steps)
    scaler = torch.amp.GradScaler(enabled=amp)
    loss_fn = nn.BCEWithLogitsLoss()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    best, history = -1.0, []

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, running = time.time(), 0.0
        for step, (enc, y, _) in enumerate(train_dl, 1):
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.autocast(device_type=device.type, enabled=amp):
                loss = loss_fn(model(**enc).float(), y.to(device)) / args.grad_accum
            scaler.scale(loss).backward()
            if step % args.grad_accum == 0 or step == len(train_dl):
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optim)
                scaler.update()
                optim.zero_grad(set_to_none=True)
                sched.step()
            running += loss.item() * args.grad_accum
            if step % 200 == 0:
                print(f"epoch {epoch} step {step}/{len(train_dl)} "
                      f"loss {running / step:.4f} ({time.time() - t0:.0f}s)", flush=True)

        val_m = evaluate(model, val_dl, device, amp)
        history.append({"epoch": epoch, "train_loss": running / len(train_dl), **val_m})
        print(f"[epoch {epoch}] val {json.dumps(val_m)}", flush=True)

        # best checkpoint chosen by recall@5 (what Layer 1 filtering depends on)
        if val_m["recall@5"] > best:
            best = val_m["recall@5"]
            torch.save(model.state_dict(), out / "model.pt")
            print(f"  saved best (recall@5={best:.4f})", flush=True)

    model.load_state_dict(torch.load(out / "model.pt", map_location=device))
    test_m = evaluate(model, test_dl, device, amp)
    print(f"[test] {json.dumps(test_m)}", flush=True)

    tok.save_pretrained(out)
    (out / "labels.json").write_text(json.dumps(labels))
    (out / "metrics.json").write_text(json.dumps(
        {"args": vars(args), "n_labels": len(labels), "history": history, "test": test_m},
        indent=2))
    print(f"saved model, labels and metrics to {out}")


if __name__ == "__main__":
    main()
