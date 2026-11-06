"""
Encode the corpus into search indexes (one-time, best on a GPU).

Outputs in the index folder:
    emb_abstract.npy, emb_summary.npy, emb_claims.npy   float16, L2-normalised
    has_summary.npy                                      bool mask
    bm25/                                                keyword index
    meta.json                                            model + settings

Usage:
    python -m retrieval.build_index --index retrieval/index
"""

import argparse
import json
from pathlib import Path

import bm25s
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

EMBED_MODEL = "AI-Growth-Lab/PatentSBERTa"
FIELDS = {  # field -> max tokens for that section
    "abstract": 256,
    "summary": 384,
    "claims": 384,
}


def field_texts(df: pd.DataFrame, field: str) -> list[str]:
    if field == "abstract":
        return (df["title"] + ". " + df["abstract"]).tolist()
    if field == "claims":
        return df["independent_claims"].tolist()
    return df["summary"].fillna("").tolist()


def bm25_texts(df: pd.DataFrame) -> list[str]:
    return (df["title"] + " " + df["abstract"] + " " + df["independent_claims"]).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="retrieval/index")
    ap.add_argument("--batch-size", type=int, default=128)
    args = ap.parse_args()

    idx = Path(args.index)
    df = pd.read_parquet(idx / "corpus.parquet")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(EMBED_MODEL, device=device)
    if device == "cuda":
        model.half()
    print(f"{len(df)} patents, encoding on {device}", flush=True)

    for field, max_len in FIELDS.items():
        model.max_seq_length = max_len
        texts = field_texts(df, field)
        emb = model.encode(texts, batch_size=args.batch_size, normalize_embeddings=True,
                           convert_to_numpy=True, show_progress_bar=True)
        np.save(idx / f"emb_{field}.npy", emb.astype(np.float16))
        print(f"==> {field}: {emb.shape}", flush=True)

    np.save(idx / "has_summary.npy", (df["summary"].fillna("").str.len() > 50).to_numpy())

    retriever = bm25s.BM25()
    retriever.index(bm25s.tokenize(bm25_texts(df), stopwords="en", show_progress=False))
    retriever.save(idx / "bm25")
    print("==> bm25 index saved", flush=True)

    (idx / "meta.json").write_text(json.dumps(
        {"embed_model": EMBED_MODEL, "fields": FIELDS, "n_patents": len(df)}, indent=2))


if __name__ == "__main__":
    main()
