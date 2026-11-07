"""
Prior-art sources behind one interface, so the search pipeline does not care
where patents come from.

    PriorArtSource            interface
    └── LocalIndexSource      our HUPD corpus + vector/BM25 indexes (this file)
        (later: PatentsViewSource, EpoOpsSource, OpenAlexSource for papers)
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import bm25s
import numpy as np
import pandas as pd

from retrieval.claims import independent_claims

# Patent-level score weights (claims matter most: novelty is judged on claims)
WEIGHTS = {"abstract": 0.3, "summary": 0.2, "claims": 0.5}
RRF_K = 60  # standard Reciprocal Rank Fusion constant


@dataclass
class Query:
    """The user's invention, as extracted from their uploaded document."""
    title: str = ""
    abstract: str = ""
    summary: str = ""
    claims: str = ""                                       # full claims block
    cpc_codes: list[str] = field(default_factory=list)     # confirmed in Layer 1
    before_date: str | None = None                         # "YYYYMMDD"; prior art must be older

    def independent_claims(self) -> str:
        indep = independent_claims(self.claims)
        return " ".join(indep) if indep else self.claims

    def bm25_text(self) -> str:
        return f"{self.title} {self.abstract} {self.independent_claims()}"


@dataclass
class Candidate:
    patent_id: str
    title: str
    score: float                       # final fused score (RRF)
    dense: float                       # weighted cosine similarity
    dense_rank: int | None
    bm25_rank: int | None
    parts: dict                        # per-field cosine similarities
    subclasses: list[str]
    filing_date: str


class PriorArtSource(ABC):
    @abstractmethod
    def search(self, query: Query, k: int = 500) -> list[Candidate]:
        """Layer 1 + Layer 2: return the k most relevant prior-art patents."""

    @abstractmethod
    def get_claims(self, patent_ids: list[str]) -> dict[str, str]:
        """Full claims text per patent, for Layer 3 claim matching."""


class LocalIndexSource(PriorArtSource):
    def __init__(self, index_dir: str | Path = "retrieval/index", encoder=None):
        self.dir = Path(index_dir)
        self.meta = json.loads((self.dir / "meta.json").read_text())
        self._encoder = encoder
        self._loaded = False

    def _load(self):
        """Corpus, vectors and BM25 are loaded on first use (several GB of RAM)."""
        if self._loaded:
            return
        self._df = pd.read_parquet(self.dir / "corpus.parquet")
        # kept in float16 (as stored); only the rows being searched are upcast
        self._emb = {f: np.load(self.dir / f"emb_{f}.npy") for f in WEIGHTS}
        self._has_summary = np.load(self.dir / "has_summary.npy")
        self._bm25 = bm25s.BM25.load(self.dir / "bm25")
        self._dates = self._df["filing_date"].fillna("").to_numpy()
        # Layer 1 lookup: CPC subclass -> row numbers
        self._cpc_rows: dict[str, list[int]] = {}
        for i, subs in enumerate(self._df["subclasses"]):
            for s in subs:
                self._cpc_rows.setdefault(s, []).append(i)
        self._loaded = True

    df = property(lambda self: (self._load(), self._df)[1])
    emb = property(lambda self: (self._load(), self._emb)[1])
    has_summary = property(lambda self: (self._load(), self._has_summary)[1])
    bm25 = property(lambda self: (self._load(), self._bm25)[1])
    dates = property(lambda self: (self._load(), self._dates)[1])
    cpc_rows = property(lambda self: (self._load(), self._cpc_rows)[1])

    # -- helpers -----------------------------------------------------------
    @property
    def encoder(self):
        if self._encoder is None:
            import torch
            from sentence_transformers import SentenceTransformer
            cuda = torch.cuda.is_available()
            # load straight into fp16 on GPU (half the memory, ~2x faster)
            self._encoder = SentenceTransformer(
                self.meta["embed_model"],
                model_kwargs={"torch_dtype": torch.float16} if cuda else None)
        return self._encoder

    def _encode(self, text: str, field: str) -> np.ndarray:
        self.encoder.max_seq_length = self.meta["fields"][field]
        return self.encoder.encode([text], normalize_embeddings=True)[0].astype(np.float32)

    def candidate_mask(self, query: Query, min_pool: int = 1000) -> tuple[np.ndarray, str]:
        """Layer 1: CPC + date filter. Falls back to all patents if too few remain."""
        n = len(self.df)
        date_ok = np.ones(n, bool)
        if query.before_date:
            date_ok = (self.dates != "") & (self.dates < query.before_date)
        if query.cpc_codes:
            cpc_ok = np.zeros(n, bool)
            for code in query.cpc_codes:
                cpc_ok[self.cpc_rows.get(code[:4].upper(), [])] = True
            mask = cpc_ok & date_ok
            if mask.sum() >= min_pool:
                return mask, f"cpc filter kept {int(mask.sum())} of {n}"
            return date_ok, f"cpc filter kept only {int(mask.sum())}; using all {int(date_ok.sum())}"
        return date_ok, f"no cpc codes; using all {int(date_ok.sum())}"

    # -- Layer 2 -----------------------------------------------------------
    def search(self, query: Query, k: int = 500, bm25_pool: int = 3000) -> list[Candidate]:
        mask, note = self.candidate_mask(query)
        self.last_note = note
        rows = np.flatnonzero(mask)

        # dense: compare each query section with the same section of each patent
        q_text = {"abstract": f"{query.title}. {query.abstract}".strip(". "),
                  "summary": query.summary,
                  "claims": query.independent_claims()}
        parts, wsum = {}, np.zeros(len(rows), np.float32)
        dense = np.zeros(len(rows), np.float32)
        for f, w in WEIGHTS.items():
            if not q_text[f].strip():
                continue
            sims = self.emb[f][rows].astype(np.float32) @ self._encode(q_text[f], f)
            weight = np.full(len(rows), w, np.float32)
            if f == "summary":
                weight[~self.has_summary[rows]] = 0  # patent has no summary
            parts[f] = sims
            dense += weight * sims
            wsum += weight
        dense /= np.maximum(wsum, 1e-9)
        dense_order = np.argsort(-dense)
        dense_rank = np.empty(len(rows), int)
        dense_rank[dense_order] = np.arange(len(rows))

        # sparse: BM25 over the whole corpus, then keep rows that pass Layer 1
        ids, _ = self.bm25.retrieve(bm25s.tokenize([query.bm25_text()], stopwords="en",
                                                   show_progress=False),
                                    k=min(bm25_pool, len(self.df)), show_progress=False)
        pos = {r: i for i, r in enumerate(rows)}
        bm25_rank = {}
        for r in ids[0]:
            if r in pos:
                bm25_rank[pos[r]] = len(bm25_rank)

        # Reciprocal Rank Fusion of both rankings
        rrf = 1.0 / (RRF_K + dense_rank + 1)
        for i, rk in bm25_rank.items():
            rrf[i] += 1.0 / (RRF_K + rk + 1)

        top = np.argsort(-rrf)[:k]
        out = []
        for i in top:
            row = self.df.iloc[rows[i]]
            out.append(Candidate(
                patent_id=row["patent_id"], title=row["title"], score=float(rrf[i]),
                dense=float(dense[i]),
                dense_rank=int(dense_rank[i]) if dense_rank[i] < bm25_pool else None,
                bm25_rank=bm25_rank.get(i),
                parts={f: float(s[i]) for f, s in parts.items()},
                subclasses=list(row["subclasses"]), filing_date=row["filing_date"]))
        return out

    def get_claims(self, patent_ids: list[str]) -> dict[str, str]:
        sub = self.df[self.df["patent_id"].isin(set(patent_ids))]
        return dict(zip(sub["patent_id"], sub["claims"]))

    def get_documents(self, patent_ids: list[str]) -> dict[str, dict]:
        """Title, abstract, summary and claims: the disclosure the analysis checks against."""
        sub = self.df[self.df["patent_id"].isin(set(patent_ids))]
        cols = ["title", "abstract", "summary", "claims", "filing_date", "publication_number"]
        return {r["patent_id"]: {c: r[c] for c in cols} for _, r in sub.iterrows()}
