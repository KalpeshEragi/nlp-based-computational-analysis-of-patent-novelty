"""
Layer 3 - claim-level matching.

For every claim of the invention, rank the individual claims of the Layer 2
candidate patents and keep the top N. Candidate claims are encoded live on the
GPU for each search (batched, fp16); a per-session cache avoids re-encoding a
patent that comes up again.

Dependent claims are matched with their context: the claim's own text first,
then its parent chain (nearest parent -> independent root). Putting the own
text first means the length limit trims context, never the new feature.
"""

from __future__ import annotations

import time
from collections import Counter

import numpy as np

from retrieval.claims import split_claims
from retrieval.sources import Candidate, PriorArtSource


def claims_with_context(claims_text: str) -> list[dict]:
    """Split claims and add `match_text` = own text + parent chain."""
    claims = split_claims(claims_text)
    by_num = {c["num"]: c for c in claims}
    for c in claims:
        chain, parent, seen = [c["text"]], c["parent"], set()
        while parent and parent in by_num and parent not in seen:
            seen.add(parent)
            chain.append(by_num[parent]["text"])
            parent = by_num[parent]["parent"]
        c["match_text"] = " ".join(chain)
        c["depth"] = len(chain) - 1
    return claims


class ClaimMatcher:
    def __init__(self, encoder, max_len: int = 192, batch_size: int = 64,
                 patent_weight: float = 0.2):
        """
        encoder        SentenceTransformer (shared with Layer 2)
        patent_weight  how much the Layer 2 patent score adds to a claim's score
        """
        self.encoder = encoder
        self.max_len = max_len
        self.batch_size = batch_size
        self.patent_weight = patent_weight
        self._cache: dict[str, tuple[list[dict], np.ndarray]] = {}

    def _encode(self, texts: list[str]) -> np.ndarray:
        self.encoder.max_seq_length = self.max_len
        if not texts:
            return np.zeros((0, self.encoder.get_sentence_embedding_dimension()), np.float32)
        return self.encoder.encode(texts, batch_size=self.batch_size,
                                   normalize_embeddings=True, convert_to_numpy=True,
                                   show_progress_bar=False).astype(np.float32)

    def _candidate_claims(self, source: PriorArtSource, patent_ids: list[str]):
        """Split + encode candidate claims; only patents not seen this session are encoded."""
        new_ids = [p for p in patent_ids if p not in self._cache]
        texts = source.get_claims(new_ids)
        parsed = {pid: claims_with_context(texts.get(pid, "")) for pid in new_ids}
        flat = [(pid, c) for pid in new_ids for c in parsed[pid]]
        emb = self._encode([c["match_text"] for _, c in flat])
        i = 0
        for pid in new_ids:
            n = len(parsed[pid])
            self._cache[pid] = (parsed[pid], emb[i:i + n])
            i += n
        return len(new_ids), len(flat)

    def match(self, invention_claims: str, candidates: list[Candidate],
              source: PriorArtSource, top_n: int = 150, n_patents: int | None = None) -> dict:
        t0 = time.time()
        cands = candidates[:n_patents] if n_patents else candidates
        inv = claims_with_context(invention_claims)
        if not inv:
            raise ValueError("no claims found in the invention text")
        q = self._encode([c["match_text"] for c in inv])
        t_inv = time.time()

        n_new, n_claims = self._candidate_claims(source, [c.patent_id for c in cands])
        t_enc = time.time()

        # flatten candidate claims, with the patent-level score for each
        dense = np.array([c.dense for c in cands], np.float32)
        span = dense.max() - dense.min()
        pscore = (dense - dense.min()) / span if span > 0 else np.ones_like(dense)
        meta, embs = [], []
        for c, ps in zip(cands, pscore):
            claims, emb = self._cache[c.patent_id]
            embs.append(emb)
            meta += [(c, ps, cl) for cl in claims]
        P = np.vstack(embs) if embs else np.zeros((0, q.shape[1]), np.float32)

        sims = q @ P.T                                              # (n_inv, n_prior)
        pvec = np.array([m[1] for m in meta], np.float32)
        final = (1 - self.patent_weight) * sims + self.patent_weight * pvec

        k = min(top_n, final.shape[1])
        results, freq = {}, Counter()
        for i, claim in enumerate(inv):
            top = np.argpartition(-final[i], k - 1)[:k]
            top = top[np.argsort(-final[i][top])]
            rows = []
            for j in top:
                cand, ps, pc = meta[j]
                rows.append({
                    "patent_id": cand.patent_id, "title": cand.title,
                    "claim_num": pc["num"], "independent": pc["independent"],
                    "text": pc["text"], "score": float(final[i, j]),
                    "claim_sim": float(sims[i, j]), "patent_score": float(ps),
                })
            results[claim["num"]] = {"claim": claim, "matches": rows}
            freq.update({r["patent_id"] for r in rows[:20]})

        return {
            "results": results,
            # patents appearing in many claims' top-20: likely key references
            "frequent_patents": freq.most_common(10),
            "stats": {
                "invention_claims": len(inv), "candidate_patents": len(cands),
                "candidate_claims": int(P.shape[0]), "newly_encoded_claims": n_claims,
                "newly_encoded_patents": n_new,
                "time_encode_invention_s": round(t_inv - t0, 2),
                "time_encode_candidates_s": round(t_enc - t_inv, 2),
                "time_total_s": round(time.time() - t0, 2),
            },
        }

    def clear_cache(self):
        self._cache.clear()
