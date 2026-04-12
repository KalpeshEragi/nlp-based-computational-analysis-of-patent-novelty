"""
Embedding and similarity module.
Uses sentence-transformers (all-MiniLM-L6-v2) to compute claim embeddings
and cosine similarity for semantic filtering.
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load model once at module level to avoid repeated loading
MODEL_NAME = "all-MiniLM-L6-v2"
_model = SentenceTransformer(MODEL_NAME)


def compute_embeddings(parsed_claims: list[dict]) -> list[dict]:
    """
    Compute and attach sentence embeddings to each parsed claim dict.
    Mutates in place and also returns the list for convenience.
    """
    texts = [c["text"] for c in parsed_claims]
    embeddings = _model.encode(texts, show_progress_bar=False)  # shape: (N, 384)
    for claim, emb in zip(parsed_claims, embeddings):
        claim["embedding"] = emb
    return parsed_claims


def get_top_k_similar(
    invention_claim: dict,
    prior_claims: list[dict],
    k: int = 5,
) -> list[dict]:
    """
    For a single invention claim, return the TOP-K most similar prior art claims
    by cosine similarity of their embeddings.

    Returns a list of dicts, each containing:
        {
            "patent_id": str,
            "claim_text": str,
            "similarity": float,
            "parsed": dict       # full parsed prior art claim
        }
    """
    inv_emb = invention_claim["embedding"].reshape(1, -1)
    prior_embs = np.array([c["embedding"] for c in prior_claims])  # (M, 384)

    scores = cosine_similarity(inv_emb, prior_embs)[0]  # (M,)

    # Get top-k indices sorted by descending similarity
    top_indices = np.argsort(scores)[::-1][:k]

    results = []
    for idx in top_indices:
        prior = prior_claims[idx]
        results.append({
            "patent_id": prior.get("patent_id", "Unknown"),
            "claim_text": prior["text"],
            "similarity": float(scores[idx]),
            "parsed": prior,
        })
    return results