"""
Layer 1 + Layer 2 together:

    invention text -> CPC classifier suggests codes (user confirms in the UI)
                   -> CPC/date filter -> hybrid search -> top-k prior-art patents
"""

from dataclasses import replace

from cpc_classifier.predict import CPCPredictor
from retrieval.sources import LocalIndexSource, PriorArtSource, Query


class PriorArtSearch:
    def __init__(self, source: PriorArtSource | None = None,
                 predictor: CPCPredictor | None = None):
        self.source = source or LocalIndexSource("retrieval/index")
        self.predictor = predictor or CPCPredictor("cpc_classifier/model_v2")

    def suggest_cpc(self, q: Query, k: int = 5) -> list[dict]:
        """Layer 1 step 1: codes to show the user for confirmation."""
        return self.predictor.predict(q.title, q.abstract, q.claims, q.summary, k=k)

    def run(self, q: Query, k: int = 500, n_codes: int = 5) -> dict:
        """Full run. Uses q.cpc_codes if the user confirmed some, else the top predictions."""
        suggested = self.suggest_cpc(q, n_codes)
        codes = q.cpc_codes or [s["code"] for s in suggested]
        results = self.source.search(replace(q, cpc_codes=codes), k=k)
        return {"suggested_cpc": suggested, "used_cpc": codes,
                "filter_note": getattr(self.source, "last_note", ""), "results": results}
