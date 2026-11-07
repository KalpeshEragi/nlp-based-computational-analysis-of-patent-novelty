"""
Prior-art search pipeline, Layers 1-3:

    invention -> [L1] CPC classifier suggests codes (user confirms in the UI)
              -> [L1] CPC + date filter
              -> [L2] hybrid patent search (dense 3-field + BM25, RRF) -> top patents
              -> [L3] claim-vs-claim matching (live GPU encoding) -> top claims per input claim
"""

from dataclasses import replace

from cpc_classifier.predict import CPCPredictor
from retrieval.layer3 import ClaimMatcher
from retrieval.sources import LocalIndexSource, PriorArtSource, Query


class PriorArtSearch:
    def __init__(self, source: PriorArtSource | None = None,
                 predictor: CPCPredictor | None = None):
        self.source = source or LocalIndexSource("retrieval/index")
        self.predictor = predictor or CPCPredictor("cpc_classifier/model_v2")
        self._matcher = None

    @property
    def matcher(self) -> ClaimMatcher:
        if self._matcher is None:
            self._matcher = ClaimMatcher(self.source.encoder)
        return self._matcher

    def suggest_cpc(self, q: Query, k: int = 5) -> list[dict]:
        """Layer 1 step 1: codes to show the user for confirmation."""
        return self.predictor.predict(q.title, q.abstract, q.claims, q.summary, k=k)

    def run(self, q: Query, k: int = 500, n_codes: int = 5,
            claims_top_n: int = 150, layer3_patents: int | None = 200) -> dict:
        """
        Uses q.cpc_codes if the user confirmed some, else the top predictions.
        layer3_patents: how many Layer 2 patents feed Layer 3 (None = all k). Default 200:
            patents ranked 200-500 supplied <3% of each claim's top-20 matches in tests,
            while costing ~60% of the claim-encoding time.
        """
        suggested = self.suggest_cpc(q, n_codes)
        codes = q.cpc_codes or [s["code"] for s in suggested]
        patents = self.source.search(replace(q, cpc_codes=codes), k=k)
        claims = self.matcher.match(q.claims, patents, self.source,
                                    top_n=claims_top_n, n_patents=layer3_patents)
        return {"suggested_cpc": suggested, "used_cpc": codes,
                "filter_note": getattr(self.source, "last_note", ""),
                "patents": patents, "claims": claims}
