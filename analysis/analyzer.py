"""
Full patentability analysis:

    document -> claims/elements -> prior-art search (Layers 1-3)
             -> features -> disclosure check vs top references per claim
             -> advancement detection -> novelty + inventive-step verdicts
             -> patent-level result
"""

from __future__ import annotations

import time
from collections import defaultdict

import numpy as np

from analysis.disclosure import DisclosureChecker
from analysis.effects import find_effects
from analysis.elements import parse_claims
from analysis.engine import judge_claim, judge_patent
from analysis.extractor import ExtractedPatent
from analysis.features import extract_feature
from analysis.outcome_features import claim_info
from retrieval.pipeline import PriorArtSearch
from retrieval.sources import Query

# speed/coverage presets. "fast" is meant for CPU-only hosting.
MODES = {
    "fast": {"k": 200, "layer3_patents": 80, "n_refs": 3},
    "full": {"k": 500, "layer3_patents": 200, "n_refs": 5},
}


class Analyzer:
    def __init__(self, search: PriorArtSearch | None = None, n_refs: int = 5,
                 outcome_model="auto"):
        self.search = search or PriorArtSearch()
        self.n_refs = n_refs
        self._checker = None
        if outcome_model == "auto":
            from analysis.outcome_model import OutcomePredictor
            outcome_model = OutcomePredictor.load_if_available()
        self.outcome_model = outcome_model

    @property
    def checker(self) -> DisclosureChecker:
        if self._checker is None:
            self._checker = DisclosureChecker(self.search.source.encoder)
        return self._checker

    def suggest_cpc(self, doc: ExtractedPatent, k: int = 5) -> list[dict]:
        return self.search.suggest_cpc(Query(doc.title, doc.abstract, doc.summary, doc.claims), k)

    def reset_caches(self):
        """Drop per-analysis caches so a long-running server does not grow in memory."""
        if self._checker is not None:
            self._checker._refs.clear()
            self._checker._dis_cache.clear()
            self._checker._nli_cache.clear()
        if self.search._matcher is not None:
            self.search._matcher.clear_cache()

    def _refs_per_claim(self, layer3: dict, n_refs: int | None = None) -> dict[int, list[str]]:
        """Top references for each claim: patents with the strongest claim matches."""
        out = {}
        for num, r in layer3["results"].items():
            score = defaultdict(float)
            for m in r["matches"][:30]:
                score[m["patent_id"]] = max(score[m["patent_id"]], m["score"])
            out[num] = sorted(score, key=score.get, reverse=True)[:n_refs or self.n_refs]
        return out

    def analyze(self, doc: ExtractedPatent, before_date: str | None = None,
                cpc_codes: list[str] | None = None, prior_art: dict | None = None,
                prior_art_docs: dict | None = None, only_claims: list[int] | None = None,
                keep_inputs: bool = False, mode: str = "full", progress=None) -> dict:
        """
        prior_art:      optional {claim_num | "all": [patent_ids]} to analyse against
                        given references (evaluation "oracle" mode) instead of searching
        prior_art_docs: {patent_id: {title, abstract, claims, description}} for
                        references that are not in the local index
        only_claims:    judge only these claim numbers (their parents are still parsed)
        keep_inputs:    include each claim's raw outcome-model inputs (evaluation)
        mode:           "fast" or "full" (see MODES)
        progress:       optional callback(stage: str, percent: int)
        """
        report = progress or (lambda stage, pct: None)
        settings = MODES[mode]
        t0 = time.time()
        timing = {}
        q = Query(title=doc.title, abstract=doc.abstract, summary=doc.summary, claims=doc.claims,
                  cpc_codes=cpc_codes or [], before_date=before_date)
        claims = parse_claims(doc.claims)
        if not claims:
            raise ValueError("no claims could be parsed from the document")

        search = None
        if prior_art is None:
            report("searching prior art", 5)
            search = self.search.run(q, k=settings["k"], layer3_patents=settings["layer3_patents"])
            refs = self._refs_per_claim(search["claims"], settings["n_refs"])
        else:
            refs = {c.num: list(prior_art.get(c.num, prior_art.get("all", []))) for c in claims}
        timing["search_s"] = round(time.time() - t0, 1)

        # features (each unique element once)
        report("extracting claim features", 45)
        t = time.time()
        judged = [c for c in claims if not only_claims or c.num in only_claims]
        texts = list(dict.fromkeys(e for c in judged for e in c.all_elements))
        feats = {x: extract_feature(x) for x in texts}
        femb = dict(zip(texts, self.checker.encode(texts)))
        comps = [feats[x].component or x for x in texts]
        cemb = dict(zip(texts, self.checker.encode(comps, max_len=32)))
        timing["features_s"] = round(time.time() - t, 1)

        # references
        report("reading prior-art references", 55)
        t = time.time()
        all_pids = list(dict.fromkeys(p for v in refs.values() for p in v))
        local = [p for p in all_pids if not prior_art_docs or p not in prior_art_docs]
        docs = self.search.source.get_documents(local) if local else {}
        docs.update({p: {"filing_date": "", "publication_number": p, "summary": "", **d}
                     for p, d in (prior_art_docs or {}).items() if p in all_pids})
        for pid in all_pids:
            if pid in docs:
                self.checker.load_reference(pid, docs[pid])
        titles = {p: d["title"] for p, d in docs.items()}

        effects = find_effects(self.checker, doc.summary, doc.abstract)
        eemb = self.checker.encode([e.sentence for e in effects]) if effects else np.zeros((0, 768))
        timing["references_effects_s"] = round(time.time() - t, 1)

        t = time.time()
        verdicts, outcome_inputs = [], []
        for n, c in enumerate(judged):
            report(f"judging claim {c.num} of {len(judged)}", 60 + int(35 * n / max(len(judged), 1)))
            fl = [feats[x] for x in c.all_elements]
            fe = np.stack([femb[x] for x in c.all_elements])
            ce = np.stack([cemb[x] for x in c.all_elements])
            pids = [p for p in refs.get(c.num, []) if p in docs]
            self.checker.records = []           # raw signals for the outcome predictor
            evidence = self.checker.check(fl, fe, ce, pids) if pids else {}
            v = judge_claim(c.num, c.independent, fl, fe, evidence, effects, eemb, titles)
            v.parent = c.parent
            verdicts.append(v)
            outcome_inputs.append({"signals": self.checker.records,
                                   "info": claim_info(c, fl, len(v.linked_effects))})
        self.checker.records = None
        timing["judging_s"] = round(time.time() - t, 1)

        # examiner-outcome prediction: a separate learned model that reads the
        # evidence above; it never changes the legal verdicts
        predictions = [None] * len(verdicts)
        if self.outcome_model is not None:
            predictions = [self.outcome_model.predict(x["signals"], x["info"]) for x in outcome_inputs]
        timing["total_s"] = round(time.time() - t0, 1)
        report("done", 100)

        claims_out = []
        for v, pred, inp in zip(verdicts, predictions, outcome_inputs):
            d = {**v.to_dict(), "parent": getattr(v, "parent", None),
                 "predicted_examiner_outcome": pred}
            if keep_inputs:
                d["outcome_inputs"] = inp
            claims_out.append(d)

        return {
            "document": doc.to_dict(),
            "cpc": {"suggested": search["suggested_cpc"], "used": search["used_cpc"],
                    "filter_note": search["filter_note"]} if search else None,
            "patent": judge_patent(verdicts),
            "claims": claims_out,
            "effects": [e.to_dict() for e in effects],
            "references": {p: {"title": d["title"], "filing_date": d["filing_date"],
                               "publication_number": d["publication_number"]}
                           for p, d in docs.items()},
            "timing": timing,
        }
