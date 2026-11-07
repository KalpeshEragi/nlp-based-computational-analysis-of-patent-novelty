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
from retrieval.pipeline import PriorArtSearch
from retrieval.sources import Query


class Analyzer:
    def __init__(self, search: PriorArtSearch | None = None, n_refs: int = 5):
        self.search = search or PriorArtSearch()
        self.n_refs = n_refs
        self._checker = None

    @property
    def checker(self) -> DisclosureChecker:
        if self._checker is None:
            self._checker = DisclosureChecker(self.search.source.encoder)
        return self._checker

    def suggest_cpc(self, doc: ExtractedPatent, k: int = 5) -> list[dict]:
        return self.search.suggest_cpc(Query(doc.title, doc.abstract, doc.summary, doc.claims), k)

    def _refs_per_claim(self, layer3: dict) -> dict[int, list[str]]:
        """Top references for each claim: patents with the strongest claim matches."""
        out = {}
        for num, r in layer3["results"].items():
            score = defaultdict(float)
            for m in r["matches"][:30]:
                score[m["patent_id"]] = max(score[m["patent_id"]], m["score"])
            out[num] = sorted(score, key=score.get, reverse=True)[:self.n_refs]
        return out

    def analyze(self, doc: ExtractedPatent, before_date: str | None = None,
                cpc_codes: list[str] | None = None, prior_art: dict | None = None) -> dict:
        """
        prior_art: optional {claim_num: [patent_ids]} to analyse against given
        references (evaluation "oracle" mode, e.g. examiner citations) instead
        of searching.
        """
        t0 = time.time()
        timing = {}
        q = Query(title=doc.title, abstract=doc.abstract, summary=doc.summary, claims=doc.claims,
                  cpc_codes=cpc_codes or [], before_date=before_date)
        claims = parse_claims(doc.claims)
        if not claims:
            raise ValueError("no claims could be parsed from the document")

        search = None
        if prior_art is None:
            search = self.search.run(q)
            refs = self._refs_per_claim(search["claims"])
        else:
            refs = {c.num: list(prior_art.get(c.num, prior_art.get("all", []))) for c in claims}
        timing["search_s"] = round(time.time() - t0, 1)

        # features (each unique element once)
        t = time.time()
        texts = list(dict.fromkeys(e for c in claims for e in c.all_elements))
        feats = {x: extract_feature(x) for x in texts}
        femb = dict(zip(texts, self.checker.encode(texts)))
        comps = [feats[x].component or x for x in texts]
        cemb = dict(zip(texts, self.checker.encode(comps, max_len=32)))
        timing["features_s"] = round(time.time() - t, 1)

        # references
        t = time.time()
        all_pids = list(dict.fromkeys(p for v in refs.values() for p in v))
        docs = self.search.source.get_documents(all_pids)
        for pid in all_pids:
            if pid in docs:
                self.checker.load_reference(pid, docs[pid])
        titles = {p: d["title"] for p, d in docs.items()}

        effects = find_effects(self.checker, doc.summary, doc.abstract)
        eemb = self.checker.encode([e.sentence for e in effects]) if effects else np.zeros((0, 768))
        timing["references_effects_s"] = round(time.time() - t, 1)

        t = time.time()
        verdicts = []
        for c in claims:
            fl = [feats[x] for x in c.all_elements]
            fe = np.stack([femb[x] for x in c.all_elements])
            ce = np.stack([cemb[x] for x in c.all_elements])
            pids = [p for p in refs.get(c.num, []) if p in docs]
            evidence = self.checker.check(fl, fe, ce, pids) if pids else {}
            v = judge_claim(c.num, c.independent, fl, fe, evidence, effects, eemb, titles)
            v.parent = c.parent
            verdicts.append(v)
        timing["judging_s"] = round(time.time() - t, 1)
        timing["total_s"] = round(time.time() - t0, 1)

        return {
            "document": doc.to_dict(),
            "cpc": {"suggested": search["suggested_cpc"], "used": search["used_cpc"],
                    "filter_note": search["filter_note"]} if search else None,
            "patent": judge_patent(verdicts),
            "claims": [{**v.to_dict(), "parent": getattr(v, "parent", None)} for v in verdicts],
            "effects": [e.to_dict() for e in effects],
            "references": {p: {"title": d["title"], "filing_date": d["filing_date"],
                               "publication_number": d["publication_number"]}
                           for p, d in docs.items()},
            "timing": timing,
        }
