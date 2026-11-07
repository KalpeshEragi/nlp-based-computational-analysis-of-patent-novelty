"""
Novelty + inventive-step engine (EPO problem-solution approach).

Per claim, given P(feature disclosed) for every feature x reference:

  NOVELTY      closest reference = the one disclosing most features.
               every feature disclosed there        -> NOT_NOVEL
               otherwise the missing ones are the distinguishing features,
               each labelled: new component | same component, new function |
               different parameter
  INVENTIVE    are the distinguishing features found in OTHER references?
  STEP         all of them                           -> OBVIOUS (combination)
               some found nowhere                    -> INVENTIVE
               an advancement (technical / economic effect) linked to the
               distinguishing features strengthens the inventive finding

Scores (0-1, used for ranking / ROC-AUC in the evaluation):
  anticipation = max over refs of min over features P   (all features, ONE ref)
  combination  = min over features of max over refs P   (all features, ANY refs)
  novelty_score = 1 - anticipation, inventive_score = 1 - combination (+ effect bonus)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

THRESHOLD = 0.5
EFFECT_LINK = 0.45     # cosine between an effect sentence and a feature
EFFECT_BONUS = 0.1


@dataclass
class ClaimVerdict:
    num: int
    independent: bool
    verdict: str                      # NOT_NOVEL | OBVIOUS | INVENTIVE
    reason: str
    novelty_score: float
    inventive_score: float
    closest_ref: str | None
    distinguishing: list[dict] = field(default_factory=list)
    combination_refs: list[str] = field(default_factory=list)
    linked_effects: list[dict] = field(default_factory=list)
    feature_chart: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


def _label(ev) -> str:
    if ev.param_conflict:
        return "different parameter"
    if not ev.component_found:
        return "new component"
    return "same component, new function"


def judge_claim(num, independent, features, feat_emb, evidence: dict, effects, effect_emb,
                ref_titles: dict) -> ClaimVerdict:
    pids = list(evidence)
    if not pids:
        return ClaimVerdict(num, independent, "INVENTIVE", "no prior art found", 1.0, 1.0, None)

    P = np.array([[ev.prob for ev in evidence[p]] for p in pids])      # refs x features
    covered = P >= THRESHOLD
    order = sorted(range(len(pids)), key=lambda r: (covered[r].sum(), P[r].mean()), reverse=True)
    c = order[0]
    closest = pids[c]
    anticipation = float(P.min(axis=1).max())
    combination = float(P.max(axis=0).min())

    chart = []
    for i, f in enumerate(features):
        ev = evidence[closest][i]
        best_r = int(P[:, i].argmax())
        chart.append({**f.to_dict(), "closest_ref_evidence": ev.to_dict(),
                      "disclosed_in_closest": bool(covered[c, i]),
                      "best_ref": pids[best_r], "best_prob": round(float(P[best_r, i]), 3)})

    if covered[c].all():
        return ClaimVerdict(
            num, independent, "NOT_NOVEL",
            f"All {len(features)} features are disclosed in {closest} "
            f"({ref_titles.get(closest, '')[:80]}).",
            round(1 - anticipation, 3), round(1 - combination, 3), closest, feature_chart=chart)

    dist_idx = [i for i in range(len(features)) if not covered[c, i]]
    distinguishing = [{"feature": features[i].text, "label": _label(evidence[closest][i]),
                       "found_in": [pids[r] for r in range(len(pids)) if r != c and covered[r, i]]}
                      for i in dist_idx]

    linked = []
    if effects and len(effect_emb):
        sims = effect_emb @ feat_emb[dist_idx].T                       # effects x dist
        for e, row in zip(effects, sims):
            if row.max() >= EFFECT_LINK:
                linked.append({**e.to_dict(), "feature": features[dist_idx[int(row.argmax())]].text})

    inventive_score = min(1.0, 1 - combination + (EFFECT_BONUS if linked else 0))
    labels = ", ".join(sorted({d["label"] for d in distinguishing}))
    if all(d["found_in"] for d in distinguishing):
        refs = sorted({r for d in distinguishing for r in d["found_in"]})
        note = " An advancement is described, which may support an argument of unexpected effect." \
            if linked else ""
        return ClaimVerdict(
            num, independent, "OBVIOUS",
            f"Novel over {closest} ({labels}), but the distinguishing features are found in "
            f"{', '.join(refs)}; combining these references likely makes the claim obvious.{note}",
            round(1 - anticipation, 3), round(inventive_score, 3), closest, distinguishing,
            refs, linked, chart)

    missing = [d for d in distinguishing if not d["found_in"]]
    effect_txt = ""
    if linked:
        kinds = sorted({l["kind"] for l in linked})
        effect_txt = f" It also brings a {' and '.join(kinds)} advancement ({linked[0]['category']})."
    return ClaimVerdict(
        num, independent, "INVENTIVE",
        f"{len(missing)} feature(s) not found in any reviewed prior art ({labels}).{effect_txt}",
        round(1 - anticipation, 3), round(inventive_score, 3), closest, distinguishing,
        [], linked, chart)


def judge_patent(verdicts: list[ClaimVerdict]) -> dict:
    indep = [v for v in verdicts if v.independent] or verdicts
    counts = {k: sum(v.verdict == k for v in verdicts) for k in ("NOT_NOVEL", "OBVIOUS", "INVENTIVE")}
    best = max(indep, key=lambda v: v.inventive_score)
    patentable = any(v.verdict == "INVENTIVE" for v in indep)
    fallback = [v.num for v in verdicts if not v.independent and v.verdict == "INVENTIVE"]
    if patentable:
        summary = f"Likely patentable: independent claim {best.num} is inventive over the reviewed prior art."
    elif fallback:
        summary = ("At risk as filed: no independent claim is inventive, but dependent claims "
                   f"{', '.join(map(str, fallback[:6]))} are; amending an independent claim to "
                   "include their features could make it patentable.")
    else:
        summary = "Unlikely patentable as filed: every claim is anticipated or obvious over the reviewed prior art."
    return {"patentable": patentable, "patentability_score": best.inventive_score,
            "novelty_score": max(v.novelty_score for v in indep), "counts": counts,
            "amendment_candidates": [] if patentable else fallback, "summary": summary}
