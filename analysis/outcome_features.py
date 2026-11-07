"""
Outcome features: a fixed vector describing one claim's evidence, used by the
examiner-outcome predictor (analysis/outcome_model.py).

These are computed FROM the legal analysis (feature chart + disclosure
signals); they do not change it. The same function is used in evaluation and
at runtime so the model always sees identical inputs.

Groups (30 features):
  claim structure   breadth / dependency of the claim
  disclosure        anticipation / combination / coverage from P(disclosed)
  raw signals       lexical, syntactic, semantic, NLI of the best evidence
  evidence detail   component found, parameter conflicts, advancement links
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from analysis.disclosure import DEFAULT_WEIGHTS, NLI_GATE_HI, NLI_GATE_LO

FEATURE_NAMES = [
    # claim structure (7)
    "n_features", "n_own_elements", "n_inherited", "is_independent", "depth",
    "claim_words", "n_params",
    # references (1)
    "n_refs",
    # disclosure, from P(disclosed) with the engine's default weights (10)
    "anticipation", "combination", "mean_best_p", "closest_mean_p",
    "closest_cov_30", "closest_cov_50", "closest_cov_70",
    "any_cov_30", "any_cov_50", "any_cov_70",
    # raw signals of each feature's best evidence (8)
    "sem_mean", "sem_min", "nli_mean", "nli_min", "lex_mean", "lex_min", "syn_mean", "syn_min",
    # evidence detail (4)
    "comp_found_frac", "param_conflicts", "linked_effects", "uncovered_any_50",
]


def _p(s: dict, w=DEFAULT_WEIGHTS) -> float:
    g = min(1.0, max(0.0, (s["sem"] - NLI_GATE_LO) / (NLI_GATE_HI - NLI_GATE_LO)))
    z = w["bias"] + w["nli"] * s["nli_raw"] * g + w["sem"] * s["sem"] + \
        w["lex"] * s["lex"] + w["syn"] * s["syn"]
    return s["pf"] / (1 + math.exp(-z))


def build(signals: list[dict], info: dict) -> dict:
    """signals: DisclosureChecker records for one claim; info: claim structure."""
    nf = max(int(info.get("n_features", 0)), 1)
    pids = sorted({s["pid"] for s in signals})
    P = np.zeros((max(len(pids), 1), nf))
    best = {k: np.zeros(nf) for k in ("sem", "nli_raw", "lex", "syn")}
    comp = np.zeros(nf)
    conflicts = set()
    for s in signals:
        r, i = pids.index(s["pid"]), s["feature"]
        if i >= nf:
            continue
        P[r, i] = max(P[r, i], _p(s))
        for k in best:
            best[k][i] = max(best[k][i], s[k])
        comp[i] = max(comp[i], float(s.get("comp_found", 0)))
        if s["pf"] < 0.5:
            conflicts.add(i)

    if not pids:
        P[:] = 0
    c = int(np.argmax([(P[r] >= 0.5).sum() + P[r].mean() / 10 for r in range(P.shape[0])]))
    anyP = P.max(axis=0)
    f = {
        "n_features": nf, "n_own_elements": info.get("n_own", nf),
        "n_inherited": info.get("n_inherited", 0), "is_independent": int(info.get("independent", 0)),
        "depth": info.get("depth", 0), "claim_words": info.get("claim_words", 0),
        "n_params": info.get("n_params", 0), "n_refs": len(pids),
        "anticipation": float(P.min(axis=1).max()), "combination": float(anyP.min()),
        "mean_best_p": float(anyP.mean()), "closest_mean_p": float(P[c].mean()),
        "closest_cov_30": float((P[c] >= .3).mean()), "closest_cov_50": float((P[c] >= .5).mean()),
        "closest_cov_70": float((P[c] >= .7).mean()),
        "any_cov_30": float((anyP >= .3).mean()), "any_cov_50": float((anyP >= .5).mean()),
        "any_cov_70": float((anyP >= .7).mean()),
        "sem_mean": float(best["sem"].mean()), "sem_min": float(best["sem"].min()),
        "nli_mean": float(best["nli_raw"].mean()), "nli_min": float(best["nli_raw"].min()),
        "lex_mean": float(best["lex"].mean()), "lex_min": float(best["lex"].min()),
        "syn_mean": float(best["syn"].mean()), "syn_min": float(best["syn"].min()),
        "comp_found_frac": float(comp.mean()), "param_conflicts": len(conflicts),
        "linked_effects": info.get("linked_effects", 0),
        "uncovered_any_50": int((anyP < .5).sum()),
    }
    return {k: f[k] for k in FEATURE_NAMES}


def claim_info(claim, features, linked_effects: int) -> dict:
    """Structure of one claim (analysis.elements.ClaimElements)."""
    return {"n_features": len(claim.all_elements), "n_own": len(claim.elements),
            "n_inherited": len(claim.inherited), "independent": claim.independent,
            "depth": claim.depth, "claim_words": len(claim.text.split()),
            "n_params": sum(len(f.params) for f in features),
            "linked_effects": linked_effects}
