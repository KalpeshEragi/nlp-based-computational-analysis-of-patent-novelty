"""
Metrics + offline re-scoring for NOC4PC results.

Re-computes P(disclosed) from the saved raw signals with any weights /
threshold, applies the same decision logic as the engine (analysis.engine.decide)
and reports accuracy, macro-F1, per-class precision/recall, confusion matrix
and ROC-AUC of the scores.
"""

import json
import math
from collections import defaultdict

import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support, roc_auc_score)

from analysis.disclosure import DEFAULT_WEIGHTS
from analysis.engine import THRESHOLD, decide

CLASSES = ["NOT_NOVEL", "OBVIOUS", "INVENTIVE"]


def load(path):
    return [json.loads(l) for l in open(path)]


def _arrays(rec):
    """Raw signals of one claim as numpy arrays (cached on the record)."""
    if "_arr" not in rec:
        sig = rec["signals"]
        pids = sorted({s["pid"] for s in sig})
        nf = rec["n_features"]
        col = {k: np.array([s[k] for s in sig], np.float64) for k in ("sem", "dis", "lex", "syn", "pf")}
        cell = np.array([pids.index(s["pid"]) * nf + s["feature"] for s in sig], int)
        rec["_arr"] = (col, cell, len(pids), nf)
    return rec["_arr"]


def rescore(rec, w=None, threshold=None, obv_threshold=None):
    """Verdict + scores for one claim from its raw signals."""
    w = w or DEFAULT_WEIGHTS
    col, cell, n_refs, nf = _arrays(rec)
    if n_refs == 0 or nf == 0:
        return decide(np.zeros((0, 0)), threshold, obv_threshold)
    z = w["bias"] + w["dis"] * col["dis"] + w["sem"] * col["sem"] + w["lex"] * col["lex"] + w["syn"] * col["syn"]
    p = col["pf"] / (1 + np.exp(-z))
    best = np.zeros(n_refs * nf)
    np.maximum.at(best, cell, p)
    return decide(best.reshape(n_refs, nf), threshold, obv_threshold)


def rescore_t(rec, w, threshold, obv_threshold):
    return rescore(rec, w, threshold, obv_threshold=obv_threshold)


def report(recs, preds=None, scores=None, title=""):
    ok = [r for r in recs if r.get("pred")]
    y = [r["label"] for r in ok]
    p = preds if preds is not None else [r["pred"] for r in ok]
    pr, rc, f1, sup = precision_recall_fscore_support(y, p, labels=CLASSES, zero_division=0)
    out = {"n": len(ok), "failed": len(recs) - len(ok),
           "accuracy": accuracy_score(y, p), "macro_f1": f1_score(y, p, labels=CLASSES, average="macro"),
           "per_class": {c: {"precision": pr[i], "recall": rc[i], "f1": f1[i], "support": int(sup[i])}
                         for i, c in enumerate(CLASSES)},
           "confusion": {"labels": CLASSES, "matrix": confusion_matrix(y, p, labels=CLASSES).tolist()}}
    if scores is not None:
        nov, inv = scores
        out["auc_not_novel_vs_rest"] = roc_auc_score([l == "NOT_NOVEL" for l in y], [-s for s in nov])
        out["auc_allow_vs_rest"] = roc_auc_score([l == "INVENTIVE" for l in y], inv)
    if title:
        print(f"\n== {title}: n={out['n']} acc={out['accuracy']:.3f} macroF1={out['macro_f1']:.3f}"
              + (f" AUC(102)={out['auc_not_novel_vs_rest']:.3f} AUC(allow)={out['auc_allow_vs_rest']:.3f}"
                 if scores is not None else ""))
        for c, m in out["per_class"].items():
            print(f"   {c:10} P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} n={m['support']}")
        print("   confusion (rows=true):", out["confusion"]["matrix"])
    return out


def evaluate(recs, w=None, threshold=THRESHOLD, title="", obv_threshold=None):
    ok = [r for r in recs if r.get("pred")]
    res = [rescore(r, w, threshold, obv_threshold=obv_threshold) for r in ok]
    preds = [v for v, _, _ in res]
    nov = [1 - a for _, a, _ in res]
    inv = [1 - c for _, _, c in res]
    return report(recs, preds, (nov, inv), title)


def evaluate_t(recs, w, threshold, obv_threshold, title=""):
    return evaluate(recs, w, threshold, title, obv_threshold)
