"""
Error analysis on VALIDATION (out-of-fold, leak-free model without n_refs).
The test set is not used.

For misclassified claims, attribute the error to a pipeline stage using the
examiner's written reason (which reference teaches what, cited paragraphs):

  label/data     reason is not a prior-art ground (double patenting, etc.)
  evidence gap   examiner-cited paragraphs are not in the text we were given
  extraction     claim split into implausible features
  disclosure     our P(disclosed) contradicts the examiner:
                   102 claim but features of the cited reference look undisclosed (miss)
                   allowed claim but every feature looks disclosed (false positive)
  classifier     evidence is consistent with the label, the model still erred

    python -m eval.error_analysis
"""

import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from analysis.elements import parse_claims
from analysis.features import extract_feature
from analysis.outcome_features import FEATURE_NAMES, _p, build
from analysis.outcome_model import CLASSES
from eval import metrics as M
from eval.train_outcome import lgbm

VAL = Path("eval/results/noc4pc_validation.jsonl")
OUT = Path("eval/results/error_analysis.json")
TAU = 0.5


def oof_predictions(ok, drop=("n_refs",)):
    keep = [i for i, f in enumerate(FEATURE_NAMES) if f not in drop]
    X = np.array([[build(r["signals"], r["info"])[f] for f in FEATURE_NAMES] for r in ok])[:, keep]
    y = np.array([CLASSES.index(r["label"]) for r in ok])
    g = np.array([r["app"] for r in ok])
    params = json.loads(Path("analysis/outcome_model/meta.json").read_text())["params"]
    pred, proba = np.zeros(len(y), int), np.zeros((len(y), 3))
    for tr, te in GroupKFold(5).split(X, y, g):
        m = lgbm(params).fit(X[tr], y[tr])
        proba[te] = m.predict_proba(X[te])
        pred[te] = proba[te].argmax(1)
    return pred, proba


def cited_paragraphs(reason: str) -> set[int]:
    keys = set(int(k) for k in re.findall(r"\[\s*0*(\d{1,4})\s*\]", reason))
    keys |= set(int(k) for k in re.findall(r"(?:para(?:graph)?s?\.?|¶)\s*\[?0*(\d{1,4})", reason, re.I))
    return keys


def primary_ref(reason: str, refs: list[str]) -> str | None:
    """First cited reference whose number appears in the examiner's reason."""
    pos = {}
    for r in refs:
        digits = re.sub(r"\D", "", r)
        for cand in (digits, digits[-7:], digits[-8:]):
            i = reason.replace(",", "").replace(" ", "").find(cand) if len(cand) >= 7 else -1
            if i >= 0:
                pos[r] = min(pos.get(r, 1e9), i)
    return min(pos, key=pos.get) if pos else (refs[0] if len(refs) == 1 else None)


def per_ref_p(r):
    """{pid: [best P per feature]} with the engine's default weights."""
    nf = r["n_features"]
    best = defaultdict(lambda: np.zeros(nf))
    for s in r["signals"]:
        if s["feature"] < nf:
            best[s["pid"]][s["feature"]] = max(best[s["pid"]][s["feature"]], _p(s))
    return best


def main():
    recs = M.load(VAL)
    ok = [r for r in recs if r.get("pred")]
    df = pd.read_parquet("eval/data/noc4pc_validation.parquet")
    rows = {(r.application_number, int(r.claim_number)): r for r in df.itertuples()}

    pred, proba = oof_predictions(ok)
    for r, p, pr in zip(ok, pred, proba):
        r["oof"] = CLASSES[p]
        r["oof_p"] = pr.round(3).tolist()
    wrong = [r for r in ok if r["oof"] != r["label"]]
    print(f"OOF macro-F1 (no n_refs): see report below; misclassified {len(wrong)}/{len(ok)}")
    M.report(ok, [r["oof"] for r in ok], title="leak-free OOF on validation")

    # stratified sample of 150 errors by (true, predicted)
    rng = random.Random(0)
    cells = defaultdict(list)
    for r in wrong:
        cells[(r["label"], r["oof"])].append(r)
    sample = []
    for k, v in cells.items():
        rng.shuffle(v)
        sample += v[:max(1, round(150 * len(v) / len(wrong)))]

    findings, cause_counts = [], Counter()
    for r in sample:
        row = rows[(r["app"], r["claim"])]
        ans = json.loads(row.answer)
        reason = ans["reason"]
        refs = json.loads(row.prior_art_specifications)
        ref_ids = [x["patent_id"] for x in refs]
        given_keys = {p.get("key") for x in refs for p in (x.get("paragraphs") or [])
                      if isinstance(p, dict)}
        cited = cited_paragraphs(reason)
        missing = {k for k in cited if k not in given_keys}

        # feature extraction quality for this claim
        ctx = json.loads(row.context)
        claims = {c.num: c for c in parse_claims(" ".join(ctx["claims"]))}
        c = claims.get(r["claim"])
        feats = [extract_feature(e) for e in c.all_elements] if c else []
        long_single = bool(c) and len(c.elements) == 1 and len(c.text.split()) > 80
        no_comp = np.mean([not f.component for f in feats]) if feats else 1.0

        P = per_ref_p(r)
        prim = primary_ref(reason, ref_ids)
        prim_P = P.get(prim) if prim else None
        any_P = np.max(np.stack(list(P.values())), 0) if P else np.zeros(r["n_features"])
        flags = {
            "double_patenting": "double patenting" in reason.lower(),
            "boilerplate_reason": len(reason) < 200,
            "cited_paras": len(cited), "cited_missing": len(missing),
            "long_single_element": long_single, "no_component_frac": round(float(no_comp), 2),
            "n_features": r["n_features"], "primary_ref": prim,
            "primary_cov": None if prim_P is None else round(float((prim_P >= TAU).mean()), 2),
            "primary_min_p": None if prim_P is None else round(float(prim_P.min()), 3),
            "any_cov": round(float((any_P >= TAU).mean()), 2),
        }

        # attribution (first matching cause)
        L, Pd = r["label"], r["oof"]
        if flags["double_patenting"]:
            cause = "label/data: non-prior-art ground"
        elif cited and len(missing) / len(cited) >= 0.5:
            cause = "evidence gap: cited paragraphs not provided"
        elif long_single or no_comp >= 0.5:
            cause = "feature extraction"
        elif L == "NOT_NOVEL" and prim_P is not None and (prim_P < TAU).any():
            cause = "disclosure: missed (102 features look undisclosed)"
        elif L == "OBVIOUS" and (any_P < TAU).any():
            cause = "disclosure: missed (103 features not found in any ref)"
        elif L == "INVENTIVE" and (any_P >= TAU).all():
            cause = "disclosure: false positive (allowed, all features 'found')"
        else:
            cause = "classifier/decision"
        cause_counts[cause] += 1
        findings.append({"app": r["app"], "claim": r["claim"], "label": L, "pred": Pd,
                         "proba": r["oof_p"], "cause": cause, **flags,
                         "reason": reason[:600], "claim_text": c.text[:400] if c else ""})

    print(f"\n=== attribution of {len(sample)} sampled errors ===")
    for k, v in cause_counts.most_common():
        print(f"  {v:4d}  {v / len(sample):5.1%}  {k}")
    by_cell = Counter((f["label"], f["pred"], f["cause"]) for f in findings)
    print("\nby confusion cell:")
    for (l, p, cz), v in sorted(by_cell.items(), key=lambda x: -x[1])[:14]:
        print(f"  {l:10} -> {p:10} {v:3d}  {cz}")

    # population-level disclosure check (all validation, not only errors)
    stats = defaultdict(list)
    for r in ok:
        row = rows[(r["app"], r["claim"])]
        reason = json.loads(row.answer)["reason"]
        P = per_ref_p(r)
        if not P:
            continue
        prim = primary_ref(reason, list(P))
        anyP = np.max(np.stack(list(P.values())), 0)
        stats[r["label"] + "_any_cov"].append((anyP >= TAU).mean())
        if prim in P:
            stats[r["label"] + "_prim_full"].append(float((P[prim] >= TAU).all()))
            stats[r["label"] + "_prim_cov"].append((P[prim] >= TAU).mean())
    print("\npopulation (all validation claims):")
    for k in sorted(stats):
        print(f"  {k:22} mean={np.mean(stats[k]):.3f}  n={len(stats[k])}")

    OUT.write_text(json.dumps({"counts": cause_counts, "findings": findings,
                               "population": {k: float(np.mean(v)) for k, v in stats.items()}},
                              indent=1, default=str))
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
