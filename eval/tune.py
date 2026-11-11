"""
Tune disclosure weights and decision thresholds on VALIDATION results only,
then save them for the engine (analysis/disclosure_weights.json).

Random search over the 4 signal weights + bias + two thresholds, maximising
macro-F1. The best setting is re-checked with 5-fold cross-validation over
applications to make sure it is not just fitting noise.

    python -m eval.tune --results eval/results/noc4pc_validation.jsonl
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

from analysis.disclosure import DEFAULT_WEIGHTS, WEIGHTS_FILE
from eval.metrics import evaluate, load, report

import eval.metrics as M


def macro_f1(recs, w, t, t_obv):
    from sklearn.metrics import f1_score
    ok = [r for r in recs if r.get("pred")]
    preds = [M.rescore_t(r, w, t, t_obv)[0] for r in ok]
    return f1_score([r["label"] for r in ok], preds, labels=M.CLASSES, average="macro")


def sample(rng):
    return ({"bias": rng.uniform(-9, -2), "dis": rng.uniform(0, 10), "sem": rng.uniform(1, 10),
             "lex": rng.uniform(0, 5), "syn": rng.uniform(0, 3)},
            rng.uniform(0.3, 0.95), rng.uniform(0.2, 0.9))


def search(recs, n_iter, seed=0):
    rng = random.Random(seed)
    best = (macro_f1(recs, DEFAULT_WEIGHTS, 0.5, 0.5), DEFAULT_WEIGHTS, 0.5, 0.5)
    for it in range(n_iter):
        w, t, to = sample(rng)
        if it > n_iter // 2:  # second half: refine around the best so far
            bw, bt, bto = best[1], best[2], best[3]
            w = {k: bw[k] + rng.gauss(0, 0.6) for k in bw}
            t, to = min(0.97, max(0.2, bt + rng.gauss(0, 0.05))), min(0.95, max(0.1, bto + rng.gauss(0, 0.05)))
        f = macro_f1(recs, w, t, to)
        if f > best[0]:
            best = (f, w, t, to)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="eval/results/noc4pc_validation.jsonl")
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()

    recs = [r for r in load(args.results) if r.get("pred")]
    report(recs, title="defaults (engine as run)")

    f, w, t, to = search(recs, args.iters)
    print(f"\nbest on all validation: macroF1={f:.3f} threshold={t:.3f} obv_threshold={to:.3f}")
    print("weights:", {k: round(v, 3) for k, v in w.items()})
    M.evaluate_t(recs, w, t, to, title="tuned (validation, in-sample)")

    # 5-fold CV over applications: tune on 4 folds, score on the held-out fold
    apps = sorted({r["app"] for r in recs})
    random.Random(0).shuffle(apps)
    folds = [set(apps[i::5]) for i in range(5)]
    cv = []
    for k, fold in enumerate(folds):
        tr = [r for r in recs if r["app"] not in fold]
        te = [r for r in recs if r["app"] in fold]
        _, wk, tk, tok = search(tr, args.iters // 2, seed=k + 1)
        cv.append(macro_f1(te, wk, tk, tok))
    print(f"\n5-fold CV macro-F1 (held-out apps): {np.mean(cv):.3f} ± {np.std(cv):.3f}  {[round(x, 3) for x in cv]}")

    if args.save:
        WEIGHTS_FILE.write_text(json.dumps({**{k: round(v, 4) for k, v in w.items()},
                                            "threshold": round(t, 4), "obv_threshold": round(to, 4)},
                                           indent=2))
        print(f"saved -> {WEIGHTS_FILE}")


if __name__ == "__main__":
    main()
