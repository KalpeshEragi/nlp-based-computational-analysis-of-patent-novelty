"""
Train + evaluate the examiner-outcome predictor against the rule-based engine.

Protocol
  * Splits are PANORAMA's own, which are disjoint by application (checked).
  * Everything is tuned on VALIDATION only, with 5-fold GroupKFold over
    applications (claims of one application never span train and held-out).
  * The final models are fit on all of validation and evaluated ONCE on TEST.
    A marker file blocks a second test run (use --force only to regenerate
    the same report, never to tune).

    python -m eval.train_outcome                 # CV on validation, fit, save model
    python -m eval.train_outcome --final-test    # one-time test evaluation + report
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

import eval.metrics as M
from analysis.engine import THRESHOLD
from analysis.disclosure import DEFAULT_WEIGHTS
from analysis.outcome_features import FEATURE_NAMES, build
from analysis.outcome_model import CLASSES, MODEL_DIR
from eval.tune import search as rule_search

RES = Path("eval/results")
VAL, TEST = RES / "noc4pc_validation.jsonl", RES / "noc4pc_test.jsonl"
MARKER = RES / ".test_evaluated"


def dataset(path):
    recs = M.load(path)
    ok = [r for r in recs if r.get("pred")]
    X = np.array([[build(r["signals"], r["info"])[n] for n in FEATURE_NAMES] for r in ok])
    y = np.array([CLASSES.index(r["label"]) for r in ok])
    g = np.array([r["app"] for r in ok])
    return recs, ok, X, y, g


def lgbm(params, seed=0):
    import lightgbm as lgb
    return lgb.LGBMClassifier(objective="multiclass", class_weight="balanced", random_state=seed,
                              verbose=-1, **params)


def sample_params(rng):
    return {"n_estimators": rng.choice([100, 200, 300, 500]),
            "learning_rate": rng.choice([0.02, 0.03, 0.05, 0.1]),
            "num_leaves": rng.choice([7, 15, 31]),
            "min_child_samples": rng.choice([20, 40, 80]),
            "colsample_bytree": rng.choice([0.6, 0.8, 1.0]),
            "subsample": rng.choice([0.7, 0.9, 1.0]), "subsample_freq": 1,
            "reg_lambda": rng.choice([0.0, 1.0, 5.0])}


def cv_gbm(X, y, g, params, folds):
    pred = np.zeros(len(y), int)
    for tr, te in folds:
        pred[te] = lgbm(params).fit(X[tr], y[tr]).predict(X[te])
    return f1_score(y, pred, average="macro"), pred


def cv_rule(ok, folds, iters):
    preds = [None] * len(ok)
    for k, (tr, te) in enumerate(folds):
        _, w, t, to = rule_search([ok[i] for i in tr], iters, seed=k + 1)
        for i in te:
            preds[i] = M.rescore(ok[i], w, t, obv_threshold=to)[0]
    y = [r["label"] for r in ok]
    return f1_score(y, preds, labels=CLASSES, average="macro")


def train(args):
    recs, ok, X, y, g = dataset(VAL)
    print(f"validation: {len(ok)} claims ({len(recs) - len(ok)} failed), {len(set(g))} applications")
    folds = list(GroupKFold(n_splits=5).split(X, y, g))

    # rule baseline: defaults (no tuning) and tuned-in-CV
    y_lab = [r["label"] for r in ok]
    rule_default = f1_score(y_lab, [M.rescore(r)[0] for r in ok], labels=CLASSES, average="macro")
    rule_cv = cv_rule(ok, folds, args.rule_iters)
    print(f"rule engine  defaults macroF1={rule_default:.3f} | tuned (grouped CV) macroF1={rule_cv:.3f}")

    # GBM hyperparameter search with grouped CV
    rng = random.Random(0)
    best = (-1, None)
    for i in range(args.trials):
        p = sample_params(rng)
        f, _ = cv_gbm(X, y, g, p, folds)
        if f > best[0]:
            best = (f, p)
            print(f"  trial {i:2d} new best grouped-CV macroF1={f:.3f} {p}")
    f_cv, pred_cv = cv_gbm(X, y, g, best[1], folds)
    print(f"GBM grouped-CV macroF1={f_cv:.3f}")
    M.report(ok, [CLASSES[p] for p in pred_cv], title="GBM, out-of-fold on validation")

    # final fits on ALL validation
    _, w, t, to = rule_search(ok, args.rule_iters)
    model = lgbm(best[1]).fit(X, y)
    MODEL_DIR.mkdir(exist_ok=True)
    model.booster_.save_model(str(MODEL_DIR / "model.txt"))
    meta = {"version": time.strftime("outcome-gbm-%Y%m%d"), "classes": CLASSES,
            "features": FEATURE_NAMES, "params": best[1],
            "trained_on": "PANORAMA NOC4PC validation (all)", "n_train": len(ok),
            "cv": {"gbm_macro_f1": f_cv, "rule_tuned_macro_f1": rule_cv,
                   "rule_default_macro_f1": rule_default},
            "rule_tuned": {"weights": w, "threshold": t, "obv_threshold": to}}
    (MODEL_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"saved model -> {MODEL_DIR}")


def final_test(args):
    if MARKER.exists() and not args.force:
        raise SystemExit("test set already evaluated once; refusing to evaluate again "
                         "(--force only regenerates the same report)")
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score

    meta = json.loads((MODEL_DIR / "meta.json").read_text())
    booster = lgb.Booster(model_file=str(MODEL_DIR / "model.txt"))
    recs, ok, X, y, g = dataset(TEST)
    y_lab = [r["label"] for r in ok]
    rt = meta["rule_tuned"]

    out = {"n_test_claims": len(recs), "n_scored": len(ok), "n_failed": len(recs) - len(ok),
           "n_applications": len(set(g)), "validation_cv": meta["cv"], "models": {}}

    # baselines
    maj = CLASSES[int(np.bincount([CLASSES.index(l) for l in [r["label"] for r in M.load(VAL) if r.get("pred")]]).argmax())]
    out["models"]["majority_class"] = M.report(ok, [maj] * len(ok), title=f"majority class ({maj})")
    rnd = np.random.default_rng(0).choice(CLASSES, len(ok))
    out["models"]["random"] = M.report(ok, list(rnd), title="random")
    out["models"]["rule_default"] = M.evaluate(ok, None, THRESHOLD, title="rule engine (default)")
    out["models"]["rule_tuned"] = M.evaluate(ok, rt["weights"], rt["threshold"],
                                             title="rule engine (tuned on validation)",
                                             obv_threshold=rt["obv_threshold"])
    P = booster.predict(X)
    pred = [CLASSES[i] for i in P.argmax(1)]
    gb = M.report(ok, pred, title="learned outcome model (GBM)")
    gb["roc_auc_ovr"] = {c: roc_auc_score(y == k, P[:, k]) for k, c in enumerate(CLASSES)}
    gb["roc_auc_macro"] = float(np.mean(list(gb["roc_auc_ovr"].values())))
    print("   ROC-AUC one-vs-rest:", {k: round(v, 3) for k, v in gb["roc_auc_ovr"].items()},
          f"macro={gb['roc_auc_macro']:.3f}")
    out["models"]["gbm"] = gb

    # explanations: SHAP (test set) + LightGBM gain importance
    import shap
    sv = shap.TreeExplainer(booster).shap_values(X)
    sv = np.stack(sv, -1) if isinstance(sv, list) else sv          # (n, features, classes)
    mean_abs = np.abs(sv).mean(0)                                   # (features, classes)
    gain = booster.feature_importance("gain")
    out["shap"] = {"mean_abs_by_class": {c: dict(sorted(
        {f: float(mean_abs[i, k]) for i, f in enumerate(FEATURE_NAMES)}.items(),
        key=lambda x: -x[1])) for k, c in enumerate(CLASSES)},
        "mean_abs_overall": dict(sorted({f: float(mean_abs[i].sum()) for i, f in enumerate(FEATURE_NAMES)}.items(),
                                        key=lambda x: -x[1])),
        "gain_importance": dict(sorted({f: float(v) for f, v in zip(FEATURE_NAMES, gain)}.items(),
                                       key=lambda x: -x[1]))}
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for k, c in enumerate(CLASSES):
            plt.figure()
            shap.summary_plot(sv[:, :, k], X, feature_names=FEATURE_NAMES, show=False, max_display=15)
            plt.title(f"SHAP – {c}")
            plt.tight_layout()
            plt.savefig(RES / f"shap_{c.lower()}.png", dpi=120)
            plt.close("all")
    except Exception as e:  # plotting is optional
        print("plot skipped:", e)

    (RES / "final_test_report.json").write_text(json.dumps(out, indent=2, default=float))
    MARKER.write_text(time.strftime("%Y-%m-%d %H:%M"))
    print("\nTop features by mean |SHAP| (all classes):")
    for f, v in list(out["shap"]["mean_abs_overall"].items())[:10]:
        print(f"   {f:18} {v:.3f}")
    print(f"report -> {RES / 'final_test_report.json'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--final-test", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--rule-iters", type=int, default=300)
    a = ap.parse_args()
    final_test(a) if a.final_test else train(a)
