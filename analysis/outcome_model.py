"""
Examiner-outcome predictor.

A gradient-boosted classifier (LightGBM) trained on USPTO examiner decisions
(PANORAMA NOC4PC) that reads the outcome features of a claim's evidence and
predicts how an examiner would likely rule: rejected under §102 (not novel),
rejected under §103 (obvious), or allowed.

It is deliberately separate from the legal analysis (analysis/engine.py): the
engine produces the evidence-based novelty / inventive-step reasoning; this
model only estimates the likely examiner outcome from that evidence. It never
changes the engine's verdicts.

Trained by eval/train_outcome.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from analysis.outcome_features import FEATURE_NAMES, build

MODEL_DIR = Path(__file__).with_name("outcome_model")
CLASSES = ["NOT_NOVEL", "OBVIOUS", "INVENTIVE"]
EXAMINER_LABEL = {"NOT_NOVEL": "rejected §102 (not novel)",
                  "OBVIOUS": "rejected §103 (obvious)", "INVENTIVE": "allowed"}


class OutcomePredictor:
    def __init__(self, booster, meta: dict):
        self.booster = booster
        self.meta = meta

    @classmethod
    def load_if_available(cls, model_dir: Path = MODEL_DIR):
        f = Path(model_dir) / "model.txt"
        if not f.exists():
            return None
        import lightgbm as lgb
        meta = json.loads((Path(model_dir) / "meta.json").read_text())
        return cls(lgb.Booster(model_file=str(f)), meta)

    def predict(self, signals: list[dict], info: dict, top_k: int = 3) -> dict:
        feats = build(signals, info)
        x = np.array([[feats[n] for n in FEATURE_NAMES]], dtype=float)
        probs = self.booster.predict(x)[0]
        k = int(np.argmax(probs))
        # per-claim explanation: SHAP contributions toward the predicted class
        contrib = self.booster.predict(x, pred_contrib=True)[0]
        n = len(FEATURE_NAMES) + 1
        cls_contrib = contrib[k * n:(k + 1) * n - 1]
        order = np.argsort(-np.abs(cls_contrib))[:top_k]
        return {
            "predicted": CLASSES[k],
            "examiner_outcome": EXAMINER_LABEL[CLASSES[k]],
            "probabilities": {c: round(float(p), 3) for c, p in zip(CLASSES, probs)},
            "top_factors": [{"feature": FEATURE_NAMES[i], "value": round(float(x[0, i]), 3),
                             "contribution": round(float(cls_contrib[i]), 3)} for i in order],
            "model": self.meta.get("version", ""),
        }
