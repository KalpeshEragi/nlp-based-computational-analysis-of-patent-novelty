"""
Step 5 - CPC prediction for the app (Layer 1).

    from cpc_classifier.predict import CPCPredictor
    p = CPCPredictor("cpc_classifier/model_v2")
    p.predict(title=..., abstract=..., claims=..., summary=..., k=5)
    -> [{"code": "H04N", "prob": 0.93}, ...]
"""

import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from cpc_classifier.train import MODEL_NAME, CPCClassifier
from retrieval.claims import split_claims


class CPCPredictor:
    def __init__(self, model_dir="cpc_classifier/model_v2", device=None, max_len=384):
        d = Path(model_dir)
        self.labels = json.loads((d / "labels.json").read_text())
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        # tokenizer is the stock PatentSBERTa one; use a local copy if present
        tok_src = d if (d / "tokenizer.json").exists() else MODEL_NAME
        self.tok = AutoTokenizer.from_pretrained(tok_src)
        self.model = CPCClassifier(len(self.labels))
        self.model.load_state_dict(torch.load(d / "model.pt", map_location="cpu"))
        self.model.to(self.device).eval()
        self.max_len = max_len

    @staticmethod
    def build_text(title="", abstract="", claims="", summary="") -> str:
        """Same input layout as training: title . abstract . first claim . summary"""
        parsed = split_claims(claims)
        first = parsed[0]["text"] if parsed else claims
        return " . ".join(p for p in (title, abstract, first, summary) if p)

    @torch.no_grad()
    def predict(self, title="", abstract="", claims="", summary="", k=5) -> list[dict]:
        text = self.build_text(title, abstract, claims, summary)
        enc = self.tok([text], truncation=True, max_length=self.max_len,
                       return_tensors="pt").to(self.device)
        probs = torch.sigmoid(self.model(**enc).float())[0]
        top = torch.topk(probs, k)
        return [{"code": self.labels[i], "prob": round(float(p), 4)}
                for p, i in zip(top.values.tolist(), top.indices.tolist())]
