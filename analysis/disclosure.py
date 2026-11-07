"""
Disclosure check: does a prior-art document actually disclose a claim feature?

For each (feature, reference) we take the reference's 3 most similar passages
(claim elements + abstract/summary sentences) and score each with four signals:

    lexical    share of the feature's content words present in the passage
    syntactic  same component head noun / same function verb in the passage
    semantic   PatentSBERTa cosine similarity
    entailment NLI model: does the passage *state* the feature? (not just topic)

The signals are combined by a logistic model into P(disclosed). Weights start
from sensible defaults and are re-fitted on examiner data by the evaluation.
Numeric parameters are checked separately: a conflicting range means the
feature is not disclosed even if the text is otherwise similar.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from analysis.elements import parse_claims
from analysis.features import Feature, _nlp, content_lemmas, extract_params

NLI_MODEL = "cross-encoder/nli-deberta-v3-base"
DEFAULT_WEIGHTS = {"bias": -5.0, "nli": 4.0, "sem": 4.0, "lex": 2.0, "syn": 1.0}
WEIGHTS_FILE = Path(__file__).with_name("disclosure_weights.json")
NLI_GATE_LO, NLI_GATE_HI = 0.45, 0.75   # semantic range over which NLI is trusted


@dataclass
class Passage:
    text: str
    source: str            # "claim 3" / "abstract" / "summary"
    lemmas: set = field(default_factory=set)
    params: list = field(default_factory=list)


@dataclass
class Evidence:
    prob: float                 # P(feature disclosed by this reference)
    passage: str
    source: str
    signals: dict
    component_found: bool
    param_conflict: bool

    def to_dict(self) -> dict:
        return {"prob": round(self.prob, 3), "passage": self.passage, "source": self.source,
                "signals": {k: round(v, 3) for k, v in self.signals.items()},
                "component_found": self.component_found, "param_conflict": self.param_conflict}


def _sentences(text: str, max_n: int = 40) -> list[str]:
    parts = re.split(r"(?<=[.;])\s+(?=[A-Z])", text or "")
    return [p.strip() for p in parts if 6 <= len(p.split()) <= 120][:max_n]


class DisclosureChecker:
    def __init__(self, encoder, nli_model: str = NLI_MODEL, top_passages: int = 3):
        import torch
        from sentence_transformers import CrossEncoder

        self.encoder = encoder
        self.nli = CrossEncoder(nli_model, max_length=256)
        if torch.cuda.is_available():
            self.nli.model.half()
        labels = self.nli.model.config.id2label
        self.ent_idx = next(i for i, l in labels.items() if l.lower() == "entailment")
        self.top_passages = top_passages
        self.weights = dict(DEFAULT_WEIGHTS)
        if WEIGHTS_FILE.exists():
            self.weights.update(json.loads(WEIGHTS_FILE.read_text()))
        self._refs: dict[str, tuple[list[Passage], np.ndarray]] = {}
        self._nli_cache: dict[tuple[str, str], float] = {}

    # -- encoding -----------------------------------------------------------
    def encode(self, texts: list[str], max_len: int = 128) -> np.ndarray:
        self.encoder.max_seq_length = max_len
        return self.encoder.encode(texts, batch_size=64, normalize_embeddings=True,
                                   convert_to_numpy=True, show_progress_bar=False).astype(np.float32)

    def load_reference(self, pid: str, doc: dict):
        """Split a reference into passages and embed them (once per session)."""
        if pid in self._refs:
            return
        passages = [Passage(e, f"claim {c.num}") for c in parse_claims(doc.get("claims", ""))
                    for e in c.elements]
        passages += [Passage(s, "abstract") for s in _sentences(doc.get("abstract", ""))]
        passages += [Passage(s, "summary") for s in _sentences(doc.get("summary", ""))]
        if not passages:
            passages = [Passage(doc.get("title", "") or "(empty)", "title")]
        for p, d in zip(passages, _nlp().pipe(p.text for p in passages)):
            p.lemmas, p.params = content_lemmas(d), extract_params(p.text)
        self._refs[pid] = (passages, self.encode([p.text for p in passages]))

    # -- scoring ------------------------------------------------------------
    def _nli(self, pairs: list[tuple[str, str]]) -> list[float]:
        todo = [p for p in dict.fromkeys(pairs) if p not in self._nli_cache]
        if todo:
            probs = self.nli.predict(todo, batch_size=32, apply_softmax=True,
                                     show_progress_bar=False)
            for p, pr in zip(todo, probs):
                self._nli_cache[p] = float(pr[self.ent_idx])
        return [self._nli_cache[p] for p in pairs]

    def _prob(self, s: dict) -> float:
        w = self.weights
        z = w["bias"] + sum(w[k] * s[k] for k in ("nli", "sem", "lex", "syn"))
        return 1 / (1 + math.exp(-z))

    def check(self, features: list[Feature], feat_emb: np.ndarray, comp_emb: np.ndarray,
              pids: list[str]) -> dict[str, list[Evidence]]:
        """Evidence for every feature against every reference: {pid: [Evidence per feature]}."""
        jobs = []  # (pid, feature idx, passage idx, cosine)
        for pid in pids:
            passages, pemb = self._refs[pid]
            sims = feat_emb @ pemb.T
            k = min(self.top_passages, len(passages))
            for i in range(len(features)):
                for j in np.argsort(-sims[i])[:k]:
                    jobs.append((pid, i, int(j), float(sims[i, j])))

        # long fragments make the NLI model unreliable: use the first ~40 words
        hyp = [" ".join(f.text.split()[:40]) for f in features]
        nli = self._nli([(self._refs[pid][0][j].text, hyp[i]) for pid, i, j, _ in jobs])

        best: dict[tuple[str, int], Evidence] = {}
        for (pid, i, j, sem), ent in zip(jobs, nli):
            f, (passages, pemb) = features[i], self._refs[pid]
            p = passages[j]
            lex = len(f.lemmas & p.lemmas) / max(len(f.lemmas), 1)
            syn = 0.5 * (f.component_head in p.lemmas) + \
                0.5 * any(v in p.lemmas for v in f.function_verbs[:3])
            # entailment only counts when the passage is on the same topic
            gate = min(1.0, max(0.0, (sem - NLI_GATE_LO) / (NLI_GATE_HI - NLI_GATE_LO)))
            sig = {"nli": ent * gate, "sem": sem, "lex": lex, "syn": syn, "nli_raw": ent}
            prob = self._prob(sig)
            conflict = False
            if f.params:  # numbers must agree when the feature is defined by them
                same_unit = [q for q in p.params if any(q.unit == fp.unit for fp in f.params)]
                if same_unit and not any(fp.overlaps(q) for fp in f.params for q in same_unit):
                    conflict, prob = True, prob * 0.2
                elif not same_unit:
                    prob *= 0.5
            comp_found = bool(f.component_head) and (
                f.component_head in p.lemmas or float(comp_emb[i] @ pemb[j]) >= 0.55)
            ev = Evidence(prob, p.text, p.source, sig, comp_found, conflict)
            if (pid, i) not in best or ev.prob > best[(pid, i)].prob:
                best[(pid, i)] = ev
        return {pid: [best[(pid, i)] for i in range(len(features))] for pid in pids}
