"""
Disclosure check: does a prior-art document actually disclose a claim feature?

For each (feature, reference) we take the reference's 3 most similar passages
(claim elements + abstract/summary sentences) and score each with four signals:

    lexical    share of the feature's content words present in the passage
    syntactic  same component head noun / same function verb in the passage
    semantic   PatentSBERTa cosine similarity
    disclosure patent-specific cross-encoder trained on examiner citations
               (PANORAMA PI4PC): does this passage disclose this claim text?

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

NLI_MODEL = "cross-encoder/nli-deberta-v3-base"          # only for advancement detection
DISCLOSURE_MODEL = str(Path(__file__).resolve().parent.parent / "disclosure_model" / "ce_patentsberta")
DEFAULT_WEIGHTS = {"bias": -4.0, "dis": 6.0, "sem": 2.0, "lex": 1.5, "syn": 0.5}
WEIGHTS_FILE = Path(__file__).with_name("disclosure_weights.json")
Q_WORDS, P_WORDS = 96, 220   # same clipping as the disclosure model's training


def p_disclosed(s: dict, w: dict = DEFAULT_WEIGHTS) -> float:
    """P(feature disclosed by a passage) from its raw signals (shared everywhere)."""
    z = w["bias"] + w["dis"] * s["dis"] + w["sem"] * s["sem"] + w["lex"] * s["lex"] + w["syn"] * s["syn"]
    return s.get("pf", 1.0) / (1 + math.exp(-z))


def _clip(text: str, words: int) -> str:
    return " ".join(text.split()[:words])


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
    def __init__(self, encoder, disclosure_model: str = DISCLOSURE_MODEL,
                 nli_model: str = NLI_MODEL, top_passages: int = 3):
        import torch
        from sentence_transformers import CrossEncoder

        self.encoder = encoder
        self.cuda = torch.cuda.is_available()
        self.dis_model = CrossEncoder(disclosure_model, num_labels=1, max_length=384)
        if self.cuda:
            self.dis_model.model.half()
        self._nli_name, self._nli_model = nli_model, None
        self._dis_cache: dict[tuple[str, str], float] = {}
        self.top_passages = top_passages
        self.weights = dict(DEFAULT_WEIGHTS)
        if WEIGHTS_FILE.exists():
            self.weights.update(json.loads(WEIGHTS_FILE.read_text()))
        self._refs: dict[str, tuple[list[Passage], np.ndarray]] = {}
        self.records: list | None = None   # set to [] to log raw signals (evaluation)
        self._nli_cache: dict[tuple[str, str], float] = {}
        self.ent_idx = None

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
        desc = doc.get("description") or ""
        if isinstance(desc, (list, tuple)):
            desc = " ".join(map(str, desc))
        passages += [Passage(s, "description") for s in _sentences(desc, 250)]
        if not passages:
            passages = [Passage(doc.get("title", "") or "(empty)", "title")]
        for p, d in zip(passages, _nlp().pipe(p.text for p in passages)):
            p.lemmas, p.params = content_lemmas(d), extract_params(p.text)
        self._refs[pid] = (passages, self.encode([p.text for p in passages]))

    # -- scoring ------------------------------------------------------------
    def _dis(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Disclosure-model probability for (claim text, passage) pairs, cached."""
        todo = [p for p in dict.fromkeys(pairs) if p not in self._dis_cache]
        if todo:
            logits = self.dis_model.predict([(_clip(q, Q_WORDS), _clip(p, P_WORDS)) for q, p in todo],
                                            batch_size=32, show_progress_bar=False)
            for p, z in zip(todo, np.asarray(logits, dtype=float).reshape(-1)):
                self._dis_cache[p] = float(1 / (1 + math.exp(-z)))
        return [self._dis_cache[p] for p in pairs]

    @property
    def nli(self):
        """General NLI model, used only to classify advancement sentences (effects.py)."""
        if self._nli_model is None:
            from sentence_transformers import CrossEncoder
            self._nli_model = CrossEncoder(self._nli_name, max_length=256)
            if self.cuda:
                self._nli_model.model.half()
            labels = self._nli_model.model.config.id2label
            self.ent_idx = next(i for i, l in labels.items() if l.lower() == "entailment")
        return self._nli_model

    def _nli(self, pairs: list[tuple[str, str]]) -> list[float]:
        model = self.nli
        todo = [p for p in dict.fromkeys(pairs) if p not in self._nli_cache]
        if todo:
            probs = model.predict(todo, batch_size=32, apply_softmax=True, show_progress_bar=False)
            for p, pr in zip(todo, probs):
                self._nli_cache[p] = float(pr[self.ent_idx])
        return [self._nli_cache[p] for p in pairs]

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

        dis = self._dis([(features[i].text, self._refs[pid][0][j].text) for pid, i, j, _ in jobs])

        best: dict[tuple[str, int], Evidence] = {}
        for (pid, i, j, sem), d in zip(jobs, dis):
            f, (passages, pemb) = features[i], self._refs[pid]
            p = passages[j]
            lex = len(f.lemmas & p.lemmas) / max(len(f.lemmas), 1)
            syn = 0.5 * (f.component_head in p.lemmas) + \
                0.5 * any(v in p.lemmas for v in f.function_verbs[:3])
            sig = {"dis": d, "sem": sem, "lex": lex, "syn": syn}
            prob = p_disclosed(sig, self.weights)
            conflict, pf = False, 1.0
            if f.params:  # numbers must agree when the feature is defined by them
                same_unit = [q for q in p.params if any(q.unit == fp.unit for fp in f.params)]
                if same_unit and not any(fp.overlaps(q) for fp in f.params for q in same_unit):
                    conflict, pf = True, 0.2
                elif not same_unit:
                    pf = 0.5
            prob *= pf
            comp_found = bool(f.component_head) and (
                f.component_head in p.lemmas or float(comp_emb[i] @ pemb[j]) >= 0.55)
            if self.records is not None:
                self.records.append({"pid": pid, "feature": i, **sig, "pf": pf,
                                     "comp_found": int(comp_found)})
            ev = Evidence(prob, p.text, p.source, sig, comp_found, conflict)
            if (pid, i) not in best or ev.prob > best[(pid, i)].prob:
                best[(pid, i)] = ev
        return {pid: [best[(pid, i)] for i in range(len(features))] for pid in pids}
