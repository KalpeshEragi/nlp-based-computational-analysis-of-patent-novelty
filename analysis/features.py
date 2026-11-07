"""
Feature extraction: claim element -> component | function | parameters.

    "a first balloon operation button for swelling or contracting a first balloon"
      component : "first balloon operation button"   (head: "button")
      function  : "swelling or contracting a first balloon"   (verbs: swell, contract)
      parameters: []

    "each sensor operates at a frequency between 40 kHz and 80 kHz"
      component : "sensor"
      function  : "operates at a frequency"
      parameters: [Param(40, 80, "khz")]
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache

_FUNC_CUES = re.compile(
    r"\b(configured to|adapted to|arranged to|operable to|capable of|so as to|in order to|"
    r"for|to|that|which|wherein|whereby|by)\b", re.IGNORECASE)
_SKIP_HEADS = {"claim", "claims", "one", "plurality", "number", "portion", "part", "end",
               "side", "least", "each", "same", "first", "second", "method", "step"}

_NUM = r"(\d+(?:[.,]\d+)?)"
_UNIT = (r"(%|percent|°c|degrees?|k?hz|mhz|ghz|nm|μm|um|mm|cm|m|km|mg|g|kg|ml|l|"
         r"v|mv|kv|a|ma|w|kw|mw|s|ms|sec|seconds?|min|minutes?|hours?|rpm|pa|kpa|mpa|bar|db|"
         r"wt%|ppm|mol|m/s|°)")
_RANGE = re.compile(rf"{_NUM}\s*{_UNIT}?\s*(?:-|–|to|and)\s*{_NUM}\s*{_UNIT}", re.IGNORECASE)
_SINGLE = re.compile(rf"(?:(less|more|greater|lower|higher) than|at least|at most|about|"
                     rf"approximately|up to)?\s*{_NUM}\s*{_UNIT}\b", re.IGNORECASE)


@dataclass
class Param:
    low: float
    high: float
    unit: str

    def overlaps(self, other: "Param") -> bool:
        return self.unit == other.unit and self.low <= other.high and other.low <= self.high

    def __str__(self):
        v = f"{self.low:g}" if self.low == self.high else f"{self.low:g}-{self.high:g}"
        return f"{v} {self.unit}"


@dataclass
class Feature:
    text: str
    component: str = ""
    component_head: str = ""
    function: str = ""
    function_verbs: list[str] = field(default_factory=list)
    params: list[Param] = field(default_factory=list)
    lemmas: set[str] = field(default_factory=set)   # content-word lemmas (lexical signal)

    def to_dict(self) -> dict:
        return {"text": self.text, "component": self.component, "function": self.function,
                "params": [str(p) for p in self.params]}


@lru_cache(maxsize=1)
def _nlp():
    import spacy
    return spacy.load("en_core_web_sm", disable=["ner"])


def _num(s: str) -> float:
    return float(s.replace(",", ""))


def _norm_unit(u: str) -> str:
    u = u.lower()
    return {"percent": "%", "degrees": "°", "degree": "°", "seconds": "s", "second": "s",
            "sec": "s", "minutes": "min", "minute": "min", "hours": "h", "hour": "h"}.get(u, u)


def extract_params(text: str) -> list[Param]:
    params, used = [], []
    for m in _RANGE.finditer(text):
        unit = _norm_unit(m.group(4) or m.group(2) or "")
        lo, hi = sorted((_num(m.group(1)), _num(m.group(3))))
        params.append(Param(lo, hi, unit))
        used.append(m.span())
    for m in _SINGLE.finditer(text):
        if any(a <= m.start(2) < b for a, b in used):
            continue
        v, unit, cmp = _num(m.group(2)), _norm_unit(m.group(3)), (m.group(1) or "").lower()
        lo, hi = v, v
        if cmp in ("less", "lower") or "at most" in m.group(0).lower() or "up to" in m.group(0).lower():
            lo = float("-inf")
        elif cmp in ("more", "greater", "higher") or "at least" in m.group(0).lower():
            hi = float("inf")
        params.append(Param(lo, hi, unit))
    return params


def content_lemmas(doc) -> set[str]:
    return {t.lemma_.lower() for t in doc
            if t.pos_ in ("NOUN", "PROPN", "VERB", "ADJ") and not t.is_stop and len(t.text) > 2}


def extract_feature(text: str) -> Feature:
    doc = _nlp()(text)
    f = Feature(text=text, params=extract_params(text), lemmas=content_lemmas(doc))

    # component: first meaningful noun chunk (the thing the element introduces)
    for chunk in doc.noun_chunks:
        if chunk.root.lemma_.lower() in _SKIP_HEADS or chunk.root.pos_ == "PRON":
            continue
        words = [t for t in chunk if t.pos_ not in ("DET", "PRON")]
        if words:
            f.component = " ".join(t.text for t in words).lower()
            f.component_head = chunk.root.lemma_.lower()
            break

    # function: what the component does = from the first real verb to the end
    # ("for swelling ...", "attached to an insertion section ...")
    verbs = [t for t in doc if t.pos_ == "VERB" and t.lemma_.lower() not in
             ("be", "have", "include", "comprise", "consist", "configure", "adapt", "arrange")]
    if verbs:
        f.function = doc[verbs[0].i:].text.strip(" ,;.")[:200]
    f.function_verbs = list(dict.fromkeys(v.lemma_.lower() for v in verbs))[:5]
    return f
