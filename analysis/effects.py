"""
Advancement (technical effect) detection.

Patents describe their advantages in the summary/description ("reduces power
consumption", "lowers manufacturing cost"). We:
  1. pre-filter sentences with effect cue words (cheap),
  2. classify each candidate with the NLI model against effect hypotheses
     (zero-shot): technical (speed, accuracy, energy, reliability, size, safety)
     or economic (cost, materials, manufacturing steps),
  3. later link each effect to the claim features that cause it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from analysis.disclosure import _sentences

_CUES = re.compile(
    r"\b(reduc\w*|improv\w*|increas\w*|decreas\w*|lower\w*|higher|faster|quicker|less|"
    r"fewer|efficien\w*|cost\w*|cheap\w*|sav(e|es|ing|ings)|simplif\w*|enhanc\w*|"
    r"prevent\w*|avoid\w*|minimi\w*|maximi\w*|eliminat\w*|without|easily|accura\w*|"
    r"reliab\w*|robust\w*|compact\w*|smaller|lighter|durab\w*|safe\w*|advantage\w*|"
    r"benefit\w*|possible to|enables?|allows?)\b", re.IGNORECASE)

HYPOTHESES = {
    "technical: speed / performance": "This makes something faster or improves performance.",
    "technical: accuracy / quality": "This improves accuracy, precision or quality.",
    "technical: energy / efficiency": "This reduces energy or power consumption or improves efficiency.",
    "technical: reliability / safety": "This improves reliability, durability or safety.",
    "technical: size / weight / simplicity": "This makes the device smaller, lighter or simpler.",
    "technical: usability": "This makes the device easier or more convenient to use.",
    "economic: cost / materials": "This reduces cost, materials or manufacturing effort.",
}


@dataclass
class Effect:
    sentence: str
    category: str
    prob: float

    @property
    def kind(self) -> str:
        return self.category.split(":")[0]

    def to_dict(self) -> dict:
        return {"sentence": self.sentence, "category": self.category,
                "kind": self.kind, "prob": round(self.prob, 3)}


def find_effects(checker, *texts: str, threshold: float = 0.6, max_sentences: int = 40) -> list[Effect]:
    """checker: DisclosureChecker (reuses its NLI model and cache)."""
    sents = []
    for t in texts:
        sents += [s for s in _sentences(t, 200) if _CUES.search(s)]
    sents = list(dict.fromkeys(sents))[:max_sentences]
    if not sents:
        return []
    labels = list(HYPOTHESES)
    pairs = [(s, HYPOTHESES[l]) for s in sents for l in labels]
    probs = checker._nli(pairs)
    effects = []
    for i, s in enumerate(sents):
        row = probs[i * len(labels):(i + 1) * len(labels)]
        j = max(range(len(labels)), key=lambda k: row[k])
        if row[j] >= threshold:
            effects.append(Effect(s, labels[j], row[j]))
    return effects
