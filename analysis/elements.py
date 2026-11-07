"""
Claim element parsing.

Patent claims follow a fixed structure:

    independent:  <preamble> <transition> <element>; <element>; and <element>.
                  "A remote controller ... comprising: a first button ...; a display ...; and ..."
    dependent:    The <thing> of claim N, <connector> <added feature(s)>.
                  "The remote controller according to claim 1, wherein the button is a push button."

Novelty is judged element by element (every element of a claim must be found
in one prior-art reference for the claim to be anticipated), so we split each
claim into those elements.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from retrieval.claims import split_claims

_TRANSITION = re.compile(
    r"\b(consisting essentially of|consisting of|comprising|comprises|including|"
    r"includes|having|characterized in that|characterised in that|"
    r"the improvement comprising)\b\s*:?", re.IGNORECASE)
_DEP_CONNECTOR = re.compile(
    r"(?:of|in|according to|as claimed in|as recited in|as defined in|as set forth in)\s+"
    r"(?:any\s+(?:one\s+)?of\s+)?claims?\s+\d+(?:\s*(?:-|to|or|and)\s*\d+)*\s*,?\s*"
    r"(?:(?:further\s+)?(?:wherein|in which|where|characterized in that|"
    r"further comprising|further including|comprising|including)\b\s*:?)?",
    re.IGNORECASE)
_MIN_WORDS = 3


@dataclass
class ClaimElements:
    num: int
    text: str
    parent: int | None
    independent: bool
    preamble: str
    transition: str
    elements: list[str] = field(default_factory=list)
    # for dependent claims: the parent chain's elements, so the claim can be
    # judged as a whole (a dependent claim includes all of its parent's elements)
    inherited: list[str] = field(default_factory=list)

    @property
    def all_elements(self) -> list[str]:
        return self.inherited + self.elements

    def to_dict(self) -> dict:
        return {"num": self.num, "text": self.text, "parent": self.parent,
                "independent": self.independent, "preamble": self.preamble,
                "transition": self.transition, "elements": self.elements,
                "inherited": self.inherited}


def _split_body(body: str) -> list[str]:
    """Split the claim body into elements: ';' first, then ', and' / ', wherein'."""
    body = body.strip().rstrip(".").strip()
    parts = [p for p in re.split(r";\s*(?:and\s+)?", body) if p.strip()]
    if len(parts) == 1 and len(body.split()) > 30:
        parts = [p for p in re.split(r",\s*(?:and\s+)?(?=wherein\b|whereby\b|and\s+(?:a|an|the)\b)",
                                     body) if p.strip()]
    out = []
    for p in parts:
        p = re.sub(r"^(and|or)\s+", "", p.strip(" ,:"), flags=re.I).strip()
        if len(p.split()) >= _MIN_WORDS:
            out.append(p)
        elif out:  # tiny fragment: glue to the previous element
            out[-1] = f"{out[-1]}; {p}"
    return out or [body]


def parse_claim(num: int, text: str, parent: int | None) -> ClaimElements:
    if parent is not None:
        m = _DEP_CONNECTOR.search(text)
        if m:
            preamble = text[:m.start()].strip(" ,")
            transition = m.group(0).strip(" ,:")
            body = text[m.end():]
        else:
            preamble, transition, body = "", "", text
    else:
        m = _TRANSITION.search(text)
        if m:
            preamble = text[:m.start()].strip(" ,")
            transition = m.group(1).lower()
            body = text[m.end():]
        else:
            preamble, transition, body = "", "", text
    return ClaimElements(num=num, text=text, parent=parent, independent=parent is None,
                         preamble=preamble, transition=transition,
                         elements=_split_body(body))


def parse_claims(claims_text: str) -> list[ClaimElements]:
    parsed = [parse_claim(c["num"], c["text"], c["parent"]) for c in split_claims(claims_text)]
    by_num = {c.num: c for c in parsed}
    for c in parsed:  # parents always come first, so their inherited list is ready
        if c.parent in by_num:
            p = by_num[c.parent]
            c.inherited = p.inherited + p.elements
    return parsed
