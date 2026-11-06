"""
Claim text utilities shared by the corpus builder, Layer 2 and Layer 3.

Splits a patent's claims block ("1. A device ... 2. The device of claim 1 ...")
into numbered claims and marks which ones are independent.
"""

import re

# A claim starts with its number, a dot and a capital letter: "12. The method"
_CLAIM_START = re.compile(r"(?:^|(?<=\s))(\d{1,3})\s*\.\s+(?=[A-Z\"'(])")
_DEPENDS_ON = re.compile(r"\b(?:claim|claims)\s+(\d{1,3})", re.IGNORECASE)
_CANCELED = re.compile(r"^\s*\(?(canceled|cancelled|deleted)\)?\.?\s*$", re.IGNORECASE)


def split_claims(text: str) -> list[dict]:
    """
    Returns [{"num": 1, "text": "...", "parent": None, "independent": True}, ...]
    Numbers must increase by one, which filters out false splits such as
    "about 2. Further" inside a claim.
    """
    text = re.sub(r"\s+", " ", text or "").strip()
    starts, expected = [], 1
    for m in _CLAIM_START.finditer(text):
        if int(m.group(1)) == expected:
            starts.append((expected, m.start(), m.end()))
            expected += 1

    claims = []
    for i, (num, _, body_start) in enumerate(starts):
        end = starts[i + 1][1] if i + 1 < len(starts) else len(text)
        body = text[body_start:end].strip()
        if not body or _CANCELED.match(body):
            continue
        ref = _DEPENDS_ON.search(body)
        parent = int(ref.group(1)) if ref and int(ref.group(1)) < num else None
        claims.append({"num": num, "text": body, "parent": parent,
                       "independent": parent is None})
    return claims


def independent_claims(text: str) -> list[str]:
    return [c["text"] for c in split_claims(text) if c["independent"]]
