"""
Claim text utilities shared by the corpus builder, Layer 2 and Layer 3.

Splits a patent's claims block ("1. A device ... 2. The device of claim 1 ...")
into numbered claims and marks which ones are independent.
"""

import re

# A claim starts with its number and a dot: "12. The method". A lowercase start
# ("19. he tangible ...", lost OCR letter) is accepted only right after the end
# of the previous claim.
_CLAIM_START = re.compile(r"(?:^|(?<=\s))(\d{1,3})\s*\.\s+(?=[A-Za-z\"'(])")
_DEPENDS_ON = re.compile(r"\b(?:claim|claims)\s+(\d{1,3})", re.IGNORECASE)
_CANCELED = re.compile(r"^\s*\(?(canceled|cancelled|deleted)\)?\.?\s*$", re.IGNORECASE)
# amendment status markers used in office-action claim listings
_STATUS = re.compile(r"^\((original|currently amended|previously presented|previously amended|"
                     r"amended|new|withdrawn|withdrawn[- ]currently amended)\)\s*", re.IGNORECASE)


def _clean(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    text = re.sub(r"\[\s*claim\s*\d+\s*\]", " ", text, flags=re.IGNORECASE)   # "[Claim 1]"
    # junk between the number and the text: "18. . The", "10. 0. The", "19. 19 The",
    # "1. 00023 An" (paragraph number), "1. 1) A"
    def drop_junk(m):
        num, junk = m.group(1), m.group(2).split()
        ok = all(t in (".", "0.", ")") or t.rstrip(".)") == num or len(t.rstrip(".)")) >= 4
                 or t.endswith(")") for t in junk)
        return f"{num}. " if ok else m.group(0)

    # only at a claim boundary (start of text or right after the previous claim's end)
    text = re.sub(r"(?:^|(?<=[.;:] ))(\d{1,3})\s*\.\s*((?:(?:[.\d]+\)?|\))\s+)+)(?=[A-Za-z(])", drop_junk, text)
    return re.sub(r"\s+", " ", text)


def split_claims(text: str) -> list[dict]:
    """
    Returns [{"num": 1, "text": "...", "parent": None, "independent": True}, ...]
    Numbers must increase by one, which filters out false splits such as
    "about 2. Further" inside a claim.
    """
    text = _clean(text)
    starts, expected = [], 1
    for m in _CLAIM_START.finditer(text):
        if int(m.group(1)) != expected:
            continue
        nxt = text[m.end():m.end() + 1]
        before = text[:m.start()].rstrip()[-1:]
        if nxt.islower() and before not in ("", ".", ";"):
            continue
        starts.append((expected, m.start(), m.end()))
        expected += 1

    claims = []
    for i, (num, _, body_start) in enumerate(starts):
        end = starts[i + 1][1] if i + 1 < len(starts) else len(text)
        body = _STATUS.sub("", text[body_start:end].strip())
        if not body or _CANCELED.match(body):
            continue
        ref = _DEPENDS_ON.search(body)
        parent = int(ref.group(1)) if ref and int(ref.group(1)) < num else None
        claims.append({"num": num, "text": body, "parent": parent,
                       "independent": parent is None})
    return claims


def independent_claims(text: str) -> list[str]:
    return [c["text"] for c in split_claims(text) if c["independent"]]
