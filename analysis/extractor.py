"""
Document extraction: patent PDF / DOCX / TXT (or a patent number) ->
title, abstract, summary and claims.

PDFs of granted patents are usually two-column scans with an OCR text layer,
so plain text extraction interleaves the columns line by line and keeps the
margin line numbers (5, 10, 15 ...). We rebuild the reading order from word
positions, drop the gutter numbers, then repair OCR artefacts.
"""

from __future__ import annotations

import io
import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from retrieval.claims import split_claims

# ----------------------------------------------------------------------------
# Section headings (matched against whole lines)
# ----------------------------------------------------------------------------
_H = {
    "claims": r"(what is claimed is|what we claim is|the invention claimed is|"
              r"i claim|we claim|claims?|what is claimed)",
    "abstract": r"(\(\s*57\s*\)\s*)?abstract( of the (disclosure|invention))?",
    "summary": r"(brief )?summary( of the (invention|disclosure))?",
    "other": r"(background( of the (invention|disclosure))?|(technical )?field( of the "
             r"(invention|disclosure))?|brief description of (the )?drawings?|"
             r"detailed description.*|description of (the )?(preferred )?embodiments?|"
             r"cross[- ]reference.*|related applications?|description)",
}
_HEADING = {k: re.compile(rf"^\s*{v}\s*:?\s*$", re.IGNORECASE) for k, v in _H.items()}


@dataclass
class ExtractedPatent:
    title: str = ""
    abstract: str = ""
    summary: str = ""
    claims: str = ""
    source: str = ""
    warnings: list[str] = field(default_factory=list)
    priority_date: str = ""   # YYYYMMDD; prior art must be older than this

    @property
    def n_claims(self) -> int:
        return len(split_claims(self.claims))

    def to_dict(self) -> dict:
        return {"title": self.title, "abstract": self.abstract, "summary": self.summary,
                "claims": self.claims, "n_claims": self.n_claims,
                "source": self.source, "warnings": self.warnings,
                "priority_date": self.priority_date}


# ----------------------------------------------------------------------------
# Raw text
# ----------------------------------------------------------------------------
def _pdf_text(data: bytes) -> str:
    import fitz  # PyMuPDF

    pages = []
    with fitz.open(stream=data, filetype="pdf") as doc:
        for page in doc:
            words = page.get_text("words")  # x0, y0, x1, y1, text, block, line, no
            if not words:
                continue
            mid, width = page.rect.width / 2, page.rect.width
            left = [w for w in words if w[2] <= mid + 5]
            right = [w for w in words if w[0] >= mid - 5]
            two_col = len(left) > 40 and len(right) > 40 and \
                sum(w[0] < mid - 5 < w[2] for w in words) < 0.05 * len(words)
            if two_col:
                # margin line numbers: multiples of 5 centred on the gutter.
                # (claim numbers start the right column a bit further right)
                gutter = lambda w: re.fullmatch(r"\d{1,2}", w[4]) and \
                    int(w[4]) % 5 == 0 and abs((w[0] + w[2]) / 2 - mid) < 0.017 * width
                cols = [[w for w in left if not gutter(w)],
                        [w for w in right if not gutter(w) and w not in left]]
            else:
                cols = [list(words)]
            for col in cols:
                pages.append(_lines(col))
    return "\n".join(pages)


def _lines(words) -> str:
    """Group words into lines by y position, then order by x."""
    words = sorted(words, key=lambda w: (w[1], w[0]))
    lines, cur, cur_y = [], [], None
    for w in words:
        if cur_y is None or abs(w[1] - cur_y) <= 3:
            cur.append(w)
            cur_y = w[1] if cur_y is None else cur_y
        else:
            lines.append(" ".join(x[4] for x in sorted(cur, key=lambda x: x[0])))
            cur, cur_y = [w], w[1]
    if cur:
        lines.append(" ".join(x[4] for x in sorted(cur, key=lambda x: x[0])))
    return "\n".join(lines)


def _docx_text(data: bytes) -> str:
    import docx

    d = docx.Document(io.BytesIO(data))
    return "\n".join(p.text for p in d.paragraphs)


def raw_text(data: bytes, filename: str) -> str:
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        return _pdf_text(data)
    if ext == ".docx":
        return _docx_text(data)
    if ext in (".txt", ".md"):
        return data.decode("utf-8", errors="replace")
    raise ValueError(f"unsupported file type: {ext} (use PDF, DOCX or TXT)")


# ----------------------------------------------------------------------------
# OCR repair
# ----------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _vocab() -> frozenset:
    """Word list from the prior-art corpus (BM25 vocabulary) to rejoin split words."""
    for p in (Path("retrieval/index/bm25/vocab.index.json"),
              Path(__file__).resolve().parent.parent / "retrieval/index/bm25/vocab.index.json"):
        if p.exists():
            return frozenset(json.loads(p.read_text(encoding="utf-8")).keys())
    return frozenset()


def _rejoin_split_words(text: str) -> str:
    """'opera\\ntion' -> 'operation' when the joined word is a known word."""
    vocab = _vocab()
    if not vocab:
        return text

    def fix(m):
        a, b = m.group(1), m.group(2)
        joined = (a + b).lower()
        # join when the result is a real word and the pieces look like fragments
        # ("dis"+"play", "sec"+"tion"); OCR noise puts some fragments in the vocab
        fragment = (a.lower() not in vocab or b.lower() not in vocab
                    or len(a) <= 3 or b.lower() in _SUFFIXES)
        return a + b if joined in vocab and fragment else m.group(0)

    return re.sub(r"([A-Za-z]{2,})-?\n([a-z]{2,})", fix, text)


_SUFFIXES = {"tion", "tions", "sion", "ment", "ments", "ing", "ings", "ed", "ly", "ity",
              "ance", "ence", "able", "ible", "ally", "ure", "ture", "tive", "ous", "ize",
              "ized", "izing", "ation", "ations", "ated", "ating", "er", "ers", "or", "ors"}

# running page headers of US documents, e.g. "US 2016/022O097 A1 Aug. 4, 2016 10"
_PAGE_HEADER = re.compile(
    r"^\s*US\s*[\dO][\dO ,/]{5,}\s*[AB]\s*\d.*$|^\s*(Sheet \d+ of \d+|Patent Application Publication.*"
    r"|U\.?S\.? Patent.*)\s*$", re.IGNORECASE | re.MULTILINE)


def normalize(text: str) -> str:
    text = text.replace("­", "").replace("\r", "")
    text = _PAGE_HEADER.sub("", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = _rejoin_split_words(text)
    text = re.sub(r"\s+([,.;:)\]])", r"\1", text)        # "device ," -> "device,"
    text = re.sub(r"([(\[])\s+", r"\1", text)             # "( 57" -> "(57"
    text = re.sub(r"(\w) - (\w)", r"\1-\2", text)         # "swollen - state"
    text = re.sub(r"(\d) ?, ?(\d{3})", r"\1,\2", text)    # "9 , 968" -> "9,968"
    return text


# ----------------------------------------------------------------------------
# Sections
# ----------------------------------------------------------------------------
def _flow(text: str) -> str:
    """Join wrapped lines into running text."""
    return re.sub(r"\s+", " ", text).strip()


def split_sections(text: str) -> dict[str, str]:
    lines = text.split("\n")
    marks = []
    for i, line in enumerate(lines):
        for name, pat in _HEADING.items():
            if len(line) < 80 and pat.match(line):
                marks.append((i, name))
                break
    sections: dict[str, str] = {}
    for n, (i, name) in enumerate(marks):
        end = marks[n + 1][0] if n + 1 < len(marks) else len(lines)
        body = _flow("\n".join(lines[i + 1:end]))
        # keep the longest block per section (front-page abstract vs repeats)
        if name != "other" and len(body) > len(sections.get(name, "")):
            sections[name] = body
    return sections


_ABSTRACT_END = re.compile(
    r"\d+\s*Claims?,\s*\d+\s*Drawing|\(\s*5[2-8]\s*\)|\bFIG\.\s*\d|Sheet \d+ of \d+", re.I)


def _clean_abstract(abstract: str) -> str:
    abstract = re.sub(r"^abstract\s*", "", abstract, flags=re.I)
    m = _ABSTRACT_END.search(abstract)
    return (abstract[:m.start()] if m else abstract).strip()[:3000]


def _clean_claims(claims: str) -> str:
    claims = re.sub(r"(\*\s*){2,}.*$", "", claims)           # "* * * * *" end marker
    m = re.search(r"\b1\s*\.\s+[A-Z]", claims)               # start at claim 1
    return claims[m.start():].strip() if m else claims.strip()


def _title(text: str) -> str:
    m = re.search(r"\(\s*54\s*\)\s*(.+?)(?=\(\s*\d{2}\s*\)|\n\s*\n)", text, re.S)
    if m:
        return _flow(m.group(1))
    m = re.search(r"title( of the invention)?\s*:?\s*\n?(.+)", text, re.I)
    if m:
        return m.group(2).strip()
    first = next((l.strip() for l in text.split("\n") if len(l.strip()) > 3), "")
    return first[:200]


def extract(data: bytes, filename: str) -> ExtractedPatent:
    text = normalize(raw_text(data, filename))
    secs = split_sections(text)
    doc = ExtractedPatent(
        title=_title(text), abstract=_clean_abstract(secs.get("abstract", "")),
        summary=secs.get("summary", ""), claims=_clean_claims(secs.get("claims", "")),
        source=filename)
    if not doc.claims:
        # no heading found: fall back to the first "1. A ..." run of numbered claims
        m = re.search(r"(?:^|\n)\s*1\s*\.\s+(A|An|The)\s.+", text, re.S)
        if m:
            doc.claims = _clean_claims(_flow(m.group(0)))
            doc.warnings.append("no claims heading found; used numbered claims")
    for f in ("abstract", "claims"):
        if not getattr(doc, f):
            doc.warnings.append(f"{f} not found")
    if doc.claims and doc.n_claims == 0:
        doc.warnings.append("claims text found but could not be split into numbered claims")
    return doc


# ----------------------------------------------------------------------------
# Patent number lookup (clean structured text from Google Patents)
# ----------------------------------------------------------------------------
def fetch_patent(number: str, timeout: int = 20) -> ExtractedPatent:
    """e.g. 'US9968239B2' or 'US20160220097A1'."""
    import urllib.request

    from bs4 import BeautifulSoup

    number = re.sub(r"[\s,/-]", "", number).upper()
    req = urllib.request.Request(f"https://patents.google.com/patent/{number}/en",
                                 headers={"User-Agent": "Mozilla/5.0 (patent-analyser)"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        soup = BeautifulSoup(r.read(), "html.parser")

    def txt(sel):
        el = soup.select_one(sel)
        return _flow(el.get_text(" ")) if el else ""

    claims_el = soup.select_one("section[itemprop=claims]")
    claims = []
    if claims_el:
        # only the inner claim divs carry a num attribute; outer wrappers repeat text
        for c in claims_el.select("div.claim[num]"):
            body = re.sub(r"^\d+\s*\.\s*", "", normalize(_flow(c.get_text(" "))))
            if body:
                claims.append(f"{int(c['num'])}. {body}")
    title = txt("meta[name=DC\\.title]") or ""
    meta = soup.select_one("meta[name='DC.title']")
    if meta:
        title = meta.get("content", "").strip()
    desc = soup.select_one("section[itemprop=description]")
    summary = ""
    if desc:
        summary = split_sections(normalize(desc.get_text("\n"))).get("summary", "")
    date = ""
    for prop in ("priorityDate", "filingDate"):
        el = soup.select_one(f"time[itemprop={prop}]")
        if el and el.get("datetime"):
            date = el["datetime"].replace("-", "")
            break
    doc = ExtractedPatent(title=title, abstract=_clean_abstract(txt("section[itemprop=abstract]")),
                          summary=summary, claims=" ".join(claims), source=number,
                          priority_date=date)
    for f in ("abstract", "claims"):
        if not getattr(doc, f):
            doc.warnings.append(f"{f} not found")
    return doc
