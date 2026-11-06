"""
Build the prior-art corpus from HUPD (streamed, nothing large stored).

Keeps per patent: title, abstract, summary, full claims, independent claims,
CPC subclasses and filing date. Prior art years default to 2012-2016 so that
2017+ applications can later be used as test inventions.

Usage:
    python -m retrieval.build_corpus --per-year 20000          # 100k corpus
    python -m retrieval.build_corpus --per-year 400 --years 2015 --out retrieval/index_small
"""

import argparse
import json
import random
import tarfile
import urllib.request
from pathlib import Path

import pandas as pd

from cpc_classifier.data_prep import BASE_URL, _as_list, clean
from retrieval.claims import independent_claims


def to_row(d: dict) -> dict | None:
    abstract = clean(d.get("abstract", ""))
    claims = clean(d.get("claims", ""))
    main = (d.get("main_cpc_label") or d.get("main_ipcr_label") or "").strip()
    labels = d.get("cpc_labels") if d.get("main_cpc_label") else d.get("ipcr_labels")
    if len(main) < 4 or not abstract or not claims:
        return None
    indep = independent_claims(claims)
    if not indep:
        return None
    subs = list(dict.fromkeys([main[:4]] + [c[:4] for c in _as_list(labels) if len(c) >= 4]))
    return {
        "patent_id": d.get("application_number"),
        "publication_number": d.get("publication_number"),
        "title": clean(d.get("title", "")),
        "abstract": abstract,
        "summary": clean(d.get("summary", ""))[:6000],
        "claims": claims[:30000],
        "independent_claims": " ".join(indep)[:6000],
        "subclasses": subs,
        "filing_date": d.get("filing_date") or "",
        "decision": d.get("decision", ""),
    }


def stream(year: int, limit: int, keep_prob: float, seed: int) -> list[dict]:
    rng, rows = random.Random(seed + year), []
    print(f"[{year}] streaming...", flush=True)
    with urllib.request.urlopen(BASE_URL.format(year=year)) as resp, \
            tarfile.open(fileobj=resp, mode="r|gz") as tar:
        for m in tar:
            if not (m.isfile() and m.name.endswith(".json")) or rng.random() > keep_prob:
                continue
            try:
                row = to_row(json.load(tar.extractfile(m)))
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            if row:
                rows.append(row)
                if len(rows) % 5000 == 0:
                    print(f"[{year}] {len(rows)}/{limit}", flush=True)
                if len(rows) >= limit:
                    break
    print(f"[{year}] done: {len(rows)}", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", default="2012,2013,2014,2015,2016")
    ap.add_argument("--per-year", type=int, default=20000)
    ap.add_argument("--keep-prob", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--out", default="retrieval/index")
    args = ap.parse_args()

    rows = []
    for y in map(int, args.years.split(",")):
        rows += stream(y, args.per_year, args.keep_prob, args.seed)
    df = pd.DataFrame(rows).drop_duplicates("patent_id").reset_index(drop=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / "corpus.parquet", index=False)
    print(f"==> corpus: {len(df)} patents -> {out / 'corpus.parquet'}", flush=True)


if __name__ == "__main__":
    main()
