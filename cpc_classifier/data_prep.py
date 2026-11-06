"""
Step 1 - Build the CPC classification dataset from HUPD.

Streams HUPD yearly archives straight from Hugging Face (nothing big is saved
to disk), keeps only the fields we need, and writes time-based splits:

    train: 2011-2015   val: 2016   test: 2017

Usage:
    python -m cpc_classifier.data_prep                 # full (~20k per train year)
    python -m cpc_classifier.data_prep --quick         # small test run
"""

import argparse
import ast
import json
import random
import re
import tarfile
import urllib.request
from pathlib import Path

import pandas as pd

BASE_URL = "https://huggingface.co/datasets/HUPD/hupd/resolve/main/data/{year}.tar.gz"
OUT_DIR = Path(__file__).parent / "data"

SPLITS = {
    "train": [2011, 2012, 2013, 2014, 2015],
    "val": [2016],
    "test": [2017],
}


def first_claim(claims: str) -> str:
    """Return claim 1 only (claims text is '1. ... 2. ...')."""
    parts = re.split(r"\s(?=\d{1,3}\s*\.\s)", claims.strip(), maxsplit=1)
    return parts[0].strip()


def clean(text: str) -> str:
    """Drop HUPD section markers like '<SOH> SUMMARY <EOH>' and squash whitespace."""
    text = re.sub(r"<SOH>.*?<EOH>", " ", text or "")
    return re.sub(r"\s+", " ", text).strip()


def _as_list(labels) -> list[str]:
    """HUPD labels come as a list, or sometimes as a stringified list."""
    if isinstance(labels, list):
        return labels
    try:
        return ast.literal_eval(labels) if labels else []
    except (ValueError, SyntaxError):
        return []


def to_subclasses(labels, main: str) -> list[str]:
    """CPC like 'A61B100082' -> subclass 'A61B'. Main subclass goes first."""
    subs = [main[:4]] + [c[:4] for c in _as_list(labels) if len(c) >= 4]
    return list(dict.fromkeys(subs))


def parse_record(d: dict, year: int) -> dict | None:
    # CPC only exists from 2013. Before that, fall back to IPC, which matches
    # CPC at the 4-character subclass level in almost all cases.
    main = (d.get("main_cpc_label") or "").strip()
    labels, source = d.get("cpc_labels"), "cpc"
    if len(main) < 4:
        main = (d.get("main_ipcr_label") or "").strip()
        labels, source = d.get("ipcr_labels"), "ipc"
    abstract = clean(d.get("abstract", ""))
    claims = clean(d.get("claims", ""))
    if len(main) < 4 or not abstract or not claims:
        return None
    return {
        "application_number": d.get("application_number"),
        "year": year,
        "title": clean(d.get("title", "")),
        "abstract": abstract,
        "first_claim": first_claim(claims)[:3000],
        "summary": clean(d.get("summary", ""))[:3000],
        "main_subclass": main[:4],
        "subclasses": to_subclasses(labels, main),
        "label_source": source,
    }


def stream_year(year: int, limit: int, keep_prob: float, seed: int) -> list[dict]:
    """Stream one year's tar.gz and collect up to `limit` usable records."""
    rng = random.Random(seed + year)
    rows, seen = [], 0
    print(f"[{year}] streaming...", flush=True)
    with urllib.request.urlopen(BASE_URL.format(year=year)) as resp:
        with tarfile.open(fileobj=resp, mode="r|gz") as tar:
            for member in tar:
                if not (member.isfile() and member.name.endswith(".json")):
                    continue
                seen += 1
                # random skip spreads samples through the archive instead of
                # taking only the first files
                if rng.random() > keep_prob:
                    continue
                try:
                    d = json.load(tar.extractfile(member))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
                rec = parse_record(d, year)
                if rec:
                    rows.append(rec)
                if len(rows) % 2000 == 0 and rows:
                    print(f"[{year}] {len(rows)}/{limit} (scanned {seen})", flush=True)
                if len(rows) >= limit:
                    break
    print(f"[{year}] done: {len(rows)} records", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-year-train", type=int, default=20000)
    ap.add_argument("--per-year-eval", type=int, default=10000)
    ap.add_argument("--keep-prob", type=float, default=0.25,
                    help="chance of keeping each file while streaming")
    ap.add_argument("--quick", action="store_true",
                    help="tiny run to test the pipeline")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.quick:
        args.per_year_train, args.per_year_eval = 500, 300

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for split, years in SPLITS.items():
        limit = args.per_year_train if split == "train" else args.per_year_eval
        rows = []
        for y in years:
            rows += stream_year(y, limit, args.keep_prob, args.seed)
        df = pd.DataFrame(rows)
        path = OUT_DIR / f"{split}{'_quick' if args.quick else ''}.parquet"
        df.to_parquet(path, index=False)
        print(f"==> {split}: {len(df)} rows, "
              f"{df['main_subclass'].nunique()} subclasses -> {path}", flush=True)


if __name__ == "__main__":
    main()
