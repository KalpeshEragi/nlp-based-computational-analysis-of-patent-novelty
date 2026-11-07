"""
Evaluate the novelty / inventive-step engine on PANORAMA NOC4PC
(USPTO examiner decisions per claim: 102 = not novel, 103 = obvious, ALLOW).

Oracle setting: the examiner's cited references are given, so this measures
the analysis engine on its own (search quality is evaluated separately).

Writes one JSON line per claim with the verdict and every raw signal, so
weights and thresholds can be re-tuned offline (eval/tune.py) without
re-running the models.

    python -m eval.run_noc4pc --split validation --limit 800
    python -m eval.run_noc4pc --split test
"""

import argparse
import json
import re
import time
from pathlib import Path

import pandas as pd

from analysis.analyzer import Analyzer
from analysis.extractor import ExtractedPatent

DATA = Path("eval/data")
LABEL = {"102": "NOT_NOVEL", "103": "OBVIOUS", "ALLOW": "INVENTIVE"}


def clean_abstract(a: str) -> str:
    return re.sub(r"^\s*ABSTRACT\s*\d*\s*\d{4}-\d{2}-\d{2}\s*", "", a or "").strip()


def ref_doc(r: dict) -> dict:
    claims = r.get("claims") or []
    claims = " ".join(c if re.match(r"^\s*\d+\s*\.", c) else f"{i}. {c}"
                      for i, c in enumerate(claims, 1)) if isinstance(claims, list) else str(claims)
    paras = r.get("paragraphs") or []
    desc = " ".join(p.get("content", "") if isinstance(p, dict) else str(p) for p in paras)
    return {"title": r.get("title", ""), "abstract": r.get("abstract", ""),
            "claims": claims, "description": desc}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="validation")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    df = pd.read_parquet(DATA / f"noc4pc_{args.split}.parquet")
    if args.limit:
        # sample whole applications so references get reused, like a real run
        apps = df.application_number.drop_duplicates().sample(frac=1, random_state=args.seed)
        keep, n = [], 0
        for a in apps:
            keep.append(a)
            n += (df.application_number == a).sum()
            if n >= args.limit:
                break
        df = df[df.application_number.isin(keep)]
    out = Path(args.out or f"eval/results/noc4pc_{args.split}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {(r["app"], r["claim"]) for r in map(json.loads, out.open())}

    analyzer = Analyzer(outcome_model=None)   # never use a trained model while generating data
    t0, n = time.time(), 0
    with out.open("a") as fh:
        for _, row in df.iterrows():
            key = (row.application_number, int(row.claim_number))
            if key in done:
                continue
            ctx = json.loads(row.context)
            claims = ctx["claims"]
            claims_text = " ".join(claims) if isinstance(claims, list) else str(claims)
            doc = ExtractedPatent(ctx.get("title", ""), clean_abstract(ctx.get("abstract", "")),
                                  "", claims_text)
            refs = {r["patent_id"]: ref_doc(r) for r in json.loads(row.prior_art_specifications)}
            answer = json.loads(row.answer)
            try:
                res = analyzer.analyze(doc, prior_art={"all": list(refs)}, prior_art_docs=refs,
                                       only_claims=[key[1]], keep_inputs=True)
                c = res["claims"][0]
                inp = c["outcome_inputs"]
                rec = {"app": key[0], "claim": key[1], "label": LABEL[answer["code"]],
                       "pred": c["verdict"], "novelty_score": c["novelty_score"],
                       "inventive_score": c["inventive_score"],
                       "n_features": len(c["feature_chart"]), "refs": list(refs),
                       "signals": inp["signals"], "info": inp["info"],
                       "linked_effects": len(c["linked_effects"])}
            except Exception as e:  # keep going; failures are counted in the metrics
                rec = {"app": key[0], "claim": key[1], "label": LABEL[answer["code"]],
                       "pred": None, "error": repr(e)[:300]}
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            n += 1
            if n % 25 == 0:
                print(f"{n} claims, {(time.time() - t0) / n:.2f}s/claim", flush=True)
    print(f"done: {n} new claims -> {out}")


if __name__ == "__main__":
    main()
