"""
Disclosure-model benchmark on PANORAMA PI4PC: for a claim and one cited
reference, pick the paragraph the examiner cited (5 candidates, 1 gold).

This measures the LEGAL-ANALYSIS component on its own (does our disclosure
scorer find the passage that discloses the claim?), separately from the
examiner-outcome predictor.

Claims overlapping NOC4PC test applications are excluded so this benchmark
never touches the held-out outcome test set.

    python -m eval.pi4pc --split validation --scorers dense,nli,current
    python -m eval.pi4pc --split validation --scorers ce --ce-model path/to/model
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.elements import parse_claims
from analysis.features import extract_feature
from analysis.disclosure import _nlp, content_lemmas

# the previous (general NLI) disclosure formula, kept to benchmark against
LEGACY_W = {"bias": -5.0, "nli": 4.0, "sem": 4.0, "lex": 2.0, "syn": 1.0}
NLI_GATE_LO, NLI_GATE_HI = 0.45, 0.75

DATA = Path("eval/data")


def load(split: str, limit: int = 0, seed: int = 0) -> list[dict]:
    df = pd.read_parquet(DATA / f"pi4pc_{split}.parquet")
    excluded = set(pd.read_parquet(DATA / "noc4pc_test.parquet",
                                   columns=["application_number"]).application_number)
    df = df[~df.application_number.isin(excluded)]
    rows = []
    for r in df.itertuples():
        ctx = json.loads(r.context)
        claims = {c.num: c for c in parse_claims(" ".join(ctx["claims"]))}
        c = claims.get(int(r.claim_number))
        if not c:
            continue
        opts = json.loads(r.options)
        rows.append({"app": r.application_number, "claim": int(r.claim_number),
                     "own": " ".join(c.elements), "full": c.text, "elements": c.all_elements,
                     "options": {int(k): v for k, v in opts.items()},
                     "gold": set(int(x) for x in r.gold_answers)})
    if limit:
        random.Random(seed).shuffle(rows)
        rows = rows[:limit]
    return rows


def score_rows(rows, scorer) -> dict:
    top1, rr = [], []
    for r in rows:
        keys = list(r["options"])
        s = scorer(r, [r["options"][k] for k in keys])
        order = [keys[i] for i in np.argsort(-np.asarray(s))]
        top1.append(order[0] in r["gold"])
        rr.append(1 / (1 + min(order.index(g) for g in r["gold"])))
    return {"n": len(rows), "top1_acc": float(np.mean(top1)), "mrr": float(np.mean(rr))}


# ---- scorers -------------------------------------------------------------
class Scorers:
    def __init__(self):
        from retrieval.sources import LocalIndexSource
        self.src = LocalIndexSource("retrieval/index")
        self._nli = None
        self._ce = None

    def enc(self, texts, n=256):
        e = self.src.encoder
        e.max_seq_length = n
        return e.encode(texts, normalize_embeddings=True, batch_size=32, show_progress_bar=False)

    @property
    def nli(self):
        if self._nli is None:
            from analysis.disclosure import DisclosureChecker
            self._nli = DisclosureChecker(self.src.encoder)
        return self._nli

    def random(self, r, paras):
        return np.random.default_rng(hash(r["app"]) % 2**32).random(len(paras))

    def dense(self, r, paras):
        q = self.enc([r["own"]])[0]
        return self.enc(paras) @ q

    def nli_score(self, r, paras):
        hyp = " ".join(r["own"].split()[:40])
        return self.nli._nli([(p, hyp) for p in paras])

    def current(self, r, paras):
        """The previous P(disclosed) formula (general NLI) at claim level."""
        w = LEGACY_W
        q = self.enc([r["own"]])[0]
        P = self.enc(paras)
        sem = P @ q
        nli = np.array(self.nli_score(r, paras))
        f = extract_feature(r["own"])
        lem = [content_lemmas(d) for d in _nlp().pipe(paras)]
        lex = np.array([len(f.lemmas & l) / max(len(f.lemmas), 1) for l in lem])
        syn = np.array([0.5 * (f.component_head in l) + 0.5 * any(v in l for v in f.function_verbs[:3])
                        for l in lem])
        gate = np.clip((sem - NLI_GATE_LO) / (NLI_GATE_HI - NLI_GATE_LO), 0, 1)
        z = w["bias"] + w["nli"] * nli * gate + w["sem"] * sem + w["lex"] * lex + w["syn"] * syn
        return 1 / (1 + np.exp(-z))

    def ce(self, r, paras):
        # same clipping as training (disclosure_model/train_ce.py)
        from disclosure_model.train_ce import P_WORDS, Q_WORDS, clip
        return self._ce.predict([(clip(r["own"], Q_WORDS), clip(p, P_WORDS)) for p in paras],
                                batch_size=16, show_progress_bar=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="validation")
    ap.add_argument("--limit", type=int, default=800)
    ap.add_argument("--scorers", default="random,dense,nli,current")
    ap.add_argument("--ce-model", default=None)
    args = ap.parse_args()
    rows = load(args.split, args.limit)
    S = Scorers()
    if args.ce_model:
        from sentence_transformers import CrossEncoder
        S._ce = CrossEncoder(args.ce_model, max_length=512)
    names = {"random": S.random, "dense": S.dense, "nli": S.nli_score, "current": S.current, "ce": S.ce}
    out = {}
    for n in args.scorers.split(","):
        out[n] = score_rows(rows, names[n])
        print(f"{args.split:10} {n:8} n={out[n]['n']} top1={out[n]['top1_acc']:.3f} MRR={out[n]['mrr']:.3f}",
              flush=True)
    Path("eval/results").mkdir(exist_ok=True)
    (Path("eval/results") / f"pi4pc_{args.split}_{'_'.join(out)}.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
