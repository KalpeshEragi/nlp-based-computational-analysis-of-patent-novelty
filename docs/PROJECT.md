# Patent Analyser — how it works, how it was built, how it was measured

A system that reads a patent application (PDF, Word or a patent number), finds
the closest prior art in a patent corpus, and analyses every claim the way a
patent examiner would: which features are already known, which are new, and
whether the claim is novel and inventive. Alongside that legal analysis it
predicts the likely USPTO examiner outcome (§102 rejection, §103 rejection or
allowance).

This document is the project's running record. It is updated after every
module.

---

## 1. The big picture

```
 Upload (PDF / DOCX / TXT / patent number)
        │
        ▼
 ① Document extraction ── title, abstract, summary, claims (+ priority date)
        │
        ▼
 ② Claim parsing ───────── numbered claims, dependencies, elements
        │
        ▼
 ③ LAYER 1  CPC classifier ── suggests technology classes → user confirms
        │                       + date filter (prior art must be older)
        ▼
 ④ LAYER 2  Patent search ─── hybrid: meaning (3 vectors) + keywords (BM25)
        │                       fused with Reciprocal Rank Fusion → top patents
        ▼
 ⑤ LAYER 3  Claim matching ── every input claim vs every claim of those patents
        │                       → closest prior-art claims per claim
        ▼
 ⑥ Legal analysis (EPO problem-solution approach)
        │    features → disclosure check → novelty → inventive step → advancement
        ▼
 ⑦ Examiner-outcome predictor ── LightGBM on the evidence → §102 / §103 / allowed
        │
        ▼
 Report: patent verdict, per-claim verdicts with evidence, distinguishing
         features, predicted examiner outcome
```

Two separate goals are kept apart on purpose:

| Goal | What it produces | How it is judged |
|---|---|---|
| **Legal analysis** | novelty, distinguishing features, inventive-step reasoning with evidence passages | disclosure accuracy (does it find the passage an examiner cites?) |
| **Outcome prediction** | probability of §102 / §103 / allowed | agreement with real examiner decisions (macro-F1, ROC-AUC) |

The predictor reads the legal analysis; it never changes it.

---

## 2. Data

| Dataset | What it is | Used for |
|---|---|---|
| **HUPD** — Harvard USPTO Patent Dataset (Hugging Face `HUPD/hupd`) | ~4.5 M US patent applications 2004–2018 with title, abstract, claims, summary, full description, CPC/IPC labels and the final decision | CPC classifier training; the 100k prior-art corpus |
| **PANORAMA** — LG AI Research (`LG-AI-Research/PANORAMA`, CC BY-NC 4.0) | 8,143 USPTO examination records (2015–2024): claims, examiner-cited prior art, cited paragraphs, per-claim §102/§103/allow decisions | evaluating the analysis (NOC4PC), training the disclosure model (PI4PC) |

**Licence note:** PANORAMA is non-commercial. The trained disclosure model may be
used in this free, non-commercial project but not sold.

### Preprocessing (HUPD)
- **Streaming:** yearly archives (2.6–5.6 GB each) are streamed from Hugging Face
  and only the needed fields are kept — nothing large is stored.
- **Cleaning:** HUPD section markers (`<SOH> … <EOH>`) removed, whitespace normalised.
- **Labels:** CPC codes cut to the 4-character **subclass** (e.g. `A61B100082` → `A61B`).
  CPC only exists from 2013; for 2011–2012 the IPC code is used (same subclass
  system in almost all cases) and marked `label_source = ipc`.
- **Time-based splits** (no random splits — the model must predict codes for
  newer inventions): train 2011–2015, validation 2016, test 2017.
- **Claims:** split into numbered claims; dependent claims linked to their parent.

---

## 3. Document extraction (`analysis/extractor.py`)

Real patent PDFs are two-column scans with an OCR text layer. Plain text
extraction mixes the two columns line by line and keeps the margin line numbers.

What the extractor does:
- rebuilds reading order **column by column** from word positions;
- drops margin line numbers (multiples of 5 on the centre line) without losing
  claim numbers that start the right column;
- repairs OCR damage: removes spaces before punctuation, re-joins words split
  across lines (`opera / tion` → `operation`) using the 100k-patent vocabulary,
  strips running page headers;
- finds sections by their standard headings ("What is claimed is", "Abstract",
  "Summary of the Invention"…) and cuts the abstract before the drawing pages;
- also reads DOCX and TXT, and fetches clean text by **patent number** from
  Google Patents, including the **priority date** (prior art must be older).

**Measured** against the clean Google Patents text:

| Document | Claims found | Claim text identical | Abstract identical |
|---|---|---|---|
| Granted patent PDF (US9968239B2) | 13/13 | 99.6% | 99.8% |
| Published application PDF (US20160220097A1) | 12/12 | 92.8% | 85.5% |
| Word draft | 22/22 | exact | exact |

### Claim parsing (`retrieval/claims.py`, `analysis/elements.py`)
- Claims are split by number (numbers must increase by one, so "about 2. Further"
  inside a claim is not a false split).
- Tolerates office-action noise: `[Claim 1]`, `(Original)`, `18. . The`,
  `19. 19 The`, `1. 00023 An`, `30. ) The`, a lost first letter.
  → 100% of 7,392 PANORAMA validation claims parse.
- Each claim → **preamble + transition word + elements**
  ("A remote controller … **comprising:** [button]; [display section]; …").
  Dependent claims inherit their parent chain's elements.
  Over 2,000 corpus patents: 3.9 elements per independent claim, 1.35 per dependent.

---

## 4. Layer 1 — CPC classifier (`cpc_classifier/`)

**Task:** read title + abstract + first claim + summary and predict the CPC
subclasses (multi-label; a patent usually has several).

**Model:** PatentSBERTa (`AI-Growth-Lab/PatentSBERTa`, MPNet pre-trained on
patents) + mean pooling + a linear layer over 320 subclasses (subclasses with
≥ 50 training examples), sigmoid outputs.

| Version | Training | Notes |
|---|---|---|
| v1 | 100k patents (20k/year 2011–2015), 3 epochs, BCE loss, 384 tokens, Kaggle T4 | |
| **v2** | warm start from v1 + 102k **new** CPC-labelled patents (no overlap with v1), 166,584 total, Asymmetric Loss, early stopping, 2× T4 | IPC rows dropped |

**Test set (2017, unseen):**

| Metric | v1 | **v2** |
|---|---|---|
| Main code in top 1 | 59.4% | **62.7%** |
| Main code in top 3 | 80.6% | **83.6%** |
| Main code in top 5 | 86.1% | **88.4%** |
| Any correct code in top 3 | — | **92.7%** |
| Any correct code in top 5 | — | **95.2%** |

"Any correct code" is the number that matters for Layer 1: the filter keeps a
patent if it shares *any* confirmed code.

**In the app:** the top 5 codes are suggested; the user confirms or edits them.
Patents in those subclasses and filed **before the priority date** are kept;
if fewer than 1,000 remain the filter widens to all older patents.

---

## 5. Layer 2 — patent search (`retrieval/sources.py`)

**Corpus:** 100,000 HUPD applications filed 2012–2016 (20k per year), each with
title, abstract, summary, claims and independent claims. Built and encoded on
Kaggle (T4 GPU, ~35 min).

**Indexes:**
- three PatentSBERTa vectors per patent — *abstract* (title + abstract, 256 tokens),
  *summary* (384), *claims* (independent claims, 384) — stored as float16;
- a BM25 keyword index (`bm25s`) over title + abstract + independent claims.

**Search:**
1. meaning score = 0.3·abstract + 0.2·summary + 0.5·claims cosine similarity
   (claims weigh most: novelty is judged on claims; the summary weight is dropped
   for patents without one);
2. BM25 keyword ranking (top 3,000);
3. the two rankings are merged with **Reciprocal Rank Fusion** (k = 60);
4. top 500 patents (full mode) or 200 (fast mode).

**Checks:** a patent searched with its own text is ranked #1 in 30/30 tests.

**Coverage limit:** only 1.6% of examiner-cited references in PANORAMA are in
this 100k corpus (examiners search everything, including granted patents and
older/newer years). The architecture supports adding sources (APIs) behind the
same `PriorArtSource` interface.

---

## 6. Layer 3 — claim matching (`retrieval/layer3.py`)

Every claim of the invention is compared with every claim of the top patents.

- **Dependent claims get context:** matched as *own text + parent chain*, own
  text first, so if the 192-token limit trims anything it is context, never the
  new feature.
- **Live GPU encoding** (no pre-computed claim index): claims of the top 200
  patents (~3,500 claims) are encoded per search, batched in fp16.
- **Score** = 0.8 × claim similarity + 0.2 × the patent's Layer 2 score.
- Output: top 150 prior-art claims per input claim + the patents that appear
  in many claims' top 20 (likely key references).

**Measured choice:** patents ranked 200–500 supplied < 3% of each claim's top-20
matches, so Layer 3 uses the top 200 (60 s → 17–19 s on an RTX 2050).

---

## 7. Legal analysis — EPO problem-solution approach (`analysis/`)

| Step | Module | What it does |
|---|---|---|
| Features | `features.py` | each element → **component** ("first balloon operation button"), **function** ("swelling or contracting a balloon"), numeric **parameters** ("40–80 kHz") |
| Disclosure | `disclosure.py` | for each feature × reference: P(feature disclosed) from four signals on the 3 closest passages — lexical overlap, syntactic match (same head noun / verb), semantic similarity, and the **disclosure model** (§9); conflicting numeric ranges veto disclosure |
| Novelty | `engine.py` | closest reference = the one disclosing most features. All disclosed → **NOT NOVEL**. Otherwise the missing ones are the **distinguishing features**, labelled *new component* / *same component, new function* / *different parameter* |
| Inventive step | `engine.py` | distinguishing features found in **other** references → **OBVIOUS** (combination); some found nowhere → **INVENTIVE** |
| Advancement | `effects.py` | technical / economic effect sentences in the summary (zero-shot NLI over 7 categories), linked to the distinguishing features they explain |
| Patent level | `engine.py` | patentable if an independent claim is inventive; otherwise dependent claims that are inventive are listed as **amendment candidates** |

Sanity check: every claim of a patent judged against itself → NOT NOVEL (62/62).

---

## 8. Evaluation against real examiner decisions

**PANORAMA NOC4PC** — each claim labelled by the USPTO examiner: rejected
§102 (not novel), rejected §103 (obvious) or allowed. The examiner's cited
references are given (**oracle setting** — measures the analysis, not the search).

- validation: 7,392 claims / 407 applications — all tuning
- test: 2,884 claims / 164 applications — evaluated once
- validation and test share **no** application.

### Results with the first disclosure signal (general NLI)

| Model | Validation (grouped CV) | Test |
|---|---|---|
| Random | — | 0.309 macro-F1 |
| Majority class (§103) | — | 0.224 |
| Rule engine, default thresholds | 0.148 | 0.152 |
| Rule engine, tuned on validation | 0.364 | 0.383 |
| Outcome model (LightGBM, 30 features) | 0.548 | 0.517 (ROC-AUC 0.718) |
| **Outcome model without `n_refs`** | **0.492** | — |

**Leak found with SHAP:** the top feature was `n_refs`, the number of
references the examiner cited — §103 rejections cite several by definition,
so it partly encodes the answer, and at runtime the search always gives 5.
It was removed. The test set had already been used once with it; the next
test evaluation will disclose this.

### Error analysis (validation, out-of-fold, leak-free) — `eval/FINDINGS.md`
- **~65% of errors come from disclosure detection.** For §102 claims, where the
  examiner says one reference discloses every feature, the engine confirmed it
  in only **0.7%** of cases.
- The general NLI model scored **20%** at finding the examiner's paragraph
  (random = 20%): it does not understand patent paraphrase.
- 28% of examiner reasons rely on drawings (not available as text).
- **Ceiling:** in PANORAMA's expert validation, three experts agreed on §102 vs
  §103 only **75.4%** of the time.

---

## 9. The disclosure model (`disclosure_model/`)

**Task:** given a claim's text and a paragraph of a cited patent, is this the
paragraph the examiner cited as disclosing it?

**Data:** PANORAMA **PI4PC** — 5 candidate paragraphs per claim, 1 cited.

**Leakage control:** the PANORAMA tasks were split independently — PI4PC train
shares 268 applications with NOC4PC validation and 121 with NOC4PC test. Every
application in NOC4PC validation/test and PI4PC validation/test was removed:
5,610 → **5,221** applications, 59,508 claims → **253,832 pairs** (23% positive).

**Training:** cross-encoder (claim text, paragraph) → 1 logit, weighted binary
cross-entropy, 1 epoch, text clipped to ~96 + 220 words (≤ 384 tokens), fp16,
plain PyTorch loop on a Kaggle T4.

**PI4PC validation (paragraph identification, top-1 of 5):**

| Scorer | Top-1 | MRR |
|---|---|---|
| Random | 19.4% | 0.448 |
| General NLI alone | 20.0% | 0.456 |
| Dense similarity | 40.5% | 0.622 |
| Previous combined P(disclosed) | 44.8% | 0.656 |
| **New model — PatentSBERTa** | **53.0–53.7%** | **0.709–0.718** |
| New model — NLI-DeBERTa | 54.2% | 0.720 |
| *GPT-4o (PANORAMA paper)* | *~63%* | |

PatentSBERTa was chosen: same quality, smaller and ~1.5× faster on CPU.
It replaces the general NLI signal inside P(disclosed); NLI is kept only for
advancement sentences.

### Recalibration with the new model (validation only)
Validation signals were regenerated with the new model (7,392 claims, 0 errors);
rule weights/thresholds and the outcome model were re-tuned with grouped CV.

**Legal analysis — features found in the examiner's own reference:**

| Examiner outcome | Features found (old → new) | All features found (old → new) |
|---|---|---|
| §102 (one reference discloses everything) | 15.8% → **49.4%** | 0.7% → **15.1%** |
| §103 | 25.4% → 53.9% | 10.5% → 21.4% |
| Allowed | 22.3% → 50.8% | 7.6% → 20.9% |

The engine now finds the features examiners point to about 3× more often, but
coverage rises for all three outcomes alike: the examiner's cited art is the
closest art in every case, and the difference between outcomes lies in details.

**Outcome prediction — macro-F1, grouped CV on validation (leak-free):**

| Model | General NLI signal | Trained disclosure model |
|---|---|---|
| Rule engine, tuned | 0.364 | 0.360 |
| **Outcome model (LightGBM, 29 features, no `n_refs`)** | 0.492 | **0.517** |

Out-of-fold per class (new): §102 F1 0.502, §103 0.585, allowed 0.464.

**Live engine:** uses the tuned settings (`analysis/disclosure_weights.json`:
P(disclosed) ≥ 0.708 counts as disclosed, ≥ 0.592 for the combination check).
On the granted patent US9968239B2 the analysis moved from "13 × not novel"
(untuned) to **6 inventive + 7 obvious → likely patentable**, matching the real grant.

---

## 10. Backend API (`api/main.py`) — module M2

| Endpoint | Purpose |
|---|---|
| `GET /api/health` | `ok` / `loading` / `error`; models load in the background at startup |
| `POST /api/documents` | upload PDF / DOCX / TXT → title, abstract, summary, claims, parsed claims, warnings |
| `POST /api/documents/lookup` | patent number → same fields + priority date (Google Patents) |
| `POST /api/cpc/suggest` | document text → top 5 CPC subclasses with probabilities |
| `POST /api/analyses` | document + confirmed CPC codes + mode (`fast` / `full`) → job id |
| `GET /api/analyses/{id}` | status, stage, percent, elapsed time, full result |

**Design:**
- analyses run as **background jobs**, one at a time, with progress stages
  (searching prior art → extracting features → reading references → judging claims);
- **fast mode** (default when deployed, for CPU hosting): 200 patents in Layer 2,
  80 in Layer 3, 3 references per claim; **full mode**: 500 / 200 / 5;
- **memory:** patent text lives in an on-disk SQLite store (`texts.sqlite`, built
  once in 24 s); only IDs/titles/CPC/dates stay in RAM and the vectors are
  memory-mapped; per-analysis caches are cleared after each job;
- jobs are in memory (lost on restart) — fine for a single-user demo.

**Tested** (`tests/test_api.py`, real models): unsupported file → 400, document
without claims → 400, and a full run (patent lookup → CPC suggestions → fast
analysis) — **26 s** on an RTX 2050 with stage-by-stage progress.

---

## 11. Engineering timeline (what broke and how it was fixed)

- HUPD stream kept almost nothing at first: CPC labels are missing before 2013 → IPC fallback.
- Local GPU (4 GB) too small for training → Kaggle T4; laptop used for inference.
- OCR'd PDFs mixed columns → column-aware extraction.
- NLI over-accepted long claim fragments → gated by topic similarity (later replaced by the trained model).
- Search loaded 3 GB of corpus text into RAM → SQLite text store + memory-mapped vectors.
- Disclosure-model training ran out of GPU memory inside the sentence-transformers trainer → plain PyTorch loop with explicit truncation.
- SHAP exposed the `n_refs` leak → feature removed, test history disclosed.
