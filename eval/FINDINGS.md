# Error analysis — findings (validation only)

All numbers are on PANORAMA NOC4PC **validation**, using out-of-fold predictions
of the leak-free outcome model (no `n_refs`). The test set was not used.
Script: `eval/error_analysis.py` → `eval/results/error_analysis.json`.

## Baseline being analysed
Leak-free GBM, grouped 5-fold OOF: macro-F1 **0.495**, accuracy 0.502
(§102 F1 0.487, §103 0.539, allowed 0.461).

## Where the 150 sampled errors come from

| Stage | Share |
|---|---|
| **Disclosure detection — missed** (§103 features not found in any ref) | 42.0% |
| **Disclosure detection — missed** (§102 features look undisclosed in the cited ref) | 18.7% |
| Disclosure detection — false positive (allowed, all features "found") | 4.0% |
| Classifier / decision | 30.7% |
| Label/data (double patenting etc.) | 2.7% |
| Feature extraction | 1.3% |
| Evidence gap (cited paragraphs missing) | 0.7% |

**≈65% of errors originate in disclosure detection.**

## The disclosure signal is near-uninformative
Share of features judged disclosed (P ≥ 0.5), all validation claims:

| Examiner outcome | Covered in examiner's primary ref | All features covered in primary ref |
|---|---|---|
| §102 (one ref discloses everything) | 15.8% | **0.7%** |
| §103 | 25.4% | 10.5% |
| Allowed | 22.3% | 7.6% |

For §102 claims the examiner states a single reference discloses every feature;
our engine confirms that in 0.7% of cases, and allowed claims score *higher* than §102.

## Why (manual inspection)
1. General-domain NLI (MNLI-trained DeBERTa) gives ~0.00–0.05 entailment on
   technical paraphrases examiners treat as clear disclosure.
2. Semantic similarity is 0.5–0.7 for both disclosed and undisclosed features.
3. 28% of examiner reasons rely on figures / reference numerals (drawings are not available as text).
4. For dependent claims the examiner addresses only the *added* feature; we weight
   inherited features equally.

## Retrieval coverage
Only **14 of 889 (1.6%)** examiner-cited references are in our 100k index
(cited: 725 published applications 2011–2022, 164 granted patents; index: applications 2012–2016).
End-to-end Recall@K against examiner citations is therefore bounded near zero by coverage,
not by the ranking algorithm.

## Ceiling evidence
PANORAMA expert validation: three experts agreed on reject/allow 91.1% of the time
and on the section code (§102 vs §103 …) only **75.4%**. A 3-class macro-F1 of
0.85–0.90 would exceed expert agreement on the labels.
