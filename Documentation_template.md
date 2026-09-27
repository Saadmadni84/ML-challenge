# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** [Your Team Name]
**Team Members:** [List all team members]
**Submission Date:** [Date]

---

## 1. Executive Summary

We solve cross-source business entity resolution with a **blocking + pairwise-classifier** pipeline: TF-IDF character-n-gram retrieval over two complementary channels (name, address) generates ~77 candidates per S1 entity at a **0.991 recall ceiling**, and a LightGBM classifier on 26 string-similarity + blocking-signal features keeps matches above a validation-tuned F0.5 threshold. On the 20k dev sample the pipeline reaches **val macro-F0.5 0.978** (precision 0.986, recall 0.961). Every stage streams in constant or per-country memory with resume support, so the identical code scales from the sample to the full 26.4M-row dataset.

---

## 2. Methodology

### 2.1 Problem Analysis

Key facts from the data audit (see `dataset_audit_report.md`) that shaped the design:

- **Scale:** 26.4M rows / 2.35 GB. The unblocked test comparison space is **17.27 trillion pairs** (~10M candidates per S1) — a multi-stage blocking architecture is mandatory; no classifier can score that space directly.
- **Unseen-country shift:** train covers only `US` + `India`; test adds **`France` (~14.4% of test records)** and flips the country balance (India 47.3% of test vs 40.0% of train). Consequence: no country one-hots or hard-coded country lists anywhere — per-country logic derives labels from the data at runtime, and all features are country-agnostic.
- **Cross-script gap:** train S1 names are 100% Latin/ASCII, but S2/S3 contain up to ~19% non-ASCII text (Devanagari, Tamil, Gujarati). Pure string matching cannot reach these; they bound the achievable recall and must be measured, not ignored (we report latin-only recall alongside overall recall).
- **Multiplicity + distractors:** ~89% of S1 entities match multiple S2/S3 records (up to 11), so the output is a ranked *set* per S1, not 1:1 linkage. ~26% of S2/S3 records are unlinked distractors. The metric is macro-F0.5 (precision-heavy, β=0.5): false merges hurt more than misses, and correctly-empty singletons score 1.0 — so the decision threshold is tuned directly on validation F0.5, and a strong empty-address/name-mismatch veto matters.
- **Noise patterns:** abbreviations (Corp/Corporation, Pvt/Private, Rd/Road), legal-suffix inconsistency, DBA/trade names, `&` vs `and`, word-order transpositions, typos; addresses with missing components (no PIN/state), transliteration variants, landmark references, component reordering. Character n-grams + fuzzy token ratios are robust to exactly this noise family.

### 2.2 Solution Strategy

**Approach Type:** Blocking + Classifier (retrieval → pairwise gradient boosting → thresholded sets)
**Core Innovation:** Two cheap, complementary retrieval channels whose *union* reaches 0.991 recall at only ~77 candidates/S1, plus **blocking-signal features** (per-channel cosine score/rank/hit) fed back into the matcher — retrieval strength is orthogonal evidence that string similarity alone cannot see (paired val gain +0.0040 F0.5, t=3.45, p<0.01).

Pipeline stages (each a standalone script under `code/business_entity_resolution/src/`):

1. **Blocking** (`blocking.py`): per-country TF-IDF char-n-gram top-K retrieval, name channel + address channel.
2. **Union** (`union_candidates.py`): streaming name∪address merge, dedup by max score (this is `candidate_pairs.tsv` after score-stripping).
3. **Matching** (`train.py` → `predict.py`): LightGBM pairwise classifier; keep pairs with proba ≥ tau (this is `matching_results.tsv`).
4. **Validation**: local macro-F0.5 scorer (`metrics.py`, exact replica of the leaderboard rule) + the official `utils/validate_submission.py` before every upload.

Experiment trail (full log in `EXPERIMENTS.md`, all on the 20k dev sample / 16k-train/4k-val split): Exp0 empty-floor 0.054 → Exp1 exact-name 0.461 → Exp2 blocking ceiling 0.991 → Exp3 LightGBM 0.972 → Exp4 +blocking signals 0.976 → Exp5 scale config (K40, DF caps) + full pipeline 0.978.

---

## 3. Candidate Generation (Blocking)

- **Blocking keys used:** TF-IDF cosine over character n-grams (`char_wb`, (3,4), sublinear TF), min_df=2, max_df=0.05 — computed **within country** (hard country partition, labels derived from data), top-K=40 per channel. Two channels: normalized business name, normalized business address.
- **Candidate pairs generated:** union of name@40 + addr@40, deduped (max score), ≤80/S1 — **avg 77.2/S1** on the sample (≈1.5M pairs for 20k S1s; ≈130M pairs projected for ~1.7M test S1s, vs 17.27T unblocked — a >100,000× reduction).
- **How we ensured true matches were not lost:** (a) channels are complementary by measurement — name-only ceiling 0.860, address-only 0.941, **union 0.9911** (@80; latin-only 0.9950; India 0.9837 / US 0.9960; 97.2% of S1s at full recall); 97% of name-channel Latin misses share an address token, justifying the address channel; (b) conservative K=40/channel (K40≈K50: −0.0013 recall for −20% pairs); (c) DF caps chosen by grid to cost only 0.0007 recall while bounding index memory; (d) exact backend equivalence proven between the C++ `sparse_dot_topn` engine and the sklearn fallback (identical 0.9911). Residual misses decompose as 41% non-Latin targets, 37% empty-address records, 0.2% hard noise — i.e. the ceiling is structural (script/empty data), not ranking quality.
- **Scale engineering:** C++ top-K retrieval with OpenMP; per-country indices (only one in RAM at a time); country-filtered loading; chunked queries with progress/ETA; per-country sharded outputs with `--resume` after disconnects; streaming lockstep union (constant RAM — the naive dict version would need ~10GB+ at full scale).

---

## 4. Matching Model

**Features used (v2, 26 total):**
- Name features: exact flag, RapidFuzz ratio / partial / token-sort / token-set, token Jaccard, trigram Dice, length diff + ratio, token coverage recall + precision (11).
- Address features: exact flag, ratio, token-set, Jaccard, trigram Dice, target-empty flag, numeric-token Jaccard across name+address (7). Address features double as a veto against generic-name false merges.
- Blocking signals (8): per-channel hit/score/rank for name and address, best rank across channels, union size. Top-gain features after adding: `baddr_score`, `bbest_rank`, `bname_score`.

All string features normalize inputs with the shared v1 normalizer (ASCII fold → lowercase → `&`→`and` → punctuation→space → collapse whitespace). No embeddings, no country features, no S1-identity features — the model is fully transferable to unseen France records.

**Model type:** LightGBM binary classifier (lr 0.05, 63 leaves, min_data_in_leaf 200, feature/bagging fraction 0.8, `scale_pos_weight` for the ~4.5% positive rate — no resampling, so the inference distribution stays intact for honest threshold tuning; early stopping on val-pair logloss, deterministic seed).
**Threshold selection method:** coarse-to-fine sweep of tau on **validation macro-F0.5** (the exact leaderboard objective), 0.05–0.95 then ±0.04 refine at 0.01 steps. Sample optimum: **tau=0.95**. The full-data model re-tunes tau on a stratified 200k/20k train/val subset blocked against full train targets.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** **0.9776 validation** (precision 0.9861, recall 0.9609, singleton accuracy 0.9771) on the held-out 4k split; full-sample fit 0.9908. Slice scores track the blocking ceiling (US ≈ 0.987, India ≈ 0.960). Public leaderboard score from the full-data run: [to be filled after Portal upload].
- **Common false positives (wrong merges):** generic/bare names ("Sharma General Store") colliding across distinct businesses when the address is empty or boilerplate on both sides — mitigated by address-veto features and the high tau, but irreducible when both records lack disambiguating text.
- **Common false negatives (missed matches):** (i) non-Latin targets unreachable by design (fold to empty); (ii) empty-address pairs with weak name overlap (DBA/trade-name aliases, heavy abbreviation); (iii) blocking misses (0.9%), dominated by the same two causes. FN:FP ≈ 3:1 at the tuned tau — expected under a precision-heavy metric.
- **France (unseen country) risk:** the model uses no country-specific parameters, but French address formats and transliteration patterns are unvalidated — the first leaderboard score grounds this, and per-country match-rate diagnostics on the submission are the planned follow-up.

---

## 6. Conclusion

A deliberately simple retrieval + gradient-boosting pipeline — dual-channel TF-IDF blocking at 0.991 recall, 26 interpretable features, F0.5-tuned threshold — reaches 0.978 validation F0.5 with a clean scaling story to 10M targets via sharding, streaming, and resume. The key lessons: address retrieval beats name retrieval and their union is nearly lossless; retrieval scores are first-class matcher features; and every scale hazard (RAM, misalignment, interpolation bugs) was caught by requiring the sample run to be output-identical across implementations before the full run.

---

## Appendix

### A. Code Artefacts

Runnable pipeline in `code/business_entity_resolution/` (`src/` + `README.md` + pinned `requirements.txt`). Entry points: `blocking.py` (per-channel candidates) → `union_candidates.py` (merge) → `train.py` (model + `threshold.json`) → `predict.py` (writes both submission files) → `utils/validate_submission.py` (pre-upload gate). Full-scale Colab orchestration: `notebooks/02a_colab_dryrun_and_blocking.ipynb` (dry run → test blocking → unions) and `notebooks/02b_colab_train_and_submit.ipynb` (200k/20k subset → scale-recall check → train → per-country inference → concat → validate). The 20k dev sample (`sample/`, seed 42) and its train/val split (`sample/split/`) ship in the repo; the `README.md` "Sample run" section reproduces every reported sample number.

### B. Additional Results

| Stage | Metric | Value |
|---|---|---|
| Blocking name-only K=100 | recall ceiling | 0.860 (@25: 0.812; latin-only: 0.922) |
| Blocking addr-only K=100 | recall ceiling | 0.941 (@25: 0.923) |
| Union K40+40 | recall ceiling | **0.9911** @ avg 77.2/S1 |
| Exp3 (v1 feats, tau=0.96) | val F0.5 | 0.9723 |
| Exp4 (+blocking signals, tau=0.92) | val F0.5 | 0.9762 (Δ+0.0040, t=3.45) |
| Exp5 final config (tau=0.95) | val F0.5 | **0.9776** (P 0.9861 / R 0.9609) |
| Exp5 full-sample fit | macro F0.5 | 0.9908 |
| Throughput (2 workers) | inference | ~160 S1/s → ~3 h for full test |
| Validator | official script, `--check-ids` | **PASS** |

---

**Note:** Teams can modify sections according to their approach while maintaining clarity and technical depth.
