# Business Entity Resolution ML Competition: Comprehensive Data Audit Report

**Date:** September 2026  
**Target Directory:** `student_resource/`  
**Evaluation Metric:** Macro-averaged $F_{0.5}$ (Precision-heavy: $\beta = 0.5$)  
**Report Type:** Read-Only Exploratory Data Analysis & Integrity Audit  

---

## Executive Summary

A comprehensive, memory-efficient audit was performed across all 7 TSV files of the Business Entity Resolution dataset (totalling **2.35 GB** and **26,435,994 data rows**). The audit verified data schemas, null distributions, entity uniqueness, noise patterns, intra-source duplicates, train-versus-test domain shifts, candidate search space complexity, and leakage vulnerabilities. 

Key structural findings reveal:
1. **Unseen Country Shift:** Training data contains only `US` and `India`, whereas the test set introduces `France` (~14.4% of all test records), alongside a major geographic distribution flip where `India` becomes the dominant country (47.3% in test vs 40.0% in train).
2. **Cross-Script Transliteration Gap:** Reference records (`Source 1`) are 100% Latin/ASCII in train, while `Source 2` and `Source 3` contain up to 18.99% non-ASCII text, including raw Devanagari (Hindi), Tamil, and Gujarati scripts. Traditional string matching without transliteration will fail on Indic records.
3. **Pervasive Intra-Source Duplicates & Distractors:** 89.02% of Source 1 entities match multiple records across Source 2 and Source 3 (up to 11 matches). Furthermore, ~26% of Source 2 and Source 3 records are unlinked distractors that do not match any Source 1 reference.
4. **Exhaustive Comparison Scale:** The unblocked candidate space on the test set is **17.27 trillion pairs** (~9.97 million candidates per S1 entity). Hard country blocking reduces this by 61.1% to 6.72 trillion pairs, establishing that a multi-stage blocking architecture is mandatory before any classifier can be evaluated.

---

## 1. Directory Structure and File Sizes

The dataset is organized cleanly into `dataset/train/` and `dataset/test/` subdirectories:

```
student_resource/
├── dataset/
│   ├── train/
│   │   ├── train_source1.tsv       (210,069,713 bytes | 200.34 MB)
│   │   ├── train_source2.tsv       (489,301,488 bytes | 466.63 MB)
│   │   ├── train_source3.tsv       (503,705,637 bytes | 480.37 MB)
│   │   └── train_ground_truth.tsv  (127,015,583 bytes | 121.13 MB)
│   └── test/
│       ├── test_source1.tsv        (175,022,086 bytes | 166.91 MB)
│       ├── test_source2.tsv        (509,456,422 bytes | 485.86 MB)
│       └── test_source3.tsv        (506,002,772 bytes | 482.56 MB)
├── utils/
│   └── validate_submission.py      (Submission integrity checker)
├── Documentation_template.md       (Methodology write-up template)
└── README.md                       (Competition guidelines)
```

### File Size Summary Table

| Split | File Name | Raw Bytes | Size (MB) | Role in ER Pipeline |
|---|---|---|---|---|
| **Train** | `train_source1.tsv` | 210,069,713 | 200.34 MB | Deduplicated reference entities (Query set) |
| **Train** | `train_source2.tsv` | 489,301,488 | 466.63 MB | Noisy business records (Target source A) |
| **Train** | `train_source3.tsv` | 503,705,637 | 480.37 MB | Noisy business records (Target source B) |
| **Train** | `train_ground_truth.tsv` | 127,015,583 | 121.13 MB | True 1-to-many match mappings |
| **Test** | `test_source1.tsv` | 175,022,086 | 166.91 MB | Inference reference entities (Must predict for all) |
| **Test** | `test_source2.tsv` | 509,456,422 | 485.86 MB | Inference target source A |
| **Test** | `test_source3.tsv` | 506,002,772 | 482.56 MB | Inference target source B |
| **Total** | **All 7 TSVs** | **2,520,573,701** | **2,403.80 MB (~2.35 GB)** | |

*Why it matters:* Storing the uncompressed data in naive in-memory structures (e.g. raw Pandas dataframes across all sources) exceeds available RAM (8 GB hardware). Pipelines must stream or chunk files and utilize memory-efficient primitive structures.

---

## 2. Row Counts Across All TSVs

All TSVs contain a single header row followed by tab-delimited records:

| File Name | Total Lines | Header Lines | Data Rows | % of Total Data |
|---|---|---|---|---|
| `train_ground_truth.tsv` | 2,206,822 | 1 | 2,206,821 | 8.35% |
| `train_source1.tsv` | 2,206,822 | 1 | 2,206,821 | 8.35% |
| `train_source2.tsv` | 5,034,617 | 1 | 5,034,616 | 19.04% |
| `train_source3.tsv` | 5,285,604 | 1 | 5,285,603 | 20.00% |
| `test_source1.tsv` | 1,732,545 | 1 | 1,732,544 | 6.55% |
| `test_source2.tsv` | 4,887,274 | 1 | 4,887,273 | 18.49% |
| `test_source3.tsv` | 5,082,317 | 1 | 5,082,316 | 19.22% |
| **Total Training** | **14,733,865** | **4** | **14,733,861** | **55.73%** |
| **Total Test** | **11,702,136** | **3** | **11,702,133** | **44.27%** |
| **Grand Total** | **26,436,001** | **7** | **26,435,994** | **100.00%** |

*Why it matters:* Notice the perfect 1:1 row alignment between `train_source1.tsv` and `train_ground_truth.tsv` (exactly 2,206,821 rows each). In the test set, exactly 1,732,544 predictions must be generated in `matching_results.tsv`. Any missing or extra row causes instant disqualification by the official evaluator.

---

## 3. Column Names & Schemas

### Source Files (`*_source1.tsv`, `*_source2.tsv`, `*_source3.tsv`)
All 6 source files strictly follow a homogeneous 4-column schema:
1. `entity_id`: Record primary key with prefix `S1-`, `S2-`, or `S3-`.
2. `business_name`: Name string (legal names, trading styles, abbreviations, typos, multi-script).
3. `business_address`: Full or partial address string (street, landmark, city, postal code, state).
4. `country`: Geographic ISO/country identifier (`US`, `India`, `France`).

### Ground Truth File (`train_ground_truth.tsv`)
Follows a 2-column relation schema:
1. `source1_entity_id`: Foreign key matching `train_source1.tsv` (`S1-XXXXXX`).
2. `matched_entity_ids`: Comma-delimited list of matching `S2-*` and `S3-*` IDs (empty string for singletons).

*Why it matters:* The schema is clean and uniform across splits. No ad-hoc column realignment is required between train and test.

---

## 4. Inferred Data Types & Characteristics

| Column | Logical Type | Storage Format | Value Constraints & Patterns |
|---|---|---|---|
| `entity_id` | Identifier | String (ASCII) | `^S[1-3]-[0-9]+$` (e.g., `S1-925783039`) |
| `business_name` | Text | UTF-8 String | ASCII, Devanagari, Tamil, Gujarati, Latin-1 accented |
| `business_address` | Text | UTF-8 String | Multi-component free text; can be blank |
| `country` | Categorical | String (ASCII) | Train: `{'US', 'India'}` \| Test: `{'US', 'India', 'France'}` |
| `matched_entity_ids` | ID List | UTF-8 String | Comma-separated list without quotation or whitespace |

---

## 5. Missing Values Audit

| Split | File Name | `entity_id` Nulls | `business_name` Nulls | `business_address` Nulls | `country` Nulls |
|---|---|---|---|---|---|
| Train | `train_source1.tsv` | 0 (0.00%) | 0 (0.00%) | **0 (0.00%)** | 0 (0.00%) |
| Train | `train_source2.tsv` | 0 (0.00%) | 0 (0.00%) | **168,967 (3.356%)** | 0 (0.00%) |
| Train | `train_source3.tsv` | 0 (0.00%) | 0 (0.00%) | **175,916 (3.328%)** | 0 (0.00%) |
| Test | `test_source1.tsv` | 0 (0.00%) | 0 (0.00%) | **0 (0.00%)** | 0 (0.00%) |
| Test | `test_source2.tsv` | 0 (0.00%) | 0 (0.00%) | **129,408 (2.648%)** | 0 (0.00%) |
| Test | `test_source3.tsv` | 0 (0.00%) | 0 (0.00%) | **136,098 (2.678%)** | 0 (0.00%) |
| **Total** | **All Sources** | **0 (0.00%)** | **0 (0.00%)** | **610,389 (2.31%)** | **0 (0.00%)** |

*Why it matters:* 
- **Source 1 is complete:** Both `train_source1` and `test_source1` have zero missing fields. It acts as an authoritative reference catalog.
- **Address is the only field with missing values:** In Sources 2 and 3, ~2.6% to 3.4% of entities have an empty address string (`""`). 
- **Blocking vulnerability:** Any blocking rule that relies strictly on address components (e.g. zip code or city matching) will automatically drop 610,389 records, creating an immediate recall ceiling penalty. Fallback name-only blocking channels are indispensable.

---

## 6. Duplicate Rows Audit

Every row in all 7 files was hashed and checked for full-line duplicate records:
- `train_source1.tsv`: **0 duplicate rows**
- `train_source2.tsv`: **0 duplicate rows**
- `train_source3.tsv`: **0 duplicate rows**
- `train_ground_truth.tsv`: **0 duplicate rows**
- `test_source1.tsv`: **0 duplicate rows**
- `test_source2.tsv`: **0 duplicate rows**
- `test_source3.tsv`: **0 duplicate rows**

*Why it matters:* There are no corrupted copy-paste rows or redundant records in the raw files.

---

## 7. Entity ID Uniqueness Audit

- `train_source1.tsv`: 2,206,821 unique IDs (0 duplicates, 100.0% unique)
- `train_source2.tsv`: 5,034,616 unique IDs (0 duplicates, 100.0% unique)
- `train_source3.tsv`: 5,285,603 unique IDs (0 duplicates, 100.0% unique)
- `test_source1.tsv`: 1,732,544 unique IDs (0 duplicates, 100.0% unique)
- `test_source2.tsv`: 4,887,273 unique IDs (0 duplicates, 100.0% unique)
- `test_source3.tsv`: 5,082,316 unique IDs (0 duplicates, 100.0% unique)

*Why it matters:* Every record possesses a strictly unique primary key within its source.

---

## 8. Country Distribution Analysis

### Training Set

| Country | `train_source1` Count (%) | `train_source2` Count (%) | `train_source3` Count (%) |
|---|---|---|---|
| **US** | 1,323,633 (59.98%) | 3,016,817 (59.92%) | 3,170,056 (59.98%) |
| **India** | 883,188 (40.02%) | 2,017,799 (40.08%) | 2,115,547 (40.02%) |
| **France** | *0 (0.00%)* | *0 (0.00%)* | *0 (0.00%)* |
| **Total** | 2,206,821 (100.0%) | 5,034,616 (100.0%) | 5,285,603 (100.0%) |

### Test Set

| Country | `test_source1` Count (%) | `test_source2` Count (%) | `test_source3` Count (%) |
|---|---|---|---|
| **India** | 809,986 (46.75%) | 2,312,565 (47.32%) | 2,405,000 (47.32%) |
| **US** | 663,106 (38.27%) | 1,871,330 (38.29%) | 1,945,701 (38.28%) |
| **France** | **259,452 (14.98%)** | **703,378 (14.39%)** | **731,615 (14.40%)** |
| **Total** | 1,732,544 (100.0%) | 4,887,273 (100.0%) | 5,082,316 (100.0%) |

### Critical Finding: Covariate & Concept Shift
1. **Unseen Country (France):** France accounts for **1,694,445 records** in the test set (14.4% across all sources), but **zero** in training!
2. **Dominant Country Flip:** In training, the US is the clear majority (60.0% vs 40.0%). In test, India is the majority (47.3% vs 38.3%).
3. **Implications:** 
   - Feature engineering must not rely on country-specific hardcoded dictionaries (e.g. US state lists or Indian PIN code lookups only).
   - The validation strategy must simulate out-of-country generalization (e.g. leave-one-country-out validation or country-stratified evaluation).

---

## 9. Business Name Statistics & Linguistic Patterns

### Name Length & Diversity Metrics

| File | Distinct Names | Empty | Min Len | Max Len | Mean Len | Median | Std Dev | P90 | P99 |
|---|---|---|---|---|---|---|---|---|---|
| `train_source1` | 1,539,229 | 0 | 3 | 105 | 24.03 | 24 | 7.74 | 34 | 42 |
| `train_source2` | 4,402,009 | 0 | 2 | 104 | 25.10 | 25 | 8.89 | 37 | 48 |
| `train_source3` | 4,651,609 | 0 | 2 | 123 | 25.20 | 25 | 9.49 | 37 | 50 |
| `test_source1` | 1,238,867 | 0 | 3 | 92 | 23.84 | 24 | 7.67 | 34 | 42 |
| `test_source2` | 4,311,041 | 0 | 2 | 102 | 25.70 | 25 | 9.12 | 38 | 49 |
| `test_source3` | 4,521,929 | 0 | 2 | 103 | 25.66 | 25 | 9.57 | 38 | 50 |

### Multi-Script & Character Complexity Breakdown

| File | Non-ASCII Names (%) | Devanagari Names (%) | Names with Digits (%) |
|---|---|---|---|
| `train_source1` | **0 (0.00%)** | **0 (0.00%)** | 35,585 (1.61%) |
| `train_source2` | **764,608 (15.19%)** | **269,424 (5.35%)** | 255,636 (5.08%) |
| `train_source3` | **606,737 (11.48%)** | **158,003 (2.99%)** | 270,847 (5.12%) |
| `test_source1` | **40,789 (2.35%)** *(French accents)* | **0 (0.00%)** | 20,207 (1.17%) |
| `test_source2` | **928,158 (18.99%)** | **309,103 (6.32%)** | 190,009 (3.89%) |
| `test_source3` | **737,515 (14.51%)** | **181,068 (3.56%)** | 202,470 (3.98%) |

### Qualitative Business Name Patterns

| Category | Examples from Dataset | ER Pipeline Implication |
|---|---|---|
| **Transliterated Indic Scripts** | `S1`: `Raj Investments LLP`<br>`S2`: `ராஜ் இன்வெஸ்ட்மெண்ட்ஸ் எல்எல்பி` (Tamil)<br>`S3`: `Raj Investments எல்எல்பி` (Hybrid)<br>`S2`: `आनंद फाउंडेशन प्राइवेट लिमिटेड` (Hindi) | Character n-grams and edit distance will yield 0 similarity between Latin S1 and Indic S2/S3. Transliteration (e.g. Indic-Xlit / transliteration rules) or multilingual character embeddings are essential. |
| **Punctuation & Legal Noise** | `M/s GOOD ESTATE LIMITED SERVICE`<br>`*** Yamuica Venus Pvt Ltd`<br>`<< Team Ecole`<br>`(Inc.) Green Patriot Natural`<br>`#yarbroughharrell` | Punctuation stripping, case normalization, and legal suffix stripping (`Pvt Ltd`, `LLC`, `SARL`, `Inc`, `Corp`) are required before blocking. |
| **Appended Metadata in Names** | `wilfordhancock.com` (domain name)<br>`Manikkam Nagar Medical C0llege Road Infrastructures Private Limited \| www.manikkamn.com`<br>`Internal Medicine Partners... - 9978643056` (phone) | Noise injection includes phone numbers, URLs, and address fragments embedded directly inside business name fields. |
| **Severe Typographical Perturbations** | `Maure Williams Colombier Inc` vs `Maure Wilblims Colombier Inc`<br>`Payne Enterprises` vs `Payne Etrepndiels` | Standard token matching fails on severe character misspellings; soft fuzzy token matching (Jaro-Winkler, Levenshtein, bi-gram Jaccard) is needed. |

---

## 10. Business Address Statistics & Structural Noise

### Address Length Metrics

| File | Empty Addresses (%) | Min Len | Max Len | Mean Len | Median | Std Dev | P90 | P99 |
|---|---|---|---|---|---|---|---|---|
| `train_source1` | 0 (0.00%) | 11 | 256 | 52.07 | 41 | 25.33 | 90 | 124 |
| `train_source2` | 168,967 (3.36%) | 0 | 249 | 46.23 | 37 | 24.84 | 83 | 118 |
| `train_source3` | 175,916 (3.33%) | 0 | 240 | 46.71 | 42 | 21.65 | 77 | 115 |
| `test_source1` | 0 (0.00%) | 11 | 268 | 57.21 | 50 | 25.03 | 93 | 126 |
| `test_source2` | 129,408 (2.65%) | 0 | 269 | 50.41 | 43 | 25.35 | 87 | 120 |
| `test_source3` | 136,098 (2.68%) | 0 | 267 | 48.74 | 43 | 22.64 | 81 | 117 |

### Qualitative Address Variations

1. **Token Inversion and Reordering:**
   - S1: `630 45th Terrace, Kansas City, MO`
   - S2 Match: `KANSAS CITY, MO, 630 45ND TERRACE, null`
   - S2 Match 2: `45ND TERRACE, null, KANSAS CITY, MO`
   - *Observation:* Components appear in reverse hierarchy (City/State preceding Street Address), with literal artifact tokens like `"null"`.
2. **Landmark-Based Indian Formatting:**
   - `BLOCK D-78 FLAT NO-B1, SAI ENCLAVE, PLOT NO-211 BACKSIDE INDIAN OIL STATE OFFICE, CHANDRA, SEKHARPUR, BHUBANESWAR, Odisha`
   - Indian addresses rely heavily on premises names, landmarks (`Opp City Union Bank`, `Behindkalpataru Bldg`), and multi-tiered administrative blocks rather than standard street numbers.
3. **Ultra-Short Truncated Addresses:**
   - Source 2 and 3 contain single-token city/state snippets: `DELHI`, `BORDEAUX`, `CALAIS`, `103, MH`, `Calais, 62`.
4. **State Code Ambiguity:**
   - Common abbreviations: `MH` (Maharashtra), `DL` (Delhi), `TN` (Tamil Nadu), `NY` (New York), `TX` (Texas), `62` (Pas-de-Calais, France).

---

## 11. Ground-Truth Statistics & Match Distributions

Analysis of all **2,206,821** entries in `train_ground_truth.tsv`:

### Match Cardinality Distribution

| Match Count per S1 Entity | Number of S1 Entities | Percentage | Cumulative % |
|---|---|---|---|
| **0 matches (Singletons)** | **123,247** | **5.58%** | 5.58% |
| **1 match** | **119,157** | **5.40%** | 10.98% |
| **2 matches** | **375,212** | **17.00%** | 27.99% |
| **3 matches** | **530,841** | **24.05% (Mode)** | 52.04% |
| **4 matches** | **484,115** | **21.94%** | 73.98% |
| **5 matches** | **321,957** | **14.59%** | 88.57% |
| **6 matches** | **164,868** | **7.47%** | 96.04% |
| **7 matches** | **63,968** | **2.90%** | 98.94% |
| **8 matches** | **18,680** | **0.85%** | 99.78% |
| **9 matches** | **4,205** | **0.19%** | 99.97% |
| **10 matches** | **534** | **0.02%** | 100.00% |
| **11 matches** | **37** | **0.00%** | 100.00% |
| **Total S1 Entities** | **2,206,821** | **100.00%** | |

- **Mean matches per S1 entity:** **3.4613**
- **Median matches per S1 entity:** **3**
- **Singletons (0 matches):** 123,247 entities (5.58%)
- **Multiple matches:** 1,964,417 entities (**89.02%**)

### Breakdown by Target Source

| Target Source | Total Matches | % of All Matches | Mean Matches per S1 | Max Matches for a single S1 |
|---|---|---|---|---|
| **Source 2** | 3,693,619 | 48.36% | 1.6737 | 5 |
| **Source 3** | 3,944,746 | 51.64% | 1.7875 | 6 |
| **Combined** | **7,638,365** | **100.00%** | **3.4613** | **11** |

### Intra-Source Target Distribution per S1 Entity

| Count | S1 with this many S2 matches | % | S1 with this many S3 matches | % |
|---|---|---|---|---|
| 0 | 287,745 | 13.04% | 266,276 | 12.07% |
| 1 | 789,108 | 35.76% | 716,417 | 32.46% |
| 2 | 652,779 | 29.58% | 668,375 | 30.29% |
| 3 | 333,957 | 15.13% | 372,443 | 16.88% |
| 4 | 119,078 | 5.40% | 145,116 | 6.58% |
| 5 | 24,154 | 1.09% | 35,378 | 1.60% |
| 6 | 0 | 0.00% | 2,816 | 0.13% |

*Why it matters:* 
- **The Evaluation Metric Impact:** $F_{0.5}$ is macro-averaged across *all* S1 entities. For the 5.58% singletons, predicting an empty list scores 1.0; predicting any false match immediately drops the score for that entity to 0.0. Conservative thresholding on uncertain entities directly boosts the leaderboard score.
- **Many-to-One Topology:** 89% of S1 entities map to multiple records. Any matching architecture that assumes a 1:1 marriage between S1 and S2 (such as Hungarian algorithm or mutual nearest neighbors) is fundamentally mismatched to this ground truth topology.

---

## 12. Intra-Source Duplicate & Distractor Analysis

### Intra-Source Duplicate Records
Ground truth mapping proves that **Source 2 and Source 3 are NOT internally deduplicated**:
- **1,129,968 S1 entities** have $\ge 2$ matches inside Source 2 alone.
- **1,224,128 S1 entities** have $\ge 2$ matches inside Source 3 alone.
- *Structural Cause:* Sources 2 and 3 capture redundant corporate records, DBA alias registrations, branch locations, and dirty duplicates of the same legal business.

### Distractor / Unmatched Records in Sources 2 & 3

| Source File | Total Records in File | Matched Records in Ground Truth | Unmatched Distractors | % Distractors |
|---|---|---|---|---|
| `train_source2` | 5,034,616 | 3,693,619 | **1,340,997** | **26.64%** |
| `train_source3` | 5,285,603 | 3,944,746 | **1,340,857** | **25.37%** |
| **Combined** | **10,320,219** | **7,638,365** | **2,681,854** | **25.99%** |

*Why it matters:* Over **2.68 million records** in the training targets never match any reference entity. Models cannot simply assign every S2/S3 entity to the closest S1 entity; a robust rejection mechanism (high confidence thresholding) is critical to prevent false-positive inflation.

---

## 13. Leakage, ID Distributions, & Anomaly Audits

1. **Entity ID Overlap:**
   - Overlap between Train S1 and Test S1: **0**
   - Overlap between Train S2 and Test S2: **0**
   - Overlap between Train S3 and Test S3: **0**
   - *Verdict:* ID sets are 100% disjoint. No ID leakage.
2. **Entity ID Formats and Numeric Ranges:**
   - Train S1 numeric suffix: `[23,933, 999,998,822]`, mean = `500,732,155`
   - Test S1 numeric suffix: `[7,217, 999,998,871]`, mean = `499,856,783`
   - Ground truth check: S1-965667 matches S2-681193310 and S3-775321672. The integer IDs are uniformly distributed random hashes. There is no mathematical or ordinal correlation between matched IDs.
3. **Cross-Country Match Invariance:**
   - Over all **7,638,365 ground truth matches** across all 2,206,821 S1 entities:
   - **Cross-Country Matches: Exactly 0 (0.00%)**
   - *Verdict:* Matching is strictly bounded by geographic borders. An entity in India never matches an entity in the US or France. Hard country blocking has 100.0% empirical recall safety.
4. **Verbatim Business Name Recurrence:**
   - 194,461 business names in `test_source1` (15.70%) appear verbatim in `train_source1` (e.g. common business titles like "Apex Enterprises", "First National Bank", "Sai Medical").
   - *Vulnerability:* Memorizing training entities or training nearest-neighbor embeddings without address validation will generate severe false merges across distinct businesses sharing generic commercial names.

---

## 14. Train vs. Test Comprehensive Comparison

| Dimension | Training Set | Test Set | Difference & Shift | ER Pipeline Impact |
|---|---|---|---|---|
| **S1 Entities (Query)** | 2,206,821 | 1,732,544 | -21.5% | Test inference is slightly smaller than train |
| **S2 Records (Target)** | 5,034,616 | 4,887,273 | -2.9% | Target pool scale remains ~5M records |
| **S3 Records (Target)** | 5,285,603 | 5,082,316 | -3.8% | Target pool scale remains ~5M records |
| **Total Records** | 12,527,040 | 11,702,133 | -6.6% | ~11.7M records to process during inference |
| **Countries Present** | `US`, `India` | `US`, `India`, `France` | **+ France (14.4%)** | Zero-shot country transfer required |
| **Country Distribution** | US: 60.0%<br>India: 40.0%<br>France: 0.0% | India: 47.3%<br>US: 38.3%<br>France: 14.4% | India +7.3%<br>US -21.7%<br>France +14.4% | Severe covariate shift; India dominates test |
| **Non-ASCII in S1** | 0 (0.00%) | 40,789 (2.35%) | +2.35% | S1 now contains accented French text |
| **Devanagari in Targets** | S2: 5.35%<br>S3: 2.99% | S2: 6.32%<br>S3: 3.56% | +18% relative | Indic script matching is even more critical in test |
| **Missing Address Rate** | S2: 3.36%<br>S3: 3.33% | S2: 2.65%<br>S3: 2.68% | -0.7% | Consistent missing address pattern across splits |

---

## 15. Candidate Generation Scale & Complexity

The candidate generation (blocking) stage must reduce the full Cartesian comparison space into a sparse candidate graph before pair scoring:

### 1. Exhaustive Cartesian Product Scale
- **Train Space:** $|S_1| \times (|S_2| + |S_3|) = 2,206,821 \times 10,320,219 =$ **22,774,875,183,499** (~22.77 trillion pairs)
- **Test Space:** $|S_1| \times (|S_2| + |S_3|) = 1,732,544 \times 9,969,589 =$ **17,272,749,271,416** (~17.27 trillion pairs)
- **Exhaustive Candidates per S1 Entity:** **9,969,589 candidates**

### 2. Space Reduction via Hard Country Blocking
Since cross-country matches are strictly 0.0%, hard blocking on `country == country` yields:
- **India:** $809,986 \times (2,312,565 + 2,405,000) =$ 3,821,161,957,090 pairs
- **US:** $663,106 \times (1,871,330 + 1,945,701) =$ 2,531,096,177,286 pairs
- **France:** $259,452 \times (703,378 + 731,615) =$ 372,311,803,836 pairs
- **Country-Blocked Test Space:** **6,724,569,938,212 pairs** (~6.72 trillion pairs)
- **Immediate Space Reduction:** **61.07% reduction** with **0.00% recall loss**

### 3. Required Multi-Stage Blocking Target
Even 6.72 trillion pairs is orders of magnitude beyond the throughput of ML classifiers or dense bi-encoders. 
- To achieve inference within competition runtime limits, blocking must reach a **Reduction Ratio $\ge 99.999\%$**, generating an average of **30 to 80 candidates per S1 entity**.
- Total candidate pairs fed to classifier: **~50M to 140M pairs**.

---

## 16. Unusual & Important Dataset Characteristics

1. **The Language Script Asymmetry:**
   Source 1 contains pure English transliterations for Indian entities (e.g. `Raj Investments LLP`), while Source 2 and Source 3 contain native scripts (Tamil: `ராஜ் இன்வெஸ்ட்மெண்ட்ஸ் எல்எல்பி`, Hindi: `आनंद फाउंडेशन प्राइवेट लिमिटेड`). A text matching pipeline that does not transliterate native Indic scripts to Latin characters will automatically lose up to 10% recall on the Indian partition.
2. **Noise Injection Artifacts:**
   Addresses contain literal string artifacts such as `", null"`, concatenated phrases (`RoadHeightsHoldings`), and inverted postal hierarchies.
3. **Severe Macro-Averaging Penalty on Singletons:**
   Singletons represent 5.58% of the dataset. Because metric evaluation is macro-averaged over all S1 entities, emitting even one incorrect candidate for a singleton turns a perfect score of 1.0 into 0.0.
4. **Precision-Dominant Objective ($F_{0.5}$):**
   $F_{0.5}$ weights precision twice as heavily as recall:
   $$F_{0.5} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$
   A false merge causes twice the penalty of a missed match. The candidate filter must prioritize clean, high-precision boundaries over loose candidate expansion.

---

## Part A. Concise Data Audit Table

| Item | Finding | Why It Matters for an Entity Resolution ML Pipeline |
|---|---|---|
| **Total Rows & Size** | 26.4M rows, 2.35 GB across 7 TSVs; 11.7M rows in test. | Naive loading causes OOM crashes. Pipelines must be streaming, memory-efficient, and chunk-oriented. |
| **Test Set Country Shift** | Test introduces `France` (14.4% of test) with zero training examples; `India` flips to 47.3% majority. | Rule-based or geographic dictionaries hardcoded to US/India will fail. Zero-shot country generalization is required. |
| **Missing Addresses** | 0.0% in Source 1; ~3.3% missing in Train S2/S3; ~2.6% missing in Test S2/S3. | Hard address blocking rules (e.g. postal code or street matching) will discard >610,000 entities. Multi-channel blocking with address-free fallbacks is mandatory. |
| **Script Asymmetry** | Source 1 is 100% Latin/ASCII in train, while Targets contain Devanagari, Tamil, and Gujarati. | Character edit distance and token overlap between S1 and Indic targets score 0. Automated script transliteration is essential to unlock candidate recall. |
| **Ground Truth Topology** | 89.02% of S1 entities have multiple matches (mean 3.46, mode 3, max 11); 5.58% singletons. | One-to-one matching algorithms (Hungarian matching, reciprocal nearest neighbor) are invalid. Multi-label, threshold-based clustering is required. |
| **Distractor Ratio** | ~26% of S2 and S3 records (2.68M records in train) never match any S1 reference. | Classifiers cannot assume every target entity has a reference match. High-confidence classification thresholds are vital to prevent false positive inflation. |
| **Cross-Country Border** | 0.00% cross-country matches across all tested ground truth pairs. | `country` is an inviolable hard blocking key, safely cutting 61.1% (10.5 trillion pairs) of the comparison space without recall loss. |
| **Disjoint ID Sets** | Zero ID overlap between train and test; IDs are pseudorandom uniform integers. | No ID-based leakage exists; pipelines must rely purely on semantic and geographic attribute matching. |
| **Candidate Search Space** | Exhaustive Cartesian space is 17.27 trillion pairs in test (~9.97M candidates per S1). | Exhaustive pair scoring is computationally impossible. A high-throughput, multi-key blocking phase is required to reduce candidates to $\le 100$ per S1 entity. |
| **Precision-Heavy Metric** | Evaluation uses Macro $F_{0.5}$ (Precision weighted $2\times$ over Recall). | False merges penalize twice as heavily as false negatives. Classification thresholds must be optimized for precision. |

---

## Part B. Recommended Next Steps

> [!IMPORTANT]
> In accordance with competition guidelines, **do NOT jump to training advanced deep neural networks or heavy transformers yet**. The dataset audit proves that the primary bottleneck is candidate generation and leakage-safe validation.

### Recommended Immediate Actions:

1. **Establish a Leakage-Safe Local Validation Split:**
   - Construct a representative validation set from `dataset/train/` that mimics test set conditions.
   - **Crucial:** Because Test contains an unseen country (`France`), the local validation protocol should incorporate a **Country-Split Validation** (e.g. train on US, validate on India, or synthesize a held-out sub-region/state) alongside a standard entity-disjoint split.
   - Ground truth clusters must never be fractured across train and validation folds.
2. **Design a High-Recall, Multi-Pass Blocking Strategy:**
   - **Pass 0 (Hard Constraint):** Partition by `country` (100% intra-country).
   - **Pass 1 (Transliterated Token Blocking):** Transliterate non-Latin scripts (Devanagari, Tamil, Gujarati) into Latin characters using standard phonetic transliteration.
   - **Pass 2 (Standardized Legal & Name Signatures):** Strip corporate noise (`LLC`, `Pvt Ltd`, `SARL`, `Inc`, `M/s`, `***`), extract high-information name prefix tokens (e.g., first 3 clean words or 3-gram minhash signatures).
   - **Pass 3 (Address & Postal Tokens):** Block on extracted PIN/ZIP codes and standardized city/locality tokens.
   - **Pass 4 (Union & Deduplication):** Union candidates across passes and measure the **Reduction Ratio** (aiming for $\le 50-100$ candidates per S1) and **Recall Ceiling** (aiming for $\ge 98\%$ recall on the validation set).
3. **Implement the Macro $F_{0.5}$ Scorer & Candidate Validator:**
   - Implement an exact replica of the competition macro $F_{0.5}$ metric in Python, verifying it against the toy example in the README (Precision=0.667, Recall=1.0 $\to$ $F_{0.5}=0.714$).
   - Integrate `utils/validate_submission.py` into local pipelines to guarantee zero formatting rejections.
