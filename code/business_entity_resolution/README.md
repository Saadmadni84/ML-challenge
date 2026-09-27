# Business Entity Resolution — runnable pipeline

Self-contained pipeline for ML Challenge 2026. Anyone should be able to
regenerate `output/matching_results.tsv` and `output/candidate_pairs.tsv`
from the training/test data using only this folder + the dataset.

## Layout

```
code/business_entity_resolution/
├── src/
│   ├── normalize.py        # shared v1 text normalizer
│   ├── metrics.py          # local replica of the leaderboard macro-F0.5 scorer
│   ├── audit.py            # streaming, low-RAM data auditor (stdlib only)
│   ├── sample.py           # stratified dev-sample builder
│   ├── split.py            # entity-disjoint stratified train/val split
│   ├── baseline_exact.py   # Exp1 normalized exact-name baseline
│   ├── blocking.py         # TF-IDF char-ngram blocking (topn/loop, sharded, resumable)
│   ├── blocking_recall.py  # recall-ceiling report for a candidates file
│   ├── union_candidates.py # streaming name+address union (lockstep, constant RAM)
│   ├── features.py         # pair features v1 (strings) + v2 (+blocking signals)
│   ├── train.py            # LightGBM + val-F0.5 threshold sweep
│   └── predict.py          # batched inference -> matching + candidate files
├── README.md               # this file (run instructions)
└── requirements.txt        # dependencies (pinned before final submission)
```

## Environment

```bash
pip install -r code/business_entity_resolution/requirements.txt
```

## Full run (what the Colab notebooks 02a/02b execute)

Locked config: char n-grams (3,4), min_df=2, max_df=0.05, K=40/channel,
union k-per-file=40, LightGBM v2 features, tau from `threshold.json`.

```bash
S=code/business_entity_resolution/src
CFG="--k 40 --min-ngram 3 --max-ngram 4 --min-df 2 --max-df 0.05"

# 1) Test blocking, per country+field (resumable; countries from the data):
python3 $S/blocking.py --s1 dataset/test/test_source1.tsv \
  --s2 dataset/test/test_source2.tsv --s3 dataset/test/test_source3.tsv \
  --field name --out output/test_{country}_name.tsv $CFG \
  --countries India --resume
# ... repeat for (US, France) x (name, address)

# 2) Per-country unions (streaming):
python3 $S/union_candidates.py --a output/test_India_name.tsv \
  --b output/test_India_addr.tsv --k-per-file 40 \
  --out output/test_union_India.tsv
# ... repeat per country

# 3) Train-subset blocking vs full train targets, then split by id
#    (see notebook 02b cells 2-3 for the stratified 200k/20k sampling):
python3 $S/blocking.py --s1 subset/sub220k_s1.tsv \
  --s2 dataset/train/train_source2.tsv --s3 dataset/train/train_source3.tsv \
  --field name --out output/sub220k_name_{country}.tsv $CFG \
  --countries all --resume
# ... address channel, unions, then split into cands_train200k_*/cands_val20k_*

# 4) Scale-recall check (expect ~0.99 @80):
python3 $S/blocking_recall.py --candidates output/cands_val20k_union.tsv \
  --ground-truth subset/val20k_gt.tsv --s1 subset/val20k_s1.tsv \
  --s2 dataset/train/train_source2.tsv --s3 dataset/train/train_source3.tsv \
  --ks 25,50,80

# 5) Train (model + tuned tau):
python3 $S/train.py --s1 subset/sub220k_s1.tsv \
  --s2 dataset/train/train_source2.tsv --s3 dataset/train/train_source3.tsv \
  --train-gt subset/train200k_gt.tsv --val-gt subset/val20k_gt.tsv \
  --candidates output/cands_train200k_union.tsv \
  --cand-name output/cands_train200k_name.tsv \
  --cand-addr output/cands_train200k_addr.tsv \
  --feature-set v2 --out-dir output/full_model --seed 42

# 6) Per-country inference (S1 must be sliced per country: predict.py streams
#    S1 + candidates in lockstep, so rows must align 1:1):
python3 $S/predict.py --s1 output/test_s1_India.tsv \
  --s2 dataset/test/test_source2.tsv --s3 dataset/test/test_source3.tsv \
  --candidates output/test_union_India.tsv \
  --cand-name output/test_India_name.tsv --cand-addr output/test_India_addr.tsv \
  --model output/full_model/model.txt --threshold-json output/full_model/threshold.json \
  --feature-set v2 --country India --workers 2 \
  --out-matches output/matching_India.tsv --out-cands output/candpairs_India.tsv
# ... repeat per country, concat shards, validate (notebook 02b cell 6 does all).
```

## Sample run (20k dev sample, verifiable in this repo)

```bash
S=code/business_entity_resolution/src
CFG="--k 40 --min-ngram 3 --max-ngram 4 --min-df 2 --max-df 0.05"
python3 $S/blocking.py --s1 sample/sample_s1.csv --s2 sample/sample_s2.csv \
  --s3 sample/sample_s3.csv --field name --out output/D_name.tsv $CFG
python3 $S/blocking.py --s1 sample/sample_s1.csv --s2 sample/sample_s2.csv \
  --s3 sample/sample_s3.csv --field address --out output/D_addr.tsv $CFG
python3 $S/union_candidates.py --a output/D_name.tsv --b output/D_addr.tsv \
  --k-per-file 40 --out output/D_union.tsv
python3 $S/blocking_recall.py --candidates output/D_union.tsv \
  --ground-truth sample/sample_gt.csv --s1 sample/sample_s1.csv \
  --s2 sample/sample_s2.csv --s3 sample/sample_s3.csv --ks 25,50,80
# expect: recall 0.9911 @80, avg 77.2 candidates/S1
python3 $S/train.py --s1 sample/sample_s1.csv --s2 sample/sample_s2.csv \
  --s3 sample/sample_s3.csv --train-gt sample/split/train_gt.tsv \
  --val-gt sample/split/val_gt.tsv --candidates output/D_union.tsv \
  --cand-name output/D_name.tsv --cand-addr output/D_addr.tsv \
  --feature-set v2 --out-dir output/sample_model --seed 42
# expect: val F0.5 ~0.978
TAU=$(python3 -c "import json; print(json.load(open('output/sample_model/threshold.json'))['tau'])")
python3 $S/predict.py --s1 sample/sample_s1.csv --s2 sample/sample_s2.csv \
  --s3 sample/sample_s3.csv --candidates output/D_union.tsv \
  --cand-name output/D_name.tsv --cand-addr output/D_addr.tsv \
  --model output/sample_model/model.txt --tau $TAU --feature-set v2 \
  --workers 2 --out-matches output/sample_matching.tsv --out-cands output/sample_cands.tsv
python3 $S/metrics.py --truth sample/sample_gt.csv --pred output/sample_matching.tsv
# expect: full-sample fit F0.5 ~0.991
```

## Validate any submission before uploading

```bash
python3 utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test
```

## Metric check

README example: truth `[S2-00047, S3-00812]`, pred `[S2-00047, S2-00193, S3-00812]`
gives P=2/3, R=1.0, F_0.5=0.714. Verify:

```bash
printf 'source1_entity_id\tmatched_entity_ids\nS1-00001\tS2-00047,S3-00812\n' > /tmp/truth.tsv
printf 'source1_entity_id\tmatched_entity_ids\nS1-00001\tS2-00047,S2-00193,S3-00812\n' > /tmp/pred.tsv
python3 code/business_entity_resolution/src/metrics.py --truth /tmp/truth.tsv --pred /tmp/pred.tsv
# expected: Macro F_0.5 ≈ 0.714286
```
