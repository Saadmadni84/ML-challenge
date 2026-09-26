# Business Entity Resolution — runnable pipeline

Self-contained pipeline for ML Challenge 2026. Anyone should be able to
regenerate `output/matching_results.tsv` and `output/candidate_pairs.tsv`
from the training/test data using only this folder + the dataset.

## Layout

```
code/business_entity_resolution/
├── src/
│   ├── metrics.py    # local replica of the leaderboard macro-F0.5 scorer
│   ├── inspect.py    # streaming, low-RAM data auditor (stdlib only)
│   ├── ...           # (blocking / features / train / predict added step by step)
├── README.md         # this file (run instructions)
└── requirements.txt  # dependencies (pinned before final submission)
```

## Environment

```bash
pip install -r code/business_entity_resolution/requirements.txt
```

## Run steps (grows as we build the solution)

```bash
# 0. Sanity-check the metric on the README toy example (see "Metric check")
# 1. Audit the data (streaming; safe on small machines):
python3 code/business_entity_resolution/src/inspect.py \
    --source dataset/train/train_source1.tsv --examples 3
python3 code/business_entity_resolution/src/inspect.py \
    --ground-truth dataset/train/train_ground_truth.tsv

# 2. Validate any submission before uploading:
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
