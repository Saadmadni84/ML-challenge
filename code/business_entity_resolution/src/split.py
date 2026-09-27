#!/usr/bin/env python3
"""Entity-disjoint stratified train/val split over S1 ids (Exp3+).

Splits GROUND TRUTH rows (one per S1) into train/val, stratified by
(country, match-count bucket) so val mirrors the population (singletons
included). S2/S3 targets are NOT split: at inference all test targets are
visible, so train and val share the full target pool. Only the S1 queries
are disjoint — that is the correct leakage boundary for this problem.

Outputs: train_gt.tsv, val_gt.tsv (both in GT format).

Usage:
    python3 split.py --s1 sample/sample_s1.csv --gt sample/sample_gt.csv \\
        --out-dir sample/split --val-frac 0.2 --seed 42
"""

import argparse
import csv
import os
import random
from collections import Counter, defaultdict


def bucket(k):
    return str(k) if k <= 5 else "6+"


def main():
    ap = argparse.ArgumentParser(description="Stratified S1 train/val split.")
    ap.add_argument("--s1", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    country = {}
    with open(args.s1, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            country[row["entity_id"]] = (row.get("country") or "").strip()

    rows = {}
    with open(args.gt, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            rows[row["source1_entity_id"]] = (row.get("matched_entity_ids") or "").strip()

    strata = defaultdict(list)
    for s1id, cell in rows.items():
        k = 0 if not cell else cell.count(",") + 1
        strata[(country.get(s1id, "?"), bucket(k))].append(s1id)

    rng = random.Random(args.seed)
    train, val = {}, {}
    print(f"{'country':>8} {'k':>3} {'n':>7} {'train':>7} {'val':>6}")
    for key in sorted(strata):
        ids = strata[key]
        rng.shuffle(ids)
        n_val = max(1, int(round(len(ids) * args.val_frac))) if len(ids) > 1 else 0
        for i in ids[:n_val]:
            val[i] = rows[i]
        for i in ids[n_val:]:
            train[i] = rows[i]
        print(f"{key[0]:>8} {str(key[1]):>3} {len(ids):>7} "
              f"{len(ids) - n_val:>7} {n_val:>6}")

    for name, subset in (("train_gt.tsv", train), ("val_gt.tsv", val)):
        with open(os.path.join(args.out_dir, name), "w", encoding="utf-8") as f:
            f.write("source1_entity_id\tmatched_entity_ids\n")
            for s1id in sorted(subset):
                f.write(f"{s1id}\t{subset[s1id]}\n")

    def dist(subset):
        c = Counter(0 if not v else v.count(",") + 1 for v in subset.values())
        return dict(sorted(c.items()))

    print(f"\nTrain S1: {len(train):,}  Val S1: {len(val):,}")
    print(f"Train match-count dist: {dist(train)}")
    print(f"Val   match-count dist: {dist(val)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
