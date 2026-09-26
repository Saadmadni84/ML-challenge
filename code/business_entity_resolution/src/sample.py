#!/usr/bin/env python3
"""Build a small stratified dev sample from the full training TSVs.

Why: full training data is ~1.3 GB / 12.5M rows. We iterate fast on a small
sample (blocking rules, features, baseline) and scale to full data only for
real scores and the final submission.

What it does (streaming, low RAM — safe on Colab free tier):
  1. Stratified sample of N Source-1 ids (proportional to country shares).
  2. Keep their ground-truth rows; collect all matched S2/S3 ids.
  3. From Source-2/3 keep every matched row + a deterministic stride sample
     of distractors (unmatched rows), mimicking the real match/distractor mix.
  4. Write sample_s1.tsv, sample_s2.tsv, sample_s3.tsv, sample_gt.tsv.

Usage (from repo root):
    python3 code/business_entity_resolution/src/sample.py \
        --train-dir dataset/train --out-dir sample \
        --n-s1 20000 --distractors-per-s1 5 --seed 42
"""

import argparse
import csv
import os
import random
import sys
from collections import Counter


def count_data_rows(path):
    """Fast line count (buffered binary read). Minus 1 for the header."""
    newlines = 0
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            newlines += chunk.count(b"\n")
    return max(0, newlines - 1)


def sample_s1_ids(s1_path, n, seed):
    """Pass 1: stratified sample of S1 ids, proportional to country shares."""
    ids_by_country = {}
    with open(s1_path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            ids_by_country.setdefault((row.get("country") or "").strip(), []).append(
                row.get("entity_id")
            )
    total = sum(len(v) for v in ids_by_country.values())
    if n > total:
        raise ValueError(f"--n-s1={n} exceeds total S1 rows={total}")
    rng = random.Random(seed)
    keep, info = set(), {}
    assigned = 0
    countries = sorted(ids_by_country)
    for i, c in enumerate(countries):
        ids = ids_by_country[c]
        # proportional allocation; last country takes the remainder
        k = int(round(n * len(ids) / total)) if i < len(countries) - 1 else n - assigned
        k = min(k, len(ids))
        keep.update(rng.sample(ids, k))
        assigned += k
        info[c] = {"population": len(ids), "sampled": k}
    return keep, info


def extract_s1_rows(s1_path, keep_ids, out_path):
    """Pass 2 over S1: write only the sampled rows."""
    n = 0
    with open(s1_path, encoding="utf-8", newline="") as fin, open(
        out_path, "w", encoding="utf-8", newline=""
    ) as fout:
        reader = csv.DictReader(fin, delimiter="\t")
        writer = csv.DictWriter(fout, fieldnames=reader.fieldnames, delimiter="\t")
        writer.writeheader()
        for row in reader:
            if row.get("entity_id") in keep_ids:
                writer.writerow(row)
                n += 1
    return n


def extract_gt(gt_path, keep_ids, out_path):
    """Keep GT rows for sampled S1s; collect every matched S2/S3 id."""
    matched = set()
    match_counts = Counter()
    n = 0
    with open(gt_path, encoding="utf-8", newline="") as fin, open(
        out_path, "w", encoding="utf-8", newline=""
    ) as fout:
        reader = csv.DictReader(fin, delimiter="\t")
        writer = csv.DictWriter(fout, fieldnames=reader.fieldnames, delimiter="\t")
        writer.writeheader()
        for row in reader:
            if row.get("source1_entity_id") not in keep_ids:
                continue
            writer.writerow(row)
            n += 1
            cell = (row.get("matched_entity_ids") or "").strip()
            ids = [x.strip() for x in cell.split(",") if x.strip()] if cell else []
            match_counts[len(ids)] += 1
            matched.update(ids)
    return matched, n, match_counts


def extract_source(src_path, keep_ids, n_distractors, seed, out_path):
    """Keep matched rows + deterministic stride sample of distractors."""
    total = count_data_rows(src_path)
    stride = max(1, total // max(1, n_distractors))
    n_kept_match = n_kept_dist = 0
    with open(src_path, encoding="utf-8", newline="") as fin, open(
        out_path, "w", encoding="utf-8", newline=""
    ) as fout:
        reader = csv.DictReader(fin, delimiter="\t")
        writer = csv.DictWriter(fout, fieldnames=reader.fieldnames, delimiter="\t")
        writer.writeheader()
        for idx, row in enumerate(reader):
            eid = row.get("entity_id")
            if eid in keep_ids:
                writer.writerow(row)
                n_kept_match += 1
            elif n_kept_dist < n_distractors and (idx % stride == 0):
                writer.writerow(row)
                n_kept_dist += 1
    return n_kept_match, n_kept_dist, total


def load_ids(path):
    with open(path, encoding="utf-8", newline="") as f:
        return {row.get("entity_id") for row in csv.DictReader(f, delimiter="\t")}


def main():
    ap = argparse.ArgumentParser(description="Build a stratified dev sample (streaming).")
    ap.add_argument("--train-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-s1", type=int, default=20000)
    ap.add_argument("--distractors-per-s1", type=int, default=5,
                    help="Total distractors across S2+S2 = n_s1 * this (split evenly).")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    s1_path = os.path.join(args.train_dir, "train_source1.tsv")
    s2_path = os.path.join(args.train_dir, "train_source2.tsv")
    s3_path = os.path.join(args.train_dir, "train_source3.tsv")
    gt_path = os.path.join(args.train_dir, "train_ground_truth.tsv")
    for p in (s1_path, s2_path, s3_path, gt_path):
        if not os.path.isfile(p):
            print(f"ERROR: not found: {p}", file=sys.stderr)
            return 1
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Sampling {args.n_s1} S1 entities (seed={args.seed}) ...", flush=True)
    keep_s1, info = sample_s1_ids(s1_path, args.n_s1, args.seed)
    for c, d in info.items():
        print(f"  {c or '?'}: sampled {d['sampled']:,} of {d['population']:,}")

    n1 = extract_s1_rows(s1_path, keep_s1, os.path.join(args.out_dir, "sample_s1.tsv"))
    matched, n_gt, mc = extract_gt(gt_path, keep_s1, os.path.join(args.out_dir, "sample_gt.tsv"))
    print(f"Wrote {n1:,} S1 rows, {n_gt:,} GT rows, {len(matched):,} unique matched S2/S3 ids.")
    tot_m = sum(k * v for k, v in mc.items())
    print(f"Sample match-count distribution (mean {tot_m / max(n_gt, 1):.3f}, "
          f"singletons {mc.get(0, 0):,} = {100 * mc.get(0, 0) / max(n_gt, 1):.2f}%):")
    for k in sorted(mc):
        print(f"  {k:>2}: {mc[k]:>7,} ({100 * mc[k] / n_gt:5.2f}%)")

    per_file = (args.n_s1 * args.distractors_per_s1) // 2
    for src, name in ((s2_path, "sample_s2.tsv"), (s3_path, "sample_s3.tsv")):
        m, d, t = extract_source(src, matched, per_file, args.seed,
                                 os.path.join(args.out_dir, name))
        print(f"{name}: kept {m:,} matched + {d:,} distractors (from {t:,} rows)")

    # Verify: every matched id of the sample must exist in the sample targets.
    have = load_ids(os.path.join(args.out_dir, "sample_s2.tsv")) | load_ids(
        os.path.join(args.out_dir, "sample_s3.tsv"))
    missing = matched - have
    if missing:
        print(f"ERROR: {len(missing)} matched ids missing from sample targets!", file=sys.stderr)
        return 1
    print("VERIFY OK: all matched ids present in the sample.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
