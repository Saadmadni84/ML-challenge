#!/usr/bin/env python3
"""Exp1 baseline: normalized exact-name match within the same country.

Rule: each S1 entity matches EVERY S2/S3 record with the same `country` and
the same normalized `business_name`. No ML, no threshold, no address use.

Hypothesis: a surprising share of matches are near-verbatim copies, so this
cheap rule should beat the Exp0 (all-empty) floor and tell us how much of
the problem is 'just normalization'.

Usage (from repo root):
    python3 code/business_entity_resolution/src/baseline_exact.py \\
        --s1 sample/sample_s1.tsv --s2 sample/sample_s2.tsv \\
        --s3 sample/sample_s3.tsv --out sample/exp1_pred.tsv
"""

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from normalize import normalize_name  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Exp1: exact normalized-name match.")
    ap.add_argument("--s1", required=True)
    ap.add_argument("--s2", required=True)
    ap.add_argument("--s3", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)

    # Index every S2/S3 id by (country, normalized name). Empty normalized
    # names are skipped: matching on '' would merge unrelated records.
    index = {}
    n_indexed = n_skipped = 0
    for path in (args.s2, args.s3):
        with open(path, encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                key = ((row.get("country") or "").strip(),
                       normalize_name(row.get("business_name")))
                if not key[1]:
                    n_skipped += 1
                    continue
                index.setdefault(key, []).append(row.get("entity_id"))
                n_indexed += 1
    print(f"Indexed {n_indexed:,} S2/S3 rows ({n_skipped:,} skipped: empty after norm).")

    # Lookup each S1.
    n_s1 = n_with_pred = 0
    with open(args.s1, encoding="utf-8", newline="") as fin, open(
        args.out, "w", encoding="utf-8", newline=""
    ) as fout:
        reader = csv.DictReader(fin, delimiter="\t")
        fout.write("source1_entity_id\tmatched_entity_ids\n")
        for row in reader:
            n_s1 += 1
            key = ((row.get("country") or "").strip(),
                   normalize_name(row.get("business_name")))
            ids = index.get(key, []) if key[1] else []
            if ids:
                n_with_pred += 1
            fout.write(f"{row.get('entity_id')}\t{','.join(ids)}\n")
    print(f"Wrote {n_s1:,} predictions ({n_with_pred:,} non-empty) -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
