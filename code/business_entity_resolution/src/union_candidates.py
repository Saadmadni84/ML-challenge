#!/usr/bin/env python3
"""Merge two internal candidate files into one union (Exp2c).

Union = name-channel top-K plus address-channel top-K, deduplicated by id
(keeping the max score), sorted by score desc. Truncation to --k-per-file
happens BEFORE the union, so the result holds <= 2 * k-per-file candidates
per S1 (fair budget comparison against single-channel top-K).

Caveat: scores from two different TF-IDF spaces are not strictly
comparable, so recall@K curves on the union are approximate for K < Kmax.
Quote recall@Kmax (the full union) as the recall ceiling.

Usage:
    python3 union_candidates.py --a output/cand.tsv --b output/cand_addr.tsv \\
        --k-per-file 50 --out output/cand_union.tsv
"""

import argparse
import csv
import os


def load(path, k_per_file):
    out = {}
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            cell = (row.get("candidate_entity_ids") or "").strip()
            pairs = []
            if cell:
                for x in cell.split(","):
                    if not x.strip():
                        continue
                    cid, _, score = x.partition(":")
                    pairs.append((cid, float(score or 0.0)))
            if k_per_file > 0:
                pairs = pairs[:k_per_file]
            out[row["source1_entity_id"]] = pairs
    return out


def main():
    ap = argparse.ArgumentParser(description="Union two candidate files.")
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--k-per-file", type=int, default=50)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)

    A, B = load(args.a, args.k_per_file), load(args.b, args.k_per_file)
    n_total = 0
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for qid in A.keys() | B.keys():
            merged = {}
            for cid, s in A.get(qid, []) + B.get(qid, []):
                if cid not in merged or s > merged[cid]:
                    merged[cid] = s
            ranked = sorted(merged.items(), key=lambda kv: -kv[1])
            n_total += len(ranked)
            f.write(qid + "\t" + ",".join(f"{i}:{s:.4f}" for i, s in ranked) + "\n")
    n_q = len(A.keys() | B.keys())
    print(f"Union of {len(A):,} + {len(B):,} rows (k-per-file={args.k_per_file}) -> "
          f"{args.out}; avg {n_total / n_q:.1f} candidates/S1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
