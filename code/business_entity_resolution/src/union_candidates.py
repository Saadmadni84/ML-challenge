#!/usr/bin/env python3
"""Merge two internal candidate files into one union (Exp2c).

Union = name-channel top-K plus address-channel top-K, deduplicated by id
(keeping the max score), sorted by score desc. Truncation to --k-per-file
happens BEFORE the union, so the result holds <= 2 * k-per-file candidates
per S1 (fair budget comparison against single-channel top-K).

Caveat: scores from two different TF-IDF spaces are not strictly
comparable, so recall@K curves on the union are approximate for K < Kmax.
Quote recall@Kmax (the full union) as the recall ceiling.

--floor drops merged pairs whose max channel score is below the floor
(these are retrieval junk: weak in BOTH channels). Apply the SAME floor to
train/val/test unions so the pair distribution matches everywhere.

Row order follows the inputs, which must share S1 row order (holds by
construction when both come from blocking.py on the same S1 file). The
merge STREAMS both files in lockstep (constant RAM at any scale) and
errors immediately on any misalignment — that signals a pipeline bug.
Order matters: predict.py streams S1 + union + channels in lockstep.

Usage:
    python3 union_candidates.py --a output/cand.tsv --b output/cand_addr.tsv \\
        --k-per-file 50 --floor 0.05 --out output/cand_union.tsv
"""

import argparse
import csv
import os


def parse_cell(cell, k_per_file):
    """'id:score,...' -> [(id, score)] truncated to k_per_file (file order)."""
    cell = (cell or "").strip()
    pairs = []
    if cell:
        for x in cell.split(","):
            if not x.strip():
                continue
            cid, _, score = x.partition(":")
            pairs.append((cid, float(score or 0.0)))
    if k_per_file > 0:
        pairs = pairs[:k_per_file]
    return pairs


def main():
    ap = argparse.ArgumentParser(description="Union two candidate files.")
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--k-per-file", type=int, default=50)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)

    n_rows = n_total = 0
    with open(args.a, encoding="utf-8", newline="") as fa, \
            open(args.b, encoding="utf-8", newline="") as fb, \
            open(args.out, "w", encoding="utf-8", newline="") as fo:
        ra = csv.DictReader(fa, delimiter="\t")
        rb = csv.DictReader(fb, delimiter="\t")
        fo.write("source1_entity_id\tcandidate_entity_ids\n")
        for i, (row_a, row_b) in enumerate(zip(ra, rb)):
            qa, qb = row_a["source1_entity_id"], row_b["source1_entity_id"]
            if qa != qb:
                raise ValueError(
                    f"S1 row misalignment at row {i}: {args.a} has {qa!r} "
                    f"but {args.b} has {qb!r}. Both files must come from "
                    f"blocking.py on the same S1 file.")
            merged = {}
            for cid, s in (parse_cell(row_a.get("candidate_entity_ids"), args.k_per_file)
                           + parse_cell(row_b.get("candidate_entity_ids"), args.k_per_file)):
                if cid not in merged or s > merged[cid]:
                    merged[cid] = s
            ranked = sorted(merged.items(), key=lambda kv: -kv[1])
            n_total += len(ranked)
            n_rows += 1
            fo.write(qa + "\t" + ",".join(f"{cid}:{s:.4f}" for cid, s in ranked) + "\n")
        # zip() stops at the shorter file: a row-count mismatch is also a bug.
        extra_a = sum(1 for _ in ra)
        extra_b = sum(1 for _ in rb)
        if extra_a or extra_b:
            raise ValueError(
                f"Row-count mismatch: {args.a} has {extra_a} extra row(s), "
                f"{args.b} has {extra_b} extra row(s) after {n_rows} shared rows.")
    print(f"Union of {n_rows:,} + {n_rows:,} rows (k-per-file={args.k_per_file}) -> "
          f"{args.out}; avg {n_total / max(n_rows, 1):.1f} candidates/S1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
