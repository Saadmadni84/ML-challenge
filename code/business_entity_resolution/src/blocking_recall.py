#!/usr/bin/env python3
"""Recall-ceiling report for a candidates file (Exp2 evaluation).

recall@K = fraction of ALL true matches present in the top-K candidates
(micro-averaged over match pairs). This is the score no downstream matcher
can ever exceed: matches lost here are gone forever.

Also reports: per-country recall, latin-only recall (excludes true matches
whose target name is non-Latin, which name-based retrieval cannot reach),
% of S1s with FULL recall, and S1s with zero candidates.

Usage:
    python3 blocking_recall.py --candidates sample/cand.tsv \\
        --ground-truth sample/sample_gt.csv --s1 sample/sample_s1.csv \\
        --s2 sample/sample_s2.csv --s3 sample/sample_s3.csv \\
        --ks 1,5,10,25,50,100
"""

import argparse
import csv
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from normalize import normalize_name  # noqa: E402


def load_gt(path):
    gt = {}
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            cell = (row.get("matched_entity_ids") or "").strip()
            gt[row["source1_entity_id"]] = (
                {x.strip() for x in cell.split(",") if x.strip()} if cell else set()
            )
    return gt


def load_s1_country(path):
    out = {}
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            out[row["entity_id"]] = (row.get("country") or "").strip()
    return out


def load_target_is_latin(*paths):
    out = {}
    for path in paths:
        with open(path, encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                out[row["entity_id"]] = normalize_name(row.get("business_name")) != ""
    return out


def load_candidates(path):
    """s1 -> [cand ids in score order]."""
    out = {}
    with open(path, encoding="utf-8", newline="") as f:
        rdr = csv.DictReader(f, delimiter="\t")
        for row in rdr:
            cell = (row.get("candidate_entity_ids") or "").strip()
            ids = [x.split(":")[0] for x in cell.split(",") if x.strip()] if cell else []
            out[row["source1_entity_id"]] = ids
    return out


def main():
    ap = argparse.ArgumentParser(description="Blocking recall-ceiling report.")
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--ground-truth", required=True)
    ap.add_argument("--s1", required=True)
    ap.add_argument("--s2", required=True)
    ap.add_argument("--s3", required=True)
    ap.add_argument("--ks", default="1,5,10,25,50,100")
    args = ap.parse_args()
    ks = sorted({int(x) for x in args.ks.split(",") if x.strip()})

    gt = load_gt(args.ground_truth)
    s1c = load_s1_country(args.s1)
    is_latin = load_target_is_latin(args.s2, args.s3)
    cands = load_candidates(args.candidates)

    total = sum(len(t) for t in gt.values())
    total_latin = sum(1 for t in gt.values() for m in t if is_latin.get(m, False))
    print(f"S1 entities: {len(gt):,} | true matches: {total:,} "
          f"(latin-target: {total_latin:,}, {100 * total_latin / total:.1f}%)")
    print(f"S1s with zero candidates: "
          f"{sum(1 for k in gt if not cands.get(k)):,}")

    print(f"\n{'K':>5} {'recall':>8} {'latin-only':>10} " +
          " ".join(f"{c + '':>10}" for c in sorted({s1c.get(k, '?') for k in gt})))
    by_ctry = defaultdict(lambda: [0, 0])  # filled per K below
    for k in ks:
        hit = hit_latin = 0
        by_ctry = defaultdict(lambda: [0, 0])
        for s1id, t in gt.items():
            top = set(cands.get(s1id, [])[:k])
            c = s1c.get(s1id, "?")
            for m in t:
                lat = is_latin.get(m, False)
                h = m in top
                hit += h
                by_ctry[c][h] += 1
                if lat:
                    hit_latin += h
        row = f"{k:>5} {hit / total:>8.4f} {hit_latin / total_latin:>10.4f}"
        for c in sorted(by_ctry):
            h0, h1 = by_ctry[c][0], by_ctry[c][1]
            row += f" {(h1 / (h0 + h1)):>10.4f}"
        print(row)

    kmax = max(ks)
    full = sum(1 for s1id, t in gt.items()
               if t and t <= set(cands.get(s1id, [])[:kmax]))
    n_with_truth = sum(1 for t in gt.values() if t)
    print(f"\nS1s with FULL recall @{kmax}: {full:,}/{n_with_truth:,} "
          f"({100 * full / n_with_truth:.1f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
