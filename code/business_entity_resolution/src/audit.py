#!/usr/bin/env python3
"""Streaming, low-RAM data auditor for the Entity Resolution TSVs.

Reads each file ONCE, line by line (never loads it fully into memory), so it
runs even on small machines against the multi-GB source files. Stdlib only.

Examples:
    # Audit one source file (+ show 3 random example rows):
    python3 audit.py --source dataset/train/train_source1.tsv --examples 3

    # Audit the ground truth match distribution:
    python3 audit.py --ground-truth dataset/train/train_ground_truth.tsv

    # Quick row counts only (fastest):
    python3 audit.py --source dataset/train/train_source2.tsv --counts-only
"""

import argparse
import csv
import random
import sys
from collections import Counter


def _is_non_ascii(s):
    return any(ord(c) > 127 for c in s)


def audit_source(path, n_examples=3, seed=42, counts_only=False):
    rng = random.Random(seed)
    n = 0
    countries = Counter()
    name_min = name_max = None
    name_sum = 0
    non_ascii_names = 0
    digit_names = 0
    empty_addr = 0
    addr_min = addr_max = None
    addr_sum = 0
    len_reservoir = []   # for approximate quantiles of name length
    row_reservoir = []   # random example rows
    header = None

    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        header = reader.fieldnames
        for i, row in enumerate(reader):
            try:
                name = row.get("business_name") or ""
                addr = row.get("business_address") or ""
                ctry = (row.get("country") or "").strip()
            except AttributeError:
                continue
            n += 1
            countries[ctry] += 1
            L = len(name)
            name_sum += L
            name_min = L if name_min is None or L < name_min else name_min
            name_max = L if name_max is None or L > name_max else name_max
            if _is_non_ascii(name):
                non_ascii_names += 1
            if any(c.isdigit() for c in name):
                digit_names += 1
            if addr == "":
                empty_addr += 1
            else:
                la = len(addr)
                addr_sum += la
                addr_min = la if addr_min is None or la < addr_min else addr_min
                addr_max = la if addr_max is None or la > addr_max else addr_max
            # reservoirs (single pass, bounded memory)
            if len(len_reservoir) < 20000:
                len_reservoir.append(L)
            else:
                j = rng.randrange(i + 1)
                if j < 20000:
                    len_reservoir[j] = L
            if len(row_reservoir) < n_examples:
                row_reservoir.append((row.get("entity_id"), name, addr, ctry))
            else:
                j = rng.randrange(i + 1)
                if j < n_examples:
                    row_reservoir[j] = (row.get("entity_id"), name, addr, ctry)

    def quantile(xs, q):
        if not xs:
            return None
        xs = sorted(xs)
        return xs[min(len(xs) - 1, int(q * len(xs)))]

    print(f"FILE: {path}")
    print(f"  header : {header}")
    print(f"  rows   : {n:,}")
    if counts_only:
        return
    print(f"  country: {dict(countries)}")
    if n:
        print(f"  name len: min={name_min} median~{quantile(len_reservoir, 0.5)} "
              f"mean={name_sum / n:.1f} p90~{quantile(len_reservoir, 0.9)} max={name_max}")
        print(f"  name non-ASCII: {non_ascii_names:,} ({100 * non_ascii_names / n:.2f}%) | "
              f"with digits: {digit_names:,} ({100 * digit_names / n:.2f}%)")
        print(f"  empty address : {empty_addr:,} ({100 * empty_addr / n:.2f}%)", end="")
        if addr_min is not None:
            n_addr = n - empty_addr
            print(f" | non-empty addr len: min={addr_min} mean={addr_sum / n_addr:.1f} max={addr_max}")
        else:
            print()
    for k, (eid, nm, ad, ct) in enumerate(row_reservoir, 1):
        print(f"  ex{k} [{ct}] {eid}: name={nm[:80]!r} addr={ad[:90]!r}")


def audit_ground_truth(path, seed=42):
    n = 0
    match_counts = Counter()
    total_matches = 0
    s2_total = s3_total = 0
    dist_examples = {}
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        print(f"FILE: {path}")
        print(f"  header: {reader.fieldnames}")
        for row in reader:
            cell = (row.get("matched_entity_ids") or "").strip()
            ids = [x.strip() for x in cell.split(",") if x.strip()] if cell else []
            k = len(ids)
            n += 1
            match_counts[k] += 1
            total_matches += k
            for i in ids:
                if i.startswith("S2-"):
                    s2_total += 1
                elif i.startswith("S3-"):
                    s3_total += 1
            if k not in dist_examples:
                dist_examples[k] = (row.get("source1_entity_id"), ids[:4])
    print(f"  S1 entities      : {n:,}")
    print(f"  total matches    : {total_matches:,} "
          f"(S2: {s2_total:,}, S3: {s3_total:,})")
    print(f"  mean matches/S1  : {total_matches / n:.4f}" if n else "")
    print(f"  singletons (0)   : {match_counts.get(0, 0):,} "
          f"({100 * match_counts.get(0, 0) / max(n, 1):.2f}%)")
    print("  match-count distribution (k: count):")
    for k in sorted(match_counts):
        cnt = match_counts[k]
        print(f"    {k:>2}: {cnt:>10,} ({100 * cnt / n:5.2f}%)")
    for k in sorted(dist_examples)[:6]:
        print(f"  ex k={k}: {dist_examples[k][0]} -> {dist_examples[k][1]}")


def main():
    ap = argparse.ArgumentParser(description="Streaming auditor for ER challenge TSVs.")
    ap.add_argument("--source", help="Path to a *_source{1,2,3}.tsv file.")
    ap.add_argument("--ground-truth", help="Path to train_ground_truth.tsv.")
    ap.add_argument("--examples", type=int, default=3, help="Random example rows to show.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--counts-only", action="store_true", help="Only verify header+row count.")
    args = ap.parse_args()
    if not args.source and not args.ground_truth:
        ap.error("pass --source and/or --ground-truth")
    try:
        if args.source:
            audit_source(args.source, args.examples, args.seed, args.counts_only)
        if args.ground_truth:
            audit_ground_truth(args.ground_truth, args.seed)
    except (OSError, UnicodeDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
