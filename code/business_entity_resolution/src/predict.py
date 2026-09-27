#!/usr/bin/env python3
"""Inference: score candidates -> matching_results.tsv + candidate_pairs.tsv.

Streams S1 + union + per-channel candidate files IN LOCKSTEP (all share S1
row order by construction), featurizes pairs, scores with a trained
LightGBM model, applies the fixed val-tuned tau, and writes BOTH submission
files in one pass:
  - candidate_pairs.tsv: stripped union (no scores) — written while streaming
  - matching_results.tsv: pairs with proba >= tau — written from workers

Memory design: only the TARGET lookup dict (per country with --country) is
held in RAM (~1-2GB for the biggest test shard); S1/candidates stream in
chunks. Workers are forked AFTER loading targets, so the big read-only dict
is shared copy-on-write (Linux). Each worker loads its OWN LightGBM Booster
via initializer (OpenMP-backed objects created pre-fork can deadlock).

Usage (per-country test shard):
    python3 predict.py --s1 dataset/test/test_source1.tsv \\
        --s2 dataset/test/test_source2.tsv --s3 dataset/test/test_source3.tsv \\
        --candidates output/cand_union_India.tsv \\
        --cand-name output/cand_name_India.tsv --cand-addr output/cand_addr_India.tsv \\
        --model output/model.txt --threshold-json output/threshold.json \\
        --feature-set v2 --country India --workers 2 \\
        --out-matches output/matching_India.tsv --out-cands output/cands_India.tsv
"""

import argparse
import csv
import json
import os
import sys
import time

# Cap thread pools: with N forked workers on N CPUs, one thread each avoids
# oversubscription (featurization is single-threaded; LightGBM predict runs
# on big batches). Must be set before importing lightgbm.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import (FEATURE_NAMES, FEATURE_NAMES_V2, pair_features,  # noqa: E402
                      pair_features_v2)

import lightgbm as lgb

_W = {}  # worker globals: targets (fork-shared), model (per-worker), tau, feats


def _init_worker(model_path):
    # Load LightGBM INSIDE the forked worker (see module docstring).
    _W["model"] = lgb.Booster(model_file=model_path)


def parse_scored(cell):
    """'id:score,...' -> ([ids], {id: (score, 1-based rank)})."""
    ids, scored = [], {}
    if cell:
        for rank, x in enumerate(cell.split(","), start=1):
            if not x.strip():
                continue
            cid, _, score = x.partition(":")
            ids.append(cid)
            scored[cid] = (float(score or 0.0), rank)
    return ids, scored


def process_chunk(rows):
    """rows: [(s1id, s1name, s1addr, union_cell, name_cell, addr_cell)].

    Returns [(s1id, [kept tids])] in input order. Featurizes the whole
    chunk, then scores with ONE Booster.predict call: per-S1 predict calls
    are overhead-dominated (thread spin-up per call) and 10-100x slower.
    """
    targets, model, tau = _W["targets"], _W["model"], _W["tau"]
    feat_names, v2 = _W["feat_names"], _W["v2"]
    bounds = []  # (s1id, uids, start, end)
    mat = []
    for s1id, s1n, s1a, union_cell, name_cell, addr_cell in rows:
        uids, _ = parse_scored(union_cell)
        _, sname = parse_scored(name_cell)
        _, saddr = parse_scored(addr_cell)
        start = len(mat)
        for tid in uids:
            tn, ta = targets[tid]
            if v2:
                f = pair_features_v2(s1n, s1a, tn, ta,
                                     name_hit=sname.get(tid), addr_hit=saddr.get(tid),
                                     n_cands=len(uids))
            else:
                f = pair_features(s1n, s1a, tn, ta)
            mat.append([f[n] for n in feat_names])
        bounds.append((s1id, uids, start, len(mat)))
    proba = (model.predict(np.asarray(mat, dtype=np.float32))
             if mat else np.array([]))
    return [(s1id, [t for t, p in zip(uids, proba[a:b]) if p >= tau])
            for s1id, uids, a, b in bounds]


def stream_quads(s1_path, union_path, name_path, addr_path, country):
    """Yield aligned (s1id, s1name, s1addr, union_cell, name_cell, addr_cell).

    All four files must share S1 row order (guaranteed when every step
    writes in S1-file order). Misalignment raises immediately.
    """
    f1 = open(s1_path, encoding="utf-8", newline="")
    fu = open(union_path, encoding="utf-8", newline="")
    fn = open(name_path, encoding="utf-8", newline="") if name_path else None
    fa = open(addr_path, encoding="utf-8", newline="") if addr_path else None
    r1 = csv.DictReader(f1, delimiter="\t")
    ru = csv.DictReader(fu, delimiter="\t")
    rn = csv.DictReader(fn, delimiter="\t") if fn else None
    ra = csv.DictReader(fa, delimiter="\t") if fa else None
    n = 0
    while True:
        try:
            a = next(r1)
        except StopIteration:
            break
        b = next(ru)
        c = next(rn) if rn else {"source1_entity_id": a["entity_id"],
                                 "candidate_entity_ids": ""}
        d = next(ra) if ra else {"source1_entity_id": a["entity_id"],
                                 "candidate_entity_ids": ""}
        if not (a["entity_id"] == b["source1_entity_id"]
                == c["source1_entity_id"] == d["source1_entity_id"]):
            raise ValueError(f"Row misalignment at S1 row {n}: {a['entity_id']} vs "
                             f"{b['source1_entity_id']}")
        n += 1
        if country and (a.get("country") or "").strip() != country:
            continue
        yield (a["entity_id"], a.get("business_name") or "",
               a.get("business_address") or "",
               (b.get("candidate_entity_ids") or "").strip(),
               (c.get("candidate_entity_ids") or "").strip(),
               (d.get("candidate_entity_ids") or "").strip())
    for f in (f1, fu, fn, fa):
        if f:
            f.close()


def load_targets(s2_path, s3_path, country):
    out = {}
    for path in (s2_path, s3_path):
        with open(path, encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                if country and (row.get("country") or "").strip() != country:
                    continue
                out[row["entity_id"]] = (row.get("business_name") or "",
                                         row.get("business_address") or "")
    return out


def main():
    ap = argparse.ArgumentParser(description="Batched ER inference.")
    ap.add_argument("--s1", required=True)
    ap.add_argument("--s2", required=True)
    ap.add_argument("--s3", required=True)
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--cand-name", default=None)
    ap.add_argument("--cand-addr", default=None)
    ap.add_argument("--model", required=True)
    ap.add_argument("--threshold-json", default=None)
    ap.add_argument("--tau", type=float, default=None)
    ap.add_argument("--feature-set", choices=["v1", "v2"], default="v2")
    ap.add_argument("--country", default=None)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--chunk-s1", type=int, default=5000)
    ap.add_argument("--max-s1", type=int, default=0,
                    help="Debug: process only first N S1s (0 = all).")
    ap.add_argument("--out-matches", required=True)
    ap.add_argument("--out-cands", required=True)
    args = ap.parse_args()

    tau = args.tau
    if tau is None:
        if not args.threshold_json:
            ap.error("need --tau or --threshold-json")
        with open(args.threshold_json) as f:
            tau = json.load(f)["tau"]
    if args.feature_set == "v2" and not (args.cand_name and args.cand_addr):
        ap.error("--feature-set v2 requires --cand-name and --cand-addr")
    for p in (args.out_matches, args.out_cands):
        d = os.path.dirname(os.path.abspath(p))
        os.makedirs(d, exist_ok=True)

    t0 = time.time()
    print("Loading targets...", flush=True)
    _W["targets"] = load_targets(args.s2, args.s3, args.country)
    print(f"  {len(_W['targets']):,} target rows "
          f"({time.time() - t0:.1f}s)", flush=True)
    if args.workers > 1:
        _W["model"] = None  # each worker loads its own (see _init_worker)
    else:
        _W["model"] = lgb.Booster(model_file=args.model)
    _W["tau"] = float(tau)
    _W["v2"] = args.feature_set == "v2"
    _W["feat_names"] = FEATURE_NAMES_V2 if _W["v2"] else FEATURE_NAMES
    print(f"Model + tau={tau} + features={args.feature_set} "
          f"({len(_W['feat_names'])} feats) + workers={args.workers}",
          flush=True)

    def chunk_iter():
        buf = []
        n = 0
        for quad in stream_quads(args.s1, args.candidates, args.cand_name,
                                 args.cand_addr, args.country):
            buf.append(quad)
            n += 1
            if len(buf) >= args.chunk_s1 or (args.max_s1 and n >= args.max_s1):
                yield buf
                buf = []
                if args.max_s1 and n >= args.max_s1:
                    return
        if buf:
            yield buf

    fm = open(args.out_matches, "w", encoding="utf-8", newline="")
    fc = open(args.out_cands, "w", encoding="utf-8", newline="")
    fm.write("source1_entity_id\tmatched_entity_ids\n")
    fc.write("source1_entity_id\tcandidate_entity_ids\n")

    # candidate_pairs.tsv is stripped while streaming (parent side, in order).
    def with_cand_write(it):
        for buf in it:
            for q in buf:
                uids, _ = parse_scored(q[3])
                fc.write(q[0] + "\t" + ",".join(uids) + "\n")
            yield buf

    n_s1 = n_kept = n_empty = 0
    t1 = time.time()

    def handle(res):
        nonlocal n_s1, n_kept, n_empty
        for s1id, kept in res:
            fm.write(s1id + "\t" + ",".join(kept) + "\n")
            n_s1 += 1
            n_kept += len(kept)
            if not kept:
                n_empty += 1

    it = with_cand_write(chunk_iter())
    if args.workers > 1:
        with Pool(args.workers, initializer=_init_worker,
                  initargs=(args.model,)) as pool:
            for res in pool.imap(process_chunk, it, chunksize=1):
                handle(res)
                el = time.time() - t1
                print(f"  {n_s1:,} S1s | kept {n_kept:,} "
                      f"({n_kept / max(n_s1, 1):.2f}/S1, {100 * n_empty / max(n_s1, 1):.1f}% empty) "
                      f"| {el:.0f}s ({n_s1 / max(el, 1):.0f} S1/s)", flush=True)
    else:
        for buf in it:
            handle(process_chunk(buf))
            el = time.time() - t1
            print(f"  {n_s1:,} S1s | kept {n_kept:,} | {el:.0f}s", flush=True)

    fm.close()
    fc.close()
    print(f"Done: {n_s1:,} S1s, {n_kept:,} kept matches "
          f"({n_kept / max(n_s1, 1):.3f}/S1), {100 * n_empty / max(n_s1, 1):.1f}% empty "
          f"in {(time.time() - t0) / 60:.1f} min.", flush=True)
    print(f"  matches -> {args.out_matches}\n  candidates -> {args.out_cands}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
