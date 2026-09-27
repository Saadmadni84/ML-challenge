#!/usr/bin/env python3
"""TF-IDF character-ngram blocking (candidate generation), scale-ready.

For each S1 entity, retrieve the top-K most similar S2/S3 records **within
the same country** by TF-IDF cosine similarity over character n-grams
(--field name or address).

Scale design (10M targets):
  - Backend `topn` (sparse_dot_topn, C++/OpenMP): exact sparse top-K cosine
    with native score-floor pruning. Backend `loop` (sklearn matmul +
    row-wise argpartition) is the pure-python fallback. `--backend auto`
    tries topn first, falls back to loop with a warning.
  - `--min-df/--max-df` bound the index: max_df drops ultra-common n-grams
    ('ing', 'ter') whose posting lists would explode memory/time.
  - `--countries` + `{country}` in --out shard the work (one file per
    country) and `--resume` skips finished shards after a disconnect.
  - Queries run in chunks with progress + ETA; only the per-country target
    matrix is held in RAM at once.

Output: INTERNAL scored candidates (one row per S1 in the shard):
    source1_entity_id <TAB> cand_id:score,cand_id:score,...
Single-file mode (no {country} in --out) requires --countries all and keeps
all S1 rows (backward compatible with Exp2).

Usage (sample):
    python3 blocking.py --s1 sample/sample_s1.csv --s2 sample/sample_s2.csv \\
        --s3 sample/sample_s3.csv --out output/cand.tsv --k 100
Usage (full test, per-country shards, resumable):
    python3 blocking.py --s1 dataset/test/test_source1.tsv ... \\
        --out output/cand_test_{country}.tsv --k 50 --min-df 2 --max-df 0.05 \\
        --countries India --resume
"""

import argparse
import csv
import os
import sys
import time
from collections import defaultdict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from normalize import normalize_name  # noqa: E402

try:
    from sparse_dot_topn import sp_matmul_topn
    _HAVE_TOPN = True
except Exception as _TOPN_ERR:  # noqa: F841
    _HAVE_TOPN = False


def load_records(path, keep_countries=None):
    """id -> (normalized_name, normalized_address, country).

    keep_countries: when given, rows from other countries are skipped
    during load (big RAM win for per-country runs at full scale).
    """
    recs = {}
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            country = (row.get("country") or "").strip()
            if keep_countries is not None and country not in keep_countries:
                continue
            recs[row["entity_id"]] = (
                normalize_name(row.get("business_name")),
                normalize_name(row.get("business_address")),
                country,
            )
    return recs


def build_index(doc_texts, min_df, max_df, ngram_range):
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=ngram_range,
                          sublinear_tf=True, min_df=min_df, max_df=max_df)
    D = vec.fit_transform(doc_texts).tocsr().astype(np.float32)
    return vec, D


def retrieve_topn(Q, D, doc_ids, k, floor):
    """C++ top-K cosine. Q: csr queries, D: csr docs. Returns [[(id, s)]] rows."""
    C = sp_matmul_topn(Q.tocsr(), D.T.tocsr(), top_n=k,
                       threshold=float(floor), sort=True, density=k).tocsr()
    indptr, indices, data = C.indptr, C.indices, C.data
    return [[(doc_ids[indices[j]], float(data[j]))
             for j in range(indptr[i], indptr[i + 1])]
            for i in range(C.shape[0])]


def retrieve_loop(Q, D, doc_ids, k, floor):
    """Pure-sklearn fallback: sparse matmul + row-wise argpartition top-K."""
    S = (Q.tocsr().astype(np.float32) @ D.T.tocsr()).tocsr()
    indptr, indices, data = S.indptr, S.indices, S.data
    out = []
    for i in range(S.shape[0]):
        s, e = indptr[i], indptr[i + 1]
        if e == s:
            out.append([])
            continue
        idx, val = indices[s:e], data[s:e]
        if floor > 0:
            keep = val >= floor
            idx, val = idx[keep], val[keep]
            if idx.size == 0:
                out.append([])
                continue
        if idx.size <= k:
            top = np.argsort(-val)
        else:
            part = np.argpartition(-val, k - 1)[:k]
            top = part[np.argsort(-val[part])]
        out.append([(doc_ids[idx[j]], float(val[j])) for j in top])
    return out


def main():
    ap = argparse.ArgumentParser(description="Scale-ready TF-IDF blocking.")
    ap.add_argument("--s1", required=True)
    ap.add_argument("--s2", required=True)
    ap.add_argument("--s3", required=True)
    ap.add_argument("--out", required=True,
                    help="Output path; use {country} for per-country shards.")
    ap.add_argument("--field", choices=["name", "address"], default="name")
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--floor", type=float, default=0.0,
                    help="Min cosine kept (prunes junk early).")
    ap.add_argument("--chunk", type=int, default=20000)
    ap.add_argument("--min-ngram", type=int, default=3)
    ap.add_argument("--max-ngram", type=int, default=5)
    ap.add_argument("--min-df", type=int, default=1)
    ap.add_argument("--max-df", type=float, default=1.0,
                    help="Drop n-grams in more than this fraction of docs.")
    ap.add_argument("--countries", default="all",
                    help="'all' or comma list, e.g. 'India,US'. A subset also "
                    "skips other countries during load (lower RAM).")
    ap.add_argument("--max-q", type=int, default=0,
                    help="Dry run: first N queries per country (0 = all).")
    ap.add_argument("--resume", action="store_true",
                    help="Skip shards whose output file already exists.")
    ap.add_argument("--backend", choices=["auto", "topn", "loop"], default="auto")
    args = ap.parse_args()

    backend = args.backend
    if backend == "auto":
        backend = "topn" if _HAVE_TOPN else "loop"
        if backend == "loop":
            print("WARNING: sparse_dot_topn unavailable; using slower 'loop' backend.")
    if backend == "topn" and not _HAVE_TOPN:
        ap.error("sparse_dot_topn not installed (pip install sparse_dot_topn)")

    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    sharded = "{country}" in args.out
    want = None if args.countries.strip().lower() == "all" else {
        c.strip() for c in args.countries.split(",") if c.strip()}
    if not sharded and want is not None:
        ap.error("single-file --out requires --countries all "
                 "(or add {country} to --out for shards)")

    print(f"Loading sources...", flush=True)
    s1 = load_records(args.s1, want)
    targets = load_records(args.s2, want) | load_records(args.s3, want)
    print(f"Field '{args.field}': {len(s1):,} S1, {len(targets):,} targets | "
          f"backend={backend} k={args.k} floor={args.floor} "
          f"ngrams=({args.min_ngram},{args.max_ngram}) "
          f"min_df={args.min_df} max_df={args.max_df}", flush=True)

    fi = 0 if args.field == "name" else 1
    by_country_tgt = defaultdict(list)
    for tid, rec in targets.items():
        by_country_tgt[rec[2]].append((tid, rec[fi]))
    by_country_q = defaultdict(list)
    for qid, rec in s1.items():
        by_country_q[rec[2]].append((qid, rec[fi]))

    retrieve = retrieve_topn if backend == "topn" else retrieve_loop
    countries = sorted(by_country_q)
    if want is not None:
        countries = [c for c in countries if c in want]
    results = {}
    for ctry in countries:
        out_path = args.out.format(country=ctry) if sharded else args.out
        if sharded and args.resume and os.path.isfile(out_path) and \
                os.path.getsize(out_path) > 100:
            print(f"[{ctry}] SKIP (exists, --resume): {out_path}", flush=True)
            continue
        qlist = by_country_q[ctry]
        if args.max_q:
            qlist = qlist[:args.max_q]
        doc_ids = [t[0] for t in by_country_tgt.get(ctry, [])]
        doc_texts = [t[1] for t in by_country_tgt.get(ctry, [])]
        t0 = time.time()
        if not doc_ids:
            for qid, _ in qlist:
                results[qid] = []
        else:
            vec, D = build_index(doc_texts, args.min_df, args.max_df,
                                 (args.min_ngram, args.max_ngram))
            mb = (D.data.nbytes + D.indices.nbytes + D.indptr.nbytes) / 1e6
            print(f"[{ctry}] index: {len(doc_ids):,} docs, "
                  f"vocab={len(vec.vocabulary_):,}, nnz={D.nnz:,}, ~{mb:.0f}MB "
                  f"({time.time() - t0:.1f}s)", flush=True)
            t1 = time.time()
            for start in range(0, len(qlist), args.chunk):
                batch = qlist[start:start + args.chunk]
                Q = vec.transform([t for _, t in batch]).tocsr().astype(np.float32)
                rows = retrieve(Q, D, doc_ids, args.k, args.floor)
                for (qid, text), cands in zip(batch, rows):
                    results[qid] = cands if text else []
                done = min(start + args.chunk, len(qlist))
                el = time.time() - t1
                eta = el / done * (len(qlist) - done) if done else 0
                print(f"[{ctry}] {done:,}/{len(qlist):,} queries "
                      f"({el:.0f}s elapsed, ETA {eta:.0f}s)", flush=True)
        if sharded:
            with open(out_path, "w", encoding="utf-8", newline="") as f:
                f.write("source1_entity_id\tcandidate_entity_ids\n")
                for qid, _ in qlist:
                    cands = results.get(qid, [])
                    f.write(qid + "\t" + ",".join(f"{i}:{s:.4f}" for i, s in cands) + "\n")
            print(f"[{ctry}] wrote {len(qlist):,} rows -> {out_path} "
                  f"(total {time.time() - t0:.0f}s)", flush=True)
            for qid, _ in qlist:
                results.pop(qid, None)  # free per-shard memory

    if not sharded:
        with open(args.out, "w", encoding="utf-8", newline="") as f:
            f.write("source1_entity_id\tcandidate_entity_ids\n")
            for qid in s1:
                cands = results.get(qid, [])
                f.write(qid + "\t" + ",".join(f"{i}:{s:.4f}" for i, s in cands) + "\n")
        n_nonempty = sum(1 for v in results.values() if v)
        print(f"Wrote {len(s1):,} rows ({n_nonempty:,} non-empty) -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
