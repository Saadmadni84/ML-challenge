#!/usr/bin/env python3
"""Exp2: TF-IDF character-ngram blocking (candidate generation).

For each S1 entity, retrieve the top-K most similar S2/S3 records **within
the same country** by TF-IDF cosine similarity over character n-grams, using
--field name (default) or --field address. Run once per field, then merge
channels with union_candidates.py.

Why char n-grams: the SET of n-grams is insensitive to word order ('A B' vs
'B A' share most n-grams) yet sensitive to word content, and near-matches
with typos still share many n-grams ('flight' vs 'fligth' share 'fli','lig',
'igh'...). This catches the suffix/punctuation/typo variants that Exp1's
exact match misses.

Writes an INTERNAL candidates file (scores kept so recall@K can be measured
at any K <= --k by truncation):
    source1_entity_id <TAB> cand_id:score,cand_id:score,...

The official candidate_pairs.tsv is this file with scores stripped (done at
submission time). Every S1 gets exactly one row (possibly empty).

Design notes:
  - One TF-IDF index per country (hard country filter built in).
  - Vectorizer is fit on TARGETS only (never on S1 queries): at inference
    time S1 rows are unseen, so this mirrors reality and avoids any leakage
    debate about fitting on queries.
  - Queries are scored in chunks to bound RAM (same code path scales to
    full data; only chunk size / K need tuning).

Usage:
    python3 blocking.py --s1 sample/sample_s1.csv --s2 sample/sample_s2.csv \\
        --s3 sample/sample_s3.csv --out output/cand.tsv --k 100
    python3 blocking.py --field address ... --out output/cand_addr.tsv
"""

import argparse
import csv
import os
import sys
from collections import defaultdict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from normalize import normalize_name  # noqa: E402


def load_records(path):
    """id -> (normalized_name, normalized_address, country)."""
    recs = {}
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            recs[row["entity_id"]] = (
                normalize_name(row.get("business_name")),
                normalize_name(row.get("business_address")),
                (row.get("country") or "").strip(),
            )
    return recs


def topk_for_chunk(sims, doc_ids, k):
    """sims: (n_queries, n_docs) dense array. Returns list of [(id, score)]."""
    out = []
    for scores in sims:
        nz = np.flatnonzero(scores > 0.0)
        if nz.size == 0:
            out.append([])
            continue
        if nz.size <= k:
            top = nz[np.argsort(-scores[nz])]
        else:
            part = np.argpartition(-scores[nz], k - 1)[:k]
            top = nz[part[np.argsort(-scores[nz][part])]]
        out.append([(doc_ids[i], float(scores[i])) for i in top])
    return out


def main():
    ap = argparse.ArgumentParser(description="Exp2: TF-IDF char-ngram blocking.")
    ap.add_argument("--s1", required=True)
    ap.add_argument("--s2", required=True)
    ap.add_argument("--s3", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--field", choices=["name", "address"], default="name")
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--chunk", type=int, default=2000,
                    help="Queries scored per batch (RAM control).")
    ap.add_argument("--min-ngram", type=int, default=3)
    ap.add_argument("--max-ngram", type=int, default=5)
    args = ap.parse_args()

    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)

    s1 = load_records(args.s1)
    targets = load_records(args.s2) | load_records(args.s3)
    print(f"Field '{args.field}': {len(s1):,} S1 queries, "
          f"{len(targets):,} S2/S3 targets.", flush=True)

    fi = 0 if args.field == "name" else 1
    by_country_tgt = defaultdict(list)
    for tid, rec in targets.items():
        by_country_tgt[rec[2]].append((tid, rec[fi]))
    by_country_q = defaultdict(list)
    for qid, rec in s1.items():
        by_country_q[rec[2]].append((qid, rec[fi]))

    results = {}
    n_empty_q = 0
    for ctry, qlist in sorted(by_country_q.items()):
        doc_ids = [t[0] for t in by_country_tgt.get(ctry, [])]
        doc_texts = [t[1] for t in by_country_tgt.get(ctry, [])]
        if not doc_ids:
            for qid, _ in qlist:
                results[qid] = []
            continue
        vec = TfidfVectorizer(analyzer="char_wb",
                              ngram_range=(args.min_ngram, args.max_ngram),
                              sublinear_tf=True)
        D = vec.fit_transform(doc_texts)
        print(f"[{ctry}] index: {len(doc_ids):,} docs, vocab={len(vec.vocabulary_):,}",
              flush=True)
        for start in range(0, len(qlist), args.chunk):
            batch = qlist[start:start + args.chunk]
            Q = vec.transform([t for _, t in batch])
            sims = cosine_similarity(Q, D, dense_output=True)
            for (qid, text), cands in zip(batch, topk_for_chunk(sims, doc_ids, args.k)):
                results[qid] = cands if text else []
                if not text:
                    n_empty_q += 1
        print(f"[{ctry}] retrieved top-{args.k} for {len(qlist):,} queries.", flush=True)

    with open(args.out, "w", encoding="utf-8", newline="") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for qid in s1:
            cands = results.get(qid, [])
            f.write(qid + "\t" + ",".join(f"{i}:{s:.4f}" for i, s in cands) + "\n")
    n_nonempty = sum(1 for v in results.values() if v)
    print(f"Wrote {len(s1):,} rows ({n_nonempty:,} non-empty, "
          f"{n_empty_q} empty-text queries) -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
