#!/usr/bin/env python3
"""Exp3: LightGBM pair classifier + threshold tuned on val macro-F0.5.

Pipeline:
  1. Load train/val S1 id sets (split.py), union candidates, all sources.
  2. Featurize (train S1 x their candidates) with labels from train GT.
     All candidates are used (they already resemble inference: similar
     names/addresses, mostly negatives). Imbalance (~3-4% positive) is
     handled with scale_pos_weight, NOT resampling (keeps inference
     distribution intact for honest threshold tuning).
  3. Train LightGBM (early stopping on val-pair logloss picks #trees).
  4. Sweep the decision threshold on VAL macro-F0.5 (the actual objective);
     coarse grid then refine. Saves model.txt + threshold.json + val preds.

Usage:
    python3 train.py --s1 sample/sample_s1.csv --s2 sample/sample_s2.csv \\
        --s3 sample/sample_s3.csv --train-gt sample/split/train_gt.tsv \\
        --val-gt sample/split/val_gt.tsv --candidates output/cand_union.tsv \\
        --out-dir output/exp3 --seed 42
"""

import argparse
import csv
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import FEATURE_NAMES, pair_features  # noqa: E402
from metrics import prf_for_entity  # noqa: E402

import lightgbm as lgb


def load_sources(*paths):
    out = {}
    for path in paths:
        with open(path, encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                out[row["entity_id"]] = (row.get("business_name") or "",
                                         row.get("business_address") or "")
    return out


def load_gt(path):
    out = {}
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            cell = (row.get("matched_entity_ids") or "").strip()
            out[row["source1_entity_id"]] = (
                {x.strip() for x in cell.split(",") if x.strip()} if cell else set()
            )
    return out


def load_cands(path):
    out = {}
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            cell = (row.get("candidate_entity_ids") or "").strip()
            out[row["source1_entity_id"]] = (
                [x.split(":")[0] for x in cell.split(",") if x.strip()] if cell else []
            )
    return out


def build_matrix(s1_ids, cands, truth, sources):
    n_feat = len(FEATURE_NAMES)
    Xs = np.empty((0, n_feat), dtype=np.float32)
    ys = np.empty((0,), dtype=np.int8)
    idx = []
    buf_x, buf_y = [], []
    for qi, q in enumerate(s1_ids):
        t = truth.get(q, set())
        sn, sa = sources[q]
        for tid in cands.get(q, []):
            tn, ta = sources[tid]
            f = pair_features(sn, sa, tn, ta)
            buf_x.append([f[n] for n in FEATURE_NAMES])
            buf_y.append(1 if tid in t else 0)
            idx.append((q, tid))
        if (qi + 1) % 2000 == 0 or qi + 1 == len(s1_ids):
            Xs = np.vstack([Xs, np.asarray(buf_x, dtype=np.float32)])
            ys = np.concatenate([ys, np.asarray(buf_y, dtype=np.int8)])
            buf_x, buf_y = [], []
            print(f"  featurized {qi + 1}/{len(s1_ids)} S1s "
                  f"({len(ys):,} pairs)", flush=True)
    return Xs, ys, idx


def score_threshold(val_ids, val_truth, idx_s1, idx_tid, proba, tau):
    sum_f = sum_p = sum_r = 0.0
    n_sing = n_sing_ok = 0
    # group kept tids by s1
    preds = {}
    for s1id, tid, p in zip(idx_s1, idx_tid, proba):
        if p >= tau:
            preds.setdefault(s1id, set()).add(tid)
    for s1id in val_ids:
        t = val_truth.get(s1id, set())
        pr = preds.get(s1id, set())
        f, p, r = prf_for_entity(t, pr, 0.5)
        sum_f += f
        sum_p += p
        sum_r += r
        if not t:
            n_sing += 1
            if not pr:
                n_sing_ok += 1
    n = len(val_ids)
    return {"f05": sum_f / n, "prec": sum_p / n, "rec": sum_r / n,
            "sing_acc": (n_sing_ok / n_sing) if n_sing else 1.0}


def main():
    ap = argparse.ArgumentParser(description="Exp3: LightGBM + F0.5 threshold.")
    ap.add_argument("--s1", required=True)
    ap.add_argument("--s2", required=True)
    ap.add_argument("--s3", required=True)
    ap.add_argument("--train-gt", required=True)
    ap.add_argument("--val-gt", required=True)
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    t0 = time.time()

    sources = load_sources(args.s1, args.s2, args.s3)
    train_truth, val_truth = load_gt(args.train_gt), load_gt(args.val_gt)
    cands = load_cands(args.candidates)
    train_ids = sorted(train_truth)
    val_ids = sorted(val_truth)
    print(f"Train S1: {len(train_ids):,} | Val S1: {len(val_ids):,}", flush=True)

    print("Featurizing train pairs...", flush=True)
    Xtr, ytr, _ = build_matrix(train_ids, cands, train_truth, sources)
    print("Featurizing val pairs...", flush=True)
    Xva, yva, idx_va = build_matrix(val_ids, cands, val_truth, sources)
    pos, neg = int(ytr.sum()), int((1 - ytr).sum())
    print(f"Train pairs: {len(ytr):,} (pos {pos:,} = {100 * pos / len(ytr):.2f}%) | "
          f"Val pairs: {len(yva):,}", flush=True)

    params = {"objective": "binary", "metric": "binary_logloss",
              "learning_rate": 0.05, "num_leaves": 63,
              "min_data_in_leaf": 200, "feature_fraction": 0.8,
              "bagging_fraction": 0.8, "bagging_freq": 1,
              "scale_pos_weight": neg / max(pos, 1),
              "verbosity": -1, "seed": args.seed,
              "deterministic": True, "num_threads": 2}
    dtr = lgb.Dataset(Xtr, label=ytr)
    dva = lgb.Dataset(Xva, label=yva, reference=dtr)
    print("Training LightGBM...", flush=True)
    bst = lgb.train(params, dtr, num_boost_round=1000,
                    valid_sets=[dva],
                    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)])
    print(f"Best iteration: {bst.best_iteration}", flush=True)
    bst.save_model(os.path.join(args.out_dir, "model.txt"))

    pva = bst.predict(Xva, num_iteration=bst.best_iteration)
    idx_s1 = [q for q, _ in idx_va]
    idx_tid = [t for _, t in idx_va]

    print("\nCoarse threshold sweep (val macro-F0.5):", flush=True)
    print(f"  {'tau':>5} {'F0.5':>8} {'prec':>8} {'rec':>8} {'sing_acc':>9}")
    coarse = np.arange(0.05, 0.96, 0.05)
    results = {}
    for tau in coarse:
        r = score_threshold(val_ids, val_truth, idx_s1, idx_tid, pva, float(tau))
        results[float(tau)] = r
        print(f"  {tau:>5.2f} {r['f05']:>8.4f} {r['prec']:>8.4f} "
              f"{r['rec']:>8.4f} {r['sing_acc']:>9.4f}", flush=True)
    best_coarse = max(results, key=lambda t: results[t]["f05"])
    fine = [round(x, 2) for x in np.arange(max(0.01, best_coarse - 0.04),
                                              min(0.99, best_coarse + 0.041), 0.01)]
    for tau in fine:
        if tau not in results:
            results[tau] = score_threshold(val_ids, val_truth, idx_s1, idx_tid, pva, tau)
    best = max(results, key=lambda t: results[t]["f05"])
    r = results[best]
    print(f"\nBEST tau={best:.2f}: F0.5={r['f05']:.4f} prec={r['prec']:.4f} "
          f"rec={r['rec']:.4f} sing_acc={r['sing_acc']:.4f}")

    with open(os.path.join(args.out_dir, "threshold.json"), "w") as f:
        json.dump({"tau": best, "val_f05": r["f05"], "val_prec": r["prec"],
                   "val_rec": r["rec"], "val_sing_acc": r["sing_acc"],
                   "best_iteration": bst.best_iteration}, f, indent=2)
    with open(os.path.join(args.out_dir, "val_pred_best.tsv"), "w") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        preds = {}
        for s1id, tid, p in zip(idx_s1, idx_tid, pva):
            if p >= best:
                preds.setdefault(s1id, []).append(tid)
        for s1id in val_ids:
            f.write(f"{s1id}\t{','.join(preds.get(s1id, []))}\n")

    imp = sorted(zip(FEATURE_NAMES, bst.feature_importance(importance_type="gain")),
                 key=lambda kv: -kv[1])
    print("\nFeature importance (gain):")
    for name, g in imp:
        print(f"  {name:>18}: {g:,.0f}")
    print(f"\nDone in {(time.time() - t0) / 60:.1f} min. Artifacts -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
