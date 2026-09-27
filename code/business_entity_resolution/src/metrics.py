#!/usr/bin/env python3
"""Competition metric: macro-averaged F_beta with beta=0.5.

This is an exact local replica of the leaderboard scoring rule described in
README.md ("Evaluation Criteria"). Use it to score a validation split locally:

    python3 metrics.py --truth val_ground_truth.tsv --pred val_predictions.tsv

Scoring rule (per Source-1 entity, then averaged over ALL S1 entities):
  - truth T = set of true S2/S3 match IDs, pred P = set of predicted IDs
  - both empty        -> 1.0  (correctly identified singleton)
  - P empty, T not    -> 0.0  (missed every match)
  - T empty, P not    -> 0.0  (false merge on a singleton)
  - otherwise         -> F_0.5 = 1.25*P*R / (0.25*P + R)

Both files must be TAB-separated with header:
    source1_entity_id <TAB> matched_entity_ids

Memory note: predictions are loaded into a dict (one entry per S1 entity).
For the full test set (~1.7M rows) that needs roughly 1-2 GB RAM. For tiny
machines, generate predictions in the SAME row order as the truth file and use
--same-order to score in a fully streaming way (almost no RAM).
"""

import argparse
import sys


TRUTH_HEADER = ["source1_entity_id", "matched_entity_ids"]


def parse_id_list(cell):
    """'S2-1,S3-2' -> {'S2-1', 'S3-2'} ; ''/None -> empty set."""
    cell = (cell or "").strip()
    if not cell:
        return set()
    return {x.strip() for x in cell.split(",") if x.strip()}


def prf_for_entity(truth, pred, beta=0.5):
    """Return (F_beta, precision, recall) for ONE Source-1 entity."""
    if not truth and not pred:
        return 1.0, 1.0, 1.0  # correctly predicted singleton
    tp = len(truth & pred)
    if tp == 0:
        return 0.0, 0.0, 0.0
    precision = tp / len(pred)
    recall = tp / len(truth)
    b2 = beta * beta
    f = (1 + b2) * precision * recall / (b2 * precision + recall)
    return f, precision, recall


def read_header(f, path):
    header = f.readline()
    if not header:
        raise ValueError(f"{path} is empty.")
    if "\t" not in header and "," in header:
        raise ValueError(
            f"{path} looks COMMA-separated. Files must be TAB-separated "
            "(write with df.to_csv(sep='\\t', index=False))."
        )
    cols = [c.strip().lower() for c in header.rstrip("\n").split("\t")]
    if cols != TRUTH_HEADER:
        raise ValueError(f"{path}: unexpected header {cols}, expected {TRUTH_HEADER}.")
    return cols


def score_files(truth_path, pred_path, beta=0.5, same_order=False):
    """Score a prediction file against ground truth. Returns a result dict."""
    if same_order:
        return _score_streaming(truth_path, pred_path, beta)

    # Load predictions into memory (allows any row order).
    preds = {}
    with open(pred_path, encoding="utf-8") as f:
        read_header(f, pred_path)
        for line in f:
            if not line.strip():
                continue
            s1, _, rest = line.partition("\t")
            preds[s1.strip()] = parse_id_list(rest)

    n = 0
    sum_f = sum_p = sum_r = 0.0
    n_singleton = n_singleton_ok = 0
    n_missing = 0
    seen = set()
    with open(truth_path, encoding="utf-8") as f:
        read_header(f, truth_path)
        for line in f:
            if not line.strip():
                continue
            s1, _, rest = line.partition("\t")
            s1 = s1.strip()
            seen.add(s1)
            truth = parse_id_list(rest)
            if s1 in preds:
                pred = preds[s1]
            else:
                pred = set()  # missing row counts as "predicted no matches"
                n_missing += 1
            f05, p, r = prf_for_entity(truth, pred, beta)
            n += 1
            sum_f += f05
            sum_p += p
            sum_r += r
            if not truth:
                n_singleton += 1
                if not pred:
                    n_singleton_ok += 1

    n_extra = len(set(preds) - seen)
    return {
        "macro_f05": sum_f / n,
        "macro_precision": sum_p / n,
        "macro_recall": sum_r / n,
        "n_entities": n,
        "n_singletons": n_singleton,
        "singleton_accuracy": (n_singleton_ok / n_singleton) if n_singleton else 1.0,
        "missing_pred_rows": n_missing,
        "extra_pred_rows": n_extra,
        "beta": beta,
    }


def _score_streaming(truth_path, pred_path, beta):
    """Low-RAM scorer: both files must share the exact same row order."""
    n = 0
    sum_f = sum_p = sum_r = 0.0
    n_singleton = n_singleton_ok = 0
    with open(truth_path, encoding="utf-8") as ft, open(pred_path, encoding="utf-8") as fp:
        read_header(ft, truth_path)
        read_header(fp, pred_path)
        for lt, lp in zip(ft, fp):
            if not lt.strip():
                continue
            s1_t, _, rest_t = lt.partition("\t")
            s1_p, _, rest_p = lp.partition("\t")
            if s1_t.strip() != s1_p.strip():
                raise ValueError(
                    "--same-order given but row order differs at "
                    f"'{s1_t.strip()}' vs '{s1_p.strip()}'. Remove the flag."
                )
            truth, pred = parse_id_list(rest_t), parse_id_list(rest_p)
            f05, p, r = prf_for_entity(truth, pred, beta)
            n += 1
            sum_f += f05
            sum_p += p
            sum_r += r
            if not truth:
                n_singleton += 1
                if not pred:
                    n_singleton_ok += 1
    return {
        "macro_f05": sum_f / n,
        "macro_precision": sum_p / n,
        "macro_recall": sum_r / n,
        "n_entities": n,
        "n_singletons": n_singleton,
        "singleton_accuracy": (n_singleton_ok / n_singleton) if n_singleton else 1.0,
        "missing_pred_rows": 0,
        "extra_pred_rows": 0,
        "beta": beta,
    }


def main():
    ap = argparse.ArgumentParser(description="Local replica of the leaderboard F_0.5 scorer.")
    ap.add_argument("--truth", required=True, help="Ground truth TSV (val split).")
    ap.add_argument("--pred", required=True, help="Your predictions TSV (same format).")
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--same-order", action="store_true",
                    help="Streaming low-RAM mode; requires identical row order.")
    args = ap.parse_args()
    try:
        res = score_files(args.truth, args.pred, args.beta, args.same_order)
    except (ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"Macro F_{args.beta}   : {res['macro_f05']:.6f}")
    print(f"Macro precision  : {res['macro_precision']:.6f}")
    print(f"Macro recall     : {res['macro_recall']:.6f}")
    print(f"Entities         : {res['n_entities']}")
    print(f"Singletons       : {res['n_singletons']} "
          f"(accuracy {res['singleton_accuracy']:.4f})")
    if res["missing_pred_rows"] or res["extra_pred_rows"]:
        print(f"WARNING: {res['missing_pred_rows']} S1 rows missing from pred file, "
              f"{res['extra_pred_rows']} extra pred rows not in truth.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
