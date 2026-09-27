"""Pairwise string-similarity features, v1 (Exp3).

pair_features(s1_name, s1_addr, t_name, t_addr) -> dict[str, float].
Takes RAW strings; normalization happens inside via shared normalize.py.

v1 = 18 interpretable string features (no embeddings). Each has a job:
  - exact flags catch verbatim copies (Exp1's strength, kept as features)
  - fuzzy ratios catch typos/abbreviations/suffix noise (Exp1's FN pattern)
  - token containment catches DBA/domain names ('Dr Silver (Energy)...')
  - address features veto generic-name false merges (Exp1's FP pattern)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from normalize import normalize_name  # noqa: E402

from rapidfuzz import fuzz


def _toks(text):
    return [t for t in text.split(" ") if t]


def _trigrams(text):
    t = text.replace(" ", " ")
    return {t[i:i + 3] for i in range(max(0, len(t) - 2))} if len(t) >= 3 else ({t} if t else set())


def _jacc(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _dice(a, b):
    if not a or not b:
        return 0.0
    return 2 * len(a & b) / (len(a) + len(b))


def pair_features(s1_name, s1_addr, t_name, t_addr):
    n1, a1 = normalize_name(s1_name), normalize_name(s1_addr)
    n2, a2 = normalize_name(t_name), normalize_name(t_addr)
    nt1, nt2 = set(_toks(n1)), set(_toks(n2))
    at1, at2 = set(_toks(a1)), set(_toks(a2))
    num1 = {t for t in _toks(n1 + " " + a1) if any(c.isdigit() for c in t)}
    num2 = {t for t in _toks(n2 + " " + a2) if any(c.isdigit() for c in t)}
    return {
        # --- name ---
        "name_eq": 1.0 if n1 and n1 == n2 else 0.0,
        "name_ratio": fuzz.ratio(n1, n2) / 100.0,
        "name_partial": fuzz.partial_ratio(n1, n2) / 100.0,
        "name_tsort": fuzz.token_sort_ratio(n1, n2) / 100.0,
        "name_tset": fuzz.token_set_ratio(n1, n2) / 100.0,
        "name_jacc": _jacc(nt1, nt2),
        "name_tridice": _dice(_trigrams(n1), _trigrams(n2)),
        "name_lendiff": abs(len(n1) - len(n2)),
        "name_lenratio": min(len(n1), len(n2)) / max(len(n1), len(n2)) if max(len(n1), len(n2)) else 0.0,
        "name_cover_recall": len(nt1 & nt2) / len(nt1) if nt1 else 0.0,
        "name_cover_prec": len(nt1 & nt2) / len(nt2) if nt2 else 0.0,
        # --- address ---
        "addr_eq": 1.0 if a1 and a1 == a2 else 0.0,
        "addr_ratio": fuzz.ratio(a1, a2) / 100.0,
        "addr_tset": fuzz.token_set_ratio(a1, a2) / 100.0,
        "addr_jacc": _jacc(at1, at2),
        "addr_tridice": _dice(_trigrams(a1), _trigrams(a2)),
        "tgt_addr_empty": 1.0 if not a2 else 0.0,
        "numtok_jacc": _jacc(num1, num2),
    }


FEATURE_NAMES = [
    "name_eq", "name_ratio", "name_partial", "name_tsort", "name_tset",
    "name_jacc", "name_tridice", "name_lendiff", "name_lenratio",
    "name_cover_recall", "name_cover_prec",
    "addr_eq", "addr_ratio", "addr_tset", "addr_jacc", "addr_tridice",
    "tgt_addr_empty", "numtok_jacc",
]


# ---------------------------------------------------------------------------
# v2 additions (Exp4): blocking-signal features.
# Retrieval strength is orthogonal evidence: a truth retrieved at rank 1 with
# cosine 0.9 differs from one at rank 87 with 0.15, even when the string
# similarities look alike. The union file alone cannot tell them apart.
# ---------------------------------------------------------------------------

BLOCKING_NAMES = [
    "bname_hit", "bname_score", "bname_rank",
    "baddr_hit", "baddr_score", "baddr_rank",
    "bbest_rank", "n_cands",
]

FEATURE_NAMES_V2 = FEATURE_NAMES + BLOCKING_NAMES


def pair_features_v2(s1_name, s1_addr, t_name, t_addr,
                     name_hit=None, addr_hit=None, n_cands=0):
    """v1 string features + blocking signals.

    name_hit/addr_hit: (score, rank) from that retrieval channel, or None
    when the pair was not retrieved by that channel. Ranks are 1-based.
    n_cands: union size for this S1 (crowded neighborhoods are riskier).
    """
    f = pair_features(s1_name, s1_addr, t_name, t_addr)
    ranks = []
    if name_hit is None:
        f.update({"bname_hit": 0.0, "bname_score": 0.0, "bname_rank": 0.0})
    else:
        s, r = name_hit
        f.update({"bname_hit": 1.0, "bname_score": float(s), "bname_rank": float(r)})
        ranks.append(r)
    if addr_hit is None:
        f.update({"baddr_hit": 0.0, "baddr_score": 0.0, "baddr_rank": 0.0})
    else:
        s, r = addr_hit
        f.update({"baddr_hit": 1.0, "baddr_score": float(s), "baddr_rank": float(r)})
        ranks.append(r)
    f["bbest_rank"] = float(min(ranks)) if ranks else 0.0
    f["n_cands"] = float(n_cands)
    return f
