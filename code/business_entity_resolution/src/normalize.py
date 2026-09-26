"""Text normalization for business names/addresses (v1: deliberately simple).

v1 does ONLY: ASCII folding -> lowercase -> '&' to 'and' -> punctuation to
space -> collapse whitespace. It is shared by every experiment so results
stay comparable.

Known v1 limitations (fixed in later experiments, each backed by a val score):
  - Non-Latin scripts (Devanagari/Tamil/...) fold to '' and are SKIPPED, so
    they can never match in Exp1. This intentionally measures how much recall
    comes from Latin-script records alone.
  - Legal suffixes ('Pvt Ltd', 'Inc', ...) are kept. Stripping them is Exp4+.
"""

import re
import unicodedata

_WS_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^a-z0-9 ]+")


def fold_ascii(text):
    """Strip accents/diacritics: 'Café' -> 'Cafe'. Non-Latin -> ''. """
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")


def normalize_name(name):
    """Normalize a business name. Returns '' when nothing Latin remains."""
    t = fold_ascii(name or "").lower().replace("&", " and ")
    t = _PUNCT_RE.sub(" ", t)
    return _WS_RE.sub(" ", t).strip()


def normalize_address(addr):
    """v1: same rule as names. Address-specific parsing comes later."""
    return normalize_name(addr)
