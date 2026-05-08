"""
Pairwise disagreement metrics across saliency / explanation methods.

Implements the standard set from:
  Krishna et al. 2022 - "The Disagreement Problem in Explainable ML"
  Jukic, Tutek, Snajder ACL 2023 Findings - "Easy to Decide, Hard to Agree"

For two attribution vectors `a` and `b` over the same `n` tokens, we
report:

  - feature_agreement_at_k: |top_k(a) intersect top_k(b)| / k
                            (set overlap of the k most-attributed tokens)
  - rank_agreement_at_k:    |{i : i in top_k(a) and rank_a(i) == rank_b(i)}| / k
                            (overlap *and* same internal ordering)
  - sign_agreement:         fraction of tokens where sign(a_i) == sign(b_i)
  - signed_rank_agreement_at_k: feature_agreement, but pairs only count
                                if their signs also match
  - rank_correlation:       Spearman over the full vector (no top-k cut)

Also provides `pairwise_table()` which takes a dict of {method_name:
attribution_vector} and produces a long-form list of rows usable by
pandas / a downstream stats script.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence


Vec = Sequence[float]


def _topk_indices(v: Vec, k: int) -> list[int]:
    """Indices of the k largest values in `v` (descending order)."""
    n = len(v)
    k = min(k, n)
    return sorted(range(n), key=lambda i: -v[i])[:k]


def _ranks_desc(v: Vec) -> list[int]:
    """Dense rank (0 = largest) over `v`. Ties broken by index."""
    n = len(v)
    order = sorted(range(n), key=lambda i: -v[i])
    ranks = [0] * n
    for r, idx in enumerate(order):
        ranks[idx] = r
    return ranks


# ---------------------------------------------------------------------------
# Per-pair metrics
# ---------------------------------------------------------------------------

def feature_agreement_at_k(a: Vec, b: Vec, k: int) -> float:
    if len(a) != len(b):
        raise ValueError(f"length mismatch: {len(a)} vs {len(b)}")
    if k <= 0 or len(a) == 0:
        return float("nan")
    k = min(k, len(a))
    ta = set(_topk_indices(a, k))
    tb = set(_topk_indices(b, k))
    return len(ta & tb) / k


def rank_agreement_at_k(a: Vec, b: Vec, k: int) -> float:
    if len(a) != len(b):
        raise ValueError(f"length mismatch: {len(a)} vs {len(b)}")
    if k <= 0 or len(a) == 0:
        return float("nan")
    k = min(k, len(a))
    ra = _ranks_desc(a)
    rb = _ranks_desc(b)
    matches = sum(1 for i in range(len(a)) if ra[i] < k and rb[i] < k and ra[i] == rb[i])
    return matches / k


def sign_agreement(a: Vec, b: Vec, *, eps: float = 0.0) -> float:
    if len(a) != len(b):
        raise ValueError(f"length mismatch: {len(a)} vs {len(b)}")
    if not a:
        return float("nan")
    def s(x: float) -> int:
        if x > eps:
            return 1
        if x < -eps:
            return -1
        return 0
    return sum(1 for x, y in zip(a, b) if s(x) == s(y)) / len(a)


def signed_rank_agreement_at_k(a: Vec, b: Vec, k: int) -> float:
    if len(a) != len(b):
        raise ValueError(f"length mismatch: {len(a)} vs {len(b)}")
    if k <= 0 or len(a) == 0:
        return float("nan")
    k = min(k, len(a))
    ta = _topk_indices(a, k)
    tb_set = set(_topk_indices(b, k))
    matches = 0
    for i in ta:
        if i in tb_set and ((a[i] >= 0) == (b[i] >= 0)):
            matches += 1
    return matches / k


def rank_correlation(a: Vec, b: Vec) -> float:
    """Spearman rank correlation. NaN if either vector has zero variance."""
    if len(a) != len(b):
        raise ValueError(f"length mismatch: {len(a)} vs {len(b)}")
    n = len(a)
    if n < 2:
        return float("nan")
    ra = _ranks_desc(a)
    rb = _ranks_desc(b)
    mean = (n - 1) / 2
    num = sum((ra[i] - mean) * (rb[i] - mean) for i in range(n))
    da = math.sqrt(sum((ra[i] - mean) ** 2 for i in range(n)))
    db = math.sqrt(sum((rb[i] - mean) ** 2 for i in range(n)))
    if da == 0 or db == 0:
        return float("nan")
    return num / (da * db)


# ---------------------------------------------------------------------------
# Sentence-level driver
# ---------------------------------------------------------------------------

def pairwise_table(
    explanations: dict[str, Vec],
    *,
    k_values: Iterable[int] = (3, 5, 10),
    sentence_id: str | int | None = None,
) -> list[dict]:
    """For a dict {method: attribution_vec}, compute every pairwise
    disagreement metric and return long-form rows.

    Each row has keys: sentence_id, method_a, method_b, metric, k, value.
    `k` is None for `sign_agreement` and `rank_correlation`.
    """
    methods = sorted(explanations.keys())
    rows: list[dict] = []
    for i, ma in enumerate(methods):
        for mb in methods[i + 1:]:
            a = explanations[ma]
            b = explanations[mb]
            base = {"sentence_id": sentence_id, "method_a": ma, "method_b": mb}
            rows.append({**base, "metric": "rank_correlation", "k": None,
                         "value": rank_correlation(a, b)})
            rows.append({**base, "metric": "sign_agreement", "k": None,
                         "value": sign_agreement(a, b)})
            for k in k_values:
                rows.append({**base, "metric": "feature_agreement", "k": k,
                             "value": feature_agreement_at_k(a, b, k)})
                rows.append({**base, "metric": "rank_agreement", "k": k,
                             "value": rank_agreement_at_k(a, b, k)})
                rows.append({**base, "metric": "signed_rank_agreement", "k": k,
                             "value": signed_rank_agreement_at_k(a, b, k)})
    return rows


def aggregate(rows: list[dict]) -> dict[tuple[str, str, str, int | None], float]:
    """Mean over sentences, keyed by (method_a, method_b, metric, k)."""
    bucket: dict[tuple, list[float]] = {}
    for r in rows:
        key = (r["method_a"], r["method_b"], r["metric"], r["k"])
        v = r["value"]
        if v is None or (isinstance(v, float) and math.isnan(v)):
            continue
        bucket.setdefault(key, []).append(v)
    return {k: sum(vs) / len(vs) for k, vs in bucket.items() if vs}
