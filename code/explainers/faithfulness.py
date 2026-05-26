"""
Deletion / insertion AUC for token-attribution explanations on NER.

For a sentence `words` with attribution vector `attribution` and a
target tag `target_tag` at position `target_idx`:

  deletion_auc:  start with the original sentence, remove tokens in
    order of decreasing attribution, plot P(target_tag | x) at each
    step, return the area under that curve. A faithful explanation
    drops the probability quickly.

  insertion_auc: start with all tokens replaced by a baseline (e.g.
    [UNK] or a constant string), insert tokens in decreasing
    attribution order, plot P(target_tag | x), return the area.
    A faithful explanation raises the probability quickly.

We follow Petsiuk et al. 2018 (RISE / DeleteAndInsert tests).

The metrics are model-agnostic: callers pass a `score_fn(words_list)
-> float` that returns the model's probability of the target tag at
the target position. This isolates the metric from the specific
model framework (TF / PyTorch / black-box).

For the disagreement study, what we care about is not absolute
faithfulness but *whether the gap between deletion and insertion AUC
is comparable across domains for each method*. If it is, then any
biomed-vs-CoNLL disagreement is real, not just one method silently
failing on biomed.
"""

from __future__ import annotations

from typing import Callable, Sequence


ScoreFn = Callable[[list[str]], float]
BASELINE_TOKEN = "[UNK]"


def _trapz(ys: list[float]) -> float:
    """Trapezoidal AUC on equally-spaced points in [0,1]."""
    if len(ys) < 2:
        return float("nan")
    n = len(ys) - 1
    return sum((ys[i] + ys[i + 1]) / 2 for i in range(n)) / n


def deletion_auc(
    words: Sequence[str],
    attribution: Sequence[float],
    score_fn: ScoreFn,
    *,
    baseline: str = BASELINE_TOKEN,
) -> float:
    if len(words) != len(attribution):
        raise ValueError("words and attribution length mismatch")
    if not words:
        return float("nan")
    order = sorted(range(len(words)), key=lambda i: -attribution[i])
    cur = list(words)
    ys = [score_fn(cur)]
    for i in order:
        cur[i] = baseline
        ys.append(score_fn(cur))
    return _trapz(ys)


def insertion_auc(
    words: Sequence[str],
    attribution: Sequence[float],
    score_fn: ScoreFn,
    *,
    baseline: str = BASELINE_TOKEN,
) -> float:
    if len(words) != len(attribution):
        raise ValueError("words and attribution length mismatch")
    if not words:
        return float("nan")
    order = sorted(range(len(words)), key=lambda i: -attribution[i])
    cur = [baseline] * len(words)
    ys = [score_fn(cur)]
    for i in order:
        cur[i] = words[i]
        ys.append(score_fn(cur))
    return _trapz(ys)


def faithfulness_pair(
    words: Sequence[str],
    attribution: Sequence[float],
    score_fn: ScoreFn,
    *,
    baseline: str = BASELINE_TOKEN,
) -> dict[str, float]:
    """Return both AUCs and the gap. The gap (`insertion - deletion`)
    is what to compare across domains for a single method."""
    d = deletion_auc(words, attribution, score_fn, baseline=baseline)
    i = insertion_auc(words, attribution, score_fn, baseline=baseline)
    return {"deletion_auc": d, "insertion_auc": i, "gap": i - d}
