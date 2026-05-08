"""
Entity-level NER evaluation using seqeval (CoNLL-2003 scheme).

Replaces the previous hand-rolled token-confusion-matrix metric, which
inflated F1 by counting the dominant `O` class. The new module reports:

  - per-entity precision / recall / F1 (entity-level, micro and macro)
  - a token-level classification report (for sanity-check only)
  - the seqeval `classification_report` text block

The legacy functions `PrecisionRecall`, `PrecisionRecallEntityLevel`, and
`PrecisionRecallEntityLevelGene` are kept as deprecated shims that
forward to the new evaluator after stripping the repo's `|TAG\\n` quirk.
"""

from __future__ import annotations

import warnings
from typing import Iterable, Sequence

from seqeval.metrics import (
    classification_report as seqeval_report,
    f1_score as seqeval_f1,
    precision_score as seqeval_precision,
    recall_score as seqeval_recall,
)
from seqeval.scheme import IOB2
from sklearn.metrics import classification_report as sk_report


def _clean(tag: str) -> str:
    t = tag.strip().lstrip("|").strip()
    if t in {"", "O"}:
        return "O"
    if "-" not in t:
        return "O"
    prefix, label = t.split("-", 1)
    return f"{prefix}-{label.upper()}"


def _normalise(seqs: Iterable[Iterable[str]]) -> list[list[str]]:
    return [[_clean(t) for t in s] for s in seqs]


def evaluate(
    y_true: Sequence[Sequence[str]],
    y_pred: Sequence[Sequence[str]],
    *,
    digits: int = 4,
    print_report: bool = True,
) -> dict:
    """Entity-level evaluation using seqeval (strict IOB2 scheme).

    `y_true` and `y_pred` are lists of tag sequences (one per sentence),
    aligned token-for-token. Returns a dict with `precision`, `recall`,
    `f1`, and the full text reports.
    """
    y_t = _normalise(y_true)
    y_p = _normalise(y_pred)
    if len(y_t) != len(y_p):
        raise ValueError(f"Length mismatch: {len(y_t)} vs {len(y_p)} sentences")
    for i, (a, b) in enumerate(zip(y_t, y_p)):
        if len(a) != len(b):
            raise ValueError(f"Token-count mismatch in sentence {i}: {len(a)} vs {len(b)}")

    p = seqeval_precision(y_t, y_p, mode="strict", scheme=IOB2)
    r = seqeval_recall(y_t, y_p, mode="strict", scheme=IOB2)
    f1 = seqeval_f1(y_t, y_p, mode="strict", scheme=IOB2)
    entity_report = seqeval_report(y_t, y_p, digits=digits, mode="strict", scheme=IOB2)

    flat_t = [t for s in y_t for t in s]
    flat_p = [t for s in y_p for t in s]
    token_report = sk_report(flat_t, flat_p, digits=digits, zero_division=0)

    if print_report:
        print("=== Entity-level (seqeval, strict IOB2) ===")
        print(entity_report)
        print(f"micro-precision: {p:.{digits}f}  micro-recall: {r:.{digits}f}  micro-F1: {f1:.{digits}f}")
        print("=== Token-level (sanity check, includes O) ===")
        print(token_report)

    return {
        "precision": p,
        "recall": r,
        "f1": f1,
        "entity_report": entity_report,
        "token_report": token_report,
    }


def evaluate_flat(
    tokens_flat: Sequence[str],
    y_true_flat: Sequence[str],
    y_pred_flat: Sequence[str],
    *,
    sentence_break_token: str = ".",
    digits: int = 4,
    print_report: bool = True,
) -> dict:
    """Convenience: accepts flat token-aligned lists and re-segments by `.`."""
    y_t_sents: list[list[str]] = []
    y_p_sents: list[list[str]] = []
    cur_t: list[str] = []
    cur_p: list[str] = []
    for tok, t, p in zip(tokens_flat, y_true_flat, y_pred_flat):
        cur_t.append(_clean(t))
        cur_p.append(_clean(p))
        if tok == sentence_break_token:
            y_t_sents.append(cur_t)
            y_p_sents.append(cur_p)
            cur_t, cur_p = [], []
    if cur_t:
        y_t_sents.append(cur_t)
        y_p_sents.append(cur_p)
    return evaluate(y_t_sents, y_p_sents, digits=digits, print_report=print_report)


# ---------------------------------------------------------------------------
# Deprecated shims kept so existing scripts (main_NB.py) don't break.
# ---------------------------------------------------------------------------

def _deprecated(name: str) -> None:
    warnings.warn(
        f"{name} is deprecated; use Evaluation.evaluate / evaluate_flat. "
        "The old token-confusion metric inflated F1 by averaging in the "
        "dominant `O` class.",
        DeprecationWarning,
        stacklevel=3,
    )


def PrecisionRecall(y_pred, y_true):
    _deprecated("PrecisionRecall")
    return evaluate_flat(["x"] * len(y_true), y_true, y_pred, sentence_break_token=" ")


def PrecisionRecallEntityLevel(Xtest, y_pred, y_true):
    _deprecated("PrecisionRecallEntityLevel")
    return evaluate_flat(Xtest, y_true, y_pred, sentence_break_token=".")


def PrecisionRecallEntityLevelGene(Xtest, y_pred, y_true):
    _deprecated("PrecisionRecallEntityLevelGene")
    return evaluate_flat(Xtest, y_true, y_pred, sentence_break_token=".")
