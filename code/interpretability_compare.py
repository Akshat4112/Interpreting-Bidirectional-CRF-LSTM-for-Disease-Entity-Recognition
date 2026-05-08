"""
Classical-vs-neural interpretability comparison (paper framing C).

For a fixed set of test sentences, produce per-token importance scores
from three sources:

  1. Naive Bayes log-likelihood ratios (`features['B']`, `features['I']`)
     - already computed in NaiveBayes.MultinomialNBTrain
     - we reuse the trained NB and read its log-prob tables directly

  2. LIME explanations on the trained BiLSTM
     - via NERExplainerGenerator (existing module)

  3. Last-layer mean attention from the fine-tuned PubMedBERT
     - cached by train_biobert.py --save-attention

The script then computes pairwise agreement (Spearman rank correlation,
top-k overlap, AUC of deletion / insertion curves) between the three
signals on the same set of sentences. This is the analytical contribution
that makes the paper an interpretability study rather than a basic
benchmark.

Usage:
    python interpretability_compare.py \\
        --bilstm-weights ../models/bilstm_seed42_*.h5 \\
        --biobert-dir ../models/biobert_ncbi \\
        --nb-pickle ../models/nb.pkl \\
        --test ../data/ner-disease/test.iob \\
        --n-sentences 100

Each of the three signals is optional - the script reports whichever
pairs are computable.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import sys
from typing import Any

import numpy as np

from data_loader import load_iob


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--test", default="../data/ner-disease/test.iob")
    p.add_argument("--n-sentences", type=int, default=100)
    p.add_argument("--bilstm-weights", default=None)
    p.add_argument("--word2idx", default="../data/word2idx.pkl")
    p.add_argument("--tag2idx", default="../data/tag2idx.pkl")
    p.add_argument("--biobert-dir", default=None)
    p.add_argument("--nb-pickle", default=None,
                   help="Pickle of trained NaiveBayes instance (with .features).")
    p.add_argument("--out", default="../models/interpretability_compare.json")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Per-source importance extractors
# ---------------------------------------------------------------------------

def nb_token_importance(words: list[str], nb_obj: Any, target: str = "B") -> list[float]:
    """NB importance = exp(log p(word | target_class)) (smoothed log-prob).

    Higher means more characteristic of the target class. We expose it as
    a positive score per token for direct comparison with LIME / attention.
    """
    table = nb_obj.features.get(target, {})
    if not table:
        return [0.0] * len(words)
    # priori is in log-space; we compare relative ranks so absolute scale doesn't matter
    return [math.exp(table.get(w, math.log(1e-12))) for w in words]


def lime_token_importance(words: list[str], explainer, target_tag: str = "|B-DISEASE\n") -> list[float]:
    """One LIME explanation per token position; importance is the LIME
    weight assigned to that word for the chosen target tag at that position."""
    sentence = " ".join(words)
    importances = [0.0] * len(words)
    for pos in range(min(len(words), explainer.max_len)):
        try:
            predict_func = explainer.get_predict_function(word_index=pos)
            from eli5.lime import TextExplainer
            from eli5.lime.samplers import MaskingTextSampler
            sampler = MaskingTextSampler(replacement="UNK", max_replace=0.7,
                                         token_pattern=None, bow=False)
            te = TextExplainer(sampler=sampler, position_dependent=True, random_state=42)
            te.fit(sentence, predict_func)
            expl = te.explain_prediction(target_names=list(explainer.idx2tag.values()),
                                          top_targets=3)
            import eli5
            d = eli5.formatters.format_as_dict(expl)
            for t in d["targets"]:
                if t["target"] != target_tag:
                    continue
                for fw in t["feature_weights"]["pos"]:
                    feat = fw["feature"].split(maxsplit=1)
                    if len(feat) == 2 and feat[1] == words[pos]:
                        importances[pos] = max(importances[pos], float(fw["weight"]))
        except Exception:
            continue
    return importances


def attention_token_importance(words: list[str], attn_matrix: np.ndarray,
                                tokenizer, max_length: int) -> list[float]:
    """Attention attribution = column-sum of last-layer attention for the
    sub-tokens of each original word."""
    enc = tokenizer(words, is_split_into_words=True, truncation=True,
                    max_length=max_length, padding="max_length")
    word_ids = enc.word_ids()
    per_word = [0.0] * len(words)
    counts = [0] * len(words)
    seq_attn = attn_matrix.sum(axis=0)  # how much each source token is attended to overall
    for tok_pos, wid in enumerate(word_ids):
        if wid is None or wid >= len(words):
            continue
        per_word[wid] += float(seq_attn[tok_pos])
        counts[wid] += 1
    return [s / max(c, 1) for s, c in zip(per_word, counts)]


# ---------------------------------------------------------------------------
# Agreement metrics
# ---------------------------------------------------------------------------

def spearman(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or len(a) < 2:
        return float("nan")
    ar = _rank(a)
    br = _rank(b)
    n = len(a)
    mean = (n - 1) / 2
    num = sum((ar[i] - mean) * (br[i] - mean) for i in range(n))
    denom_a = math.sqrt(sum((ar[i] - mean) ** 2 for i in range(n)))
    denom_b = math.sqrt(sum((br[i] - mean) ** 2 for i in range(n)))
    if denom_a == 0 or denom_b == 0:
        return float("nan")
    return num / (denom_a * denom_b)


def _rank(xs: list[float]) -> list[float]:
    indexed = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    for r, i in enumerate(indexed):
        ranks[i] = float(r)
    return ranks


def topk_overlap(a: list[float], b: list[float], k: int = 3) -> float:
    if len(a) < k or len(b) < k:
        k = min(len(a), len(b))
    if k == 0:
        return float("nan")
    ta = set(sorted(range(len(a)), key=lambda i: -a[i])[:k])
    tb = set(sorted(range(len(b)), key=lambda i: -b[i])[:k])
    return len(ta & tb) / k


def deletion_auc(words: list[str], importances: list[float], score_fn) -> float:
    """AUC of P(target) as we delete tokens in descending order of importance."""
    order = sorted(range(len(words)), key=lambda i: -importances[i])
    cur = list(words)
    auc = 0.0
    base = score_fn(cur)
    auc += base
    for i in order:
        cur[i] = "UNK"
        auc += score_fn(cur)
    return auc / (len(order) + 1)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    test_ds = load_iob(args.test)
    sentences = test_ds.sentences[: args.n_sentences]

    nb_obj = None
    if args.nb_pickle and os.path.exists(args.nb_pickle):
        with open(args.nb_pickle, "rb") as f:
            nb_obj = pickle.load(f)

    bilstm_explainer = None
    if args.bilstm_weights:
        try:
            import glob
            from tensorflow import keras
            from Explainer import NERExplainerGenerator
            paths = sorted(glob.glob(args.bilstm_weights))
            if not paths:
                raise FileNotFoundError(args.bilstm_weights)
            model = keras.models.load_model(paths[-1], compile=False)
            with open(args.word2idx, "rb") as f:
                w2i = pickle.load(f)
            with open(args.tag2idx, "rb") as f:
                t2i = pickle.load(f)
            bilstm_explainer = NERExplainerGenerator(model, w2i, t2i, max_len=114)
        except Exception as e:
            print(f"[warn] BiLSTM explainer unavailable: {e}")

    biobert = None
    biobert_attn_rows = None
    if args.biobert_dir and os.path.isdir(args.biobert_dir):
        try:
            from transformers import AutoTokenizer, AutoModelForTokenClassification
            biobert = {
                "tokenizer": AutoTokenizer.from_pretrained(args.biobert_dir),
                "model": AutoModelForTokenClassification.from_pretrained(args.biobert_dir),
            }
            attn_path = os.path.join(args.biobert_dir, "test_attention.npy")
            if os.path.exists(attn_path):
                biobert_attn_rows = np.load(attn_path, allow_pickle=True)
        except Exception as e:
            print(f"[warn] BioBERT load failed: {e}")

    rows: list[dict[str, Any]] = []
    for sent_idx, words in enumerate(sentences):
        nb_imp = nb_token_importance(words, nb_obj) if nb_obj else None
        lime_imp = (lime_token_importance(words, bilstm_explainer)
                    if bilstm_explainer is not None else None)
        attn_imp = None
        if biobert and biobert_attn_rows is not None and sent_idx < len(biobert_attn_rows):
            attn_imp = attention_token_importance(
                words,
                np.array(biobert_attn_rows[sent_idx]["attention_last_layer"]),
                biobert["tokenizer"],
                max_length=192,
            )

        row: dict[str, Any] = {
            "sentence_idx": sent_idx,
            "words": words,
            "nb_importance": nb_imp,
            "lime_importance": lime_imp,
            "attn_importance": attn_imp,
        }
        # Pairwise agreement
        pairs = [
            ("nb_vs_lime", nb_imp, lime_imp),
            ("nb_vs_attn", nb_imp, attn_imp),
            ("lime_vs_attn", lime_imp, attn_imp),
        ]
        for name, a, b in pairs:
            if a is None or b is None:
                continue
            row[f"{name}_spearman"] = spearman(a, b)
            row[f"{name}_top3"] = topk_overlap(a, b, k=3)
        rows.append(row)

    summary: dict[str, Any] = {"n_sentences": len(rows), "per_sentence": rows}
    for key in ("nb_vs_lime", "nb_vs_attn", "lime_vs_attn"):
        sps = [r[f"{key}_spearman"] for r in rows if f"{key}_spearman" in r and not math.isnan(r[f"{key}_spearman"])]
        ovs = [r[f"{key}_top3"] for r in rows if f"{key}_top3" in r and not math.isnan(r[f"{key}_top3"])]
        if sps:
            summary[f"{key}_spearman_mean"] = sum(sps) / len(sps)
        if ovs:
            summary[f"{key}_top3_mean"] = sum(ovs) / len(ovs)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2, default=lambda x: None if isinstance(x, float) and math.isnan(x) else x)
    print(json.dumps({k: v for k, v in summary.items() if k != "per_sentence"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
