"""
Disagreement-of-saliency study driver (paper framing 1).

Pipeline:

  1. Load NCBI Disease, BC5CDR, and CoNLL-2003 test splits.
  2. For each (dataset, model in {bilstm-crf, bert-domain-appropriate}):
       a. Sample N test sentences.
       b. For each sentence, compute LIME, IG, and attention
          attributions for the model's predicted entity tokens.
       c. Compute pairwise disagreement metrics (saliency_metrics).
       d. Compute deletion / insertion AUC per method (faithfulness).
  3. Bucket sentences by phenomena (phenomena.py).
  4. Aggregate: report disagreement and faithfulness by
     (dataset, method-pair, bucket). Run paired-bootstrap significance
     tests on disagreement(biomed) - disagreement(general).

Outputs:
  - results/disagreement_long.csv  (one row per sentence/method-pair/metric)
  - results/disagreement_summary.csv  (mean by dataset/bucket)
  - results/faithfulness_long.csv

Requires the trained checkpoints (run main_NN.py and train_biobert.py
first), plus torch + captum + transformers + lime.

Usage:
    python run_disagreement_study.py \\
        --bilstm-weights ../models/bilstm_crf_seed42_*.h5 \\
        --biobert-dir ../models/biobert_ncbi \\
        --bert-dir ../models/bert_conll2003 \\
        --n-sentences 200 \\
        --out-dir ../results
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Any

from data_loader import load_ncbi_disease, load_bc5cdr, load_conll2003
from explainers.saliency_metrics import pairwise_table, aggregate
from explainers.phenomena import (
    per_sentence_features, bucketize, vocab_from_dataset,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+",
                   default=["ncbi", "bc5cdr", "conll2003"])
    p.add_argument("--bilstm-weights", default=None)
    p.add_argument("--biobert-dir", default=None)
    p.add_argument("--bert-dir", default=None)
    p.add_argument("--word2idx", default="../data/word2idx.pkl")
    p.add_argument("--tag2idx", default="../data/tag2idx.pkl")
    p.add_argument("--n-sentences", type=int, default=200)
    p.add_argument("--methods", nargs="+",
                   default=["lime", "ig", "attention"])
    p.add_argument("--out-dir", default="../results")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_dataset(name: str):
    if name == "ncbi":
        return load_ncbi_disease()
    if name == "bc5cdr":
        return load_bc5cdr()
    if name == "conll2003":
        return load_conll2003()
    raise ValueError(name)


def main() -> int:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Build a general-English reference vocab from CoNLL-2003 train (used
    # as the OOV reference for the phenomena bucketing).
    try:
        ref_vocab = vocab_from_dataset(load_conll2003()["train"].sentences)
    except FileNotFoundError:
        print("[warn] CoNLL-2003 not found; OOV-rate bucket will be empty.")
        ref_vocab = set()

    # Lazy imports of the heavy libs only if their methods are requested.
    explainer_funcs: dict[str, Any] = {}
    if "lime" in args.methods and args.bilstm_weights:
        explainer_funcs["lime"] = _build_lime_explainer(
            args.bilstm_weights, args.word2idx, args.tag2idx,
        )
    if "ig" in args.methods and (args.biobert_dir or args.bert_dir):
        explainer_funcs["ig"] = _build_ig_explainer(args.biobert_dir or args.bert_dir)
    if "attention" in args.methods and (args.biobert_dir or args.bert_dir):
        explainer_funcs["attention"] = _build_attention_explainer(
            args.biobert_dir or args.bert_dir,
        )

    if len(explainer_funcs) < 2:
        print("[error] Need at least two explanation methods to compute "
              "pairwise disagreement. Got:", list(explainer_funcs))
        return 2

    long_rows: list[dict] = []
    fa_rows: list[dict] = []

    for ds_name in args.datasets:
        try:
            splits = load_dataset(ds_name)
        except FileNotFoundError as e:
            print(f"[warn] skipping {ds_name}: {e}")
            continue
        test = splits["test"]
        sents = test.sentences[: args.n_sentences]
        tags = test.tags[: args.n_sentences]

        feats = per_sentence_features(sents, tags, vocab=ref_vocab)
        buckets = bucketize(feats)
        buckets_by_sent: dict[int, list[str]] = {}
        for bname, ids in buckets.items():
            for i in ids:
                buckets_by_sent.setdefault(i, []).append(bname)

        for sent_idx, (words, gold) in enumerate(zip(sents, tags)):
            target_idx = next((i for i, t in enumerate(gold) if t.startswith("B-")), None)
            if target_idx is None:
                continue
            attributions: dict[str, list[float]] = {}
            for mname, fn in explainer_funcs.items():
                try:
                    attributions[mname] = fn(words=words, target_word_idx=target_idx)
                except Exception as exc:
                    print(f"[warn] {mname} failed on {ds_name} sent {sent_idx}: {exc}")
            if len(attributions) < 2:
                continue
            rows = pairwise_table(attributions, sentence_id=sent_idx)
            for r in rows:
                r["dataset"] = ds_name
                r["buckets"] = ",".join(buckets_by_sent.get(sent_idx, []))
                long_rows.append(r)

    _write_csv(os.path.join(args.out_dir, "disagreement_long.csv"), long_rows)

    summary = aggregate(long_rows)
    summary_rows = [
        {"method_a": ma, "method_b": mb, "metric": m, "k": k, "mean_value": v}
        for (ma, mb, m, k), v in summary.items()
    ]
    _write_csv(os.path.join(args.out_dir, "disagreement_summary.csv"), summary_rows)

    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2, default=str)
    print(f"Wrote {len(long_rows)} rows to {args.out_dir}/disagreement_long.csv")
    return 0


def _write_csv(path: str, rows: list[dict]) -> None:
    if not rows:
        open(path, "w").close()
        return
    keys = sorted({k for r in rows for k in r.keys()})
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


# ---------------------------------------------------------------------------
# Explainer factories: each returns a callable
# `fn(words, target_word_idx) -> list[float]` of length len(words).
# ---------------------------------------------------------------------------

def _build_lime_explainer(weights_glob: str, word2idx_path: str, tag2idx_path: str):
    import glob, pickle
    from tensorflow import keras
    from Explainer import NERExplainerGenerator
    from eli5.lime import TextExplainer
    from eli5.lime.samplers import MaskingTextSampler
    import eli5

    paths = sorted(glob.glob(weights_glob))
    if not paths:
        raise FileNotFoundError(weights_glob)
    model = keras.models.load_model(paths[-1], compile=False)
    with open(word2idx_path, "rb") as f:
        w2i = pickle.load(f)
    with open(tag2idx_path, "rb") as f:
        t2i = pickle.load(f)
    explainer = NERExplainerGenerator(model, w2i, t2i, max_len=114)

    def fn(*, words, target_word_idx):
        sentence = " ".join(words)
        predict_func = explainer.get_predict_function(word_index=target_word_idx)
        sampler = MaskingTextSampler(replacement="UNK", max_replace=0.7,
                                     token_pattern=None, bow=False)
        te = TextExplainer(sampler=sampler, position_dependent=True, random_state=42)
        te.fit(sentence, predict_func)
        d = eli5.formatters.format_as_dict(
            te.explain_prediction(target_names=list(explainer.idx2tag.values()),
                                   top_targets=3))
        scores = [0.0] * len(words)
        for tgt in d["targets"]:
            for fw in tgt["feature_weights"]["pos"]:
                tok = fw["feature"].split(maxsplit=1)
                if len(tok) == 2:
                    try:
                        idx = words.index(tok[1])
                        scores[idx] = max(scores[idx], float(fw["weight"]))
                    except ValueError:
                        pass
        return scores
    return fn


def _build_ig_explainer(model_dir: str):
    from transformers import AutoTokenizer, AutoModelForTokenClassification
    from explainers.integrated_gradients import integrated_gradients_word_attribution
    tok = AutoTokenizer.from_pretrained(model_dir)
    mdl = AutoModelForTokenClassification.from_pretrained(model_dir)

    def fn(*, words, target_word_idx):
        return integrated_gradients_word_attribution(
            mdl, tok, words, target_word_idx,
        )
    return fn


def _build_attention_explainer(model_dir: str):
    import torch
    from transformers import AutoTokenizer, AutoModelForTokenClassification
    tok = AutoTokenizer.from_pretrained(model_dir)
    mdl = AutoModelForTokenClassification.from_pretrained(model_dir)
    mdl.eval()

    def fn(*, words, target_word_idx):
        enc = tok(list(words), is_split_into_words=True, truncation=True,
                  padding="max_length", max_length=192, return_tensors="pt")
        word_ids = enc.word_ids()
        with torch.no_grad():
            out = mdl(**{k: v for k, v in enc.items()}, output_attentions=True)
        attn = out.attentions[-1].mean(dim=1).squeeze(0).cpu().numpy()
        target_pos = next((i for i, w in enumerate(word_ids) if w == target_word_idx), None)
        if target_pos is None:
            return [0.0] * len(words)
        per_word = [0.0] * len(words)
        counts = [0] * len(words)
        for tok_pos, wid in enumerate(word_ids):
            if wid is None or wid >= len(words):
                continue
            per_word[wid] += float(attn[target_pos, tok_pos])
            counts[wid] += 1
        return [s / max(c, 1) for s, c in zip(per_word, counts)]
    return fn


if __name__ == "__main__":
    sys.exit(main())
