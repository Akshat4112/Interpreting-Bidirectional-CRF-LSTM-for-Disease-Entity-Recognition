"""
BioBERT / PubMedBERT fine-tuning for disease NER.

Token-classification fine-tuning via HuggingFace Transformers, with
seqeval entity-level metrics computed on the held-out test split. This
is the *modern* baseline missing from the original repo (Naive Bayes
vs. BiLSTM is not a 2026 baseline; PubMedBERT is).

Defaults to `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext`
(strong on biomedical text, abstract+fulltext variant). To use BioBERT
v1.1 instead pass `--model dmis-lab/biobert-v1.1`.

Saves predictions per token so the interpretability comparison module
can run attention-based attribution against the same checkpoints.

Requires: transformers, datasets, torch, seqeval. Will not run in
environments without GPU + HuggingFace Hub access.

Usage:
    python train_biobert.py \\
        --train ../data/ner-disease/train.iob \\
        --dev   ../data/ner-disease/dev.iob \\
        --test  ../data/ner-disease/test.iob \\
        --model microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext \\
        --epochs 4 --batch-size 16 --seed 42
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

from data_loader import load_iob, NERDataset
from Evaluation import evaluate


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True)
    p.add_argument("--dev", default=None)
    p.add_argument("--test", required=True)
    p.add_argument(
        "--model",
        default="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
    )
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--learning-rate", type=float, default=3e-5)
    p.add_argument("--max-length", type=int, default=192)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default="../models/biobert_ncbi")
    p.add_argument("--save-attention", action="store_true",
                   help="Cache last-layer attention on the test set for the "
                        "interpretability comparison.")
    return p.parse_args()


def encode_split(ds: NERDataset, tokenizer, label2id, max_length: int):
    import torch
    from torch.utils.data import Dataset

    class TaggedDataset(Dataset):
        def __init__(self, sents, tags):
            self.sents = sents
            self.tags = tags

        def __len__(self):
            return len(self.sents)

        def __getitem__(self, idx):
            words = self.sents[idx]
            labels = self.tags[idx]
            enc = tokenizer(
                words,
                is_split_into_words=True,
                return_offsets_mapping=False,
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            word_ids = enc.word_ids()
            aligned = []
            prev_word = None
            for wid in word_ids:
                if wid is None:
                    aligned.append(-100)
                elif wid != prev_word:
                    aligned.append(label2id[labels[wid]])
                else:
                    # subsequent subword tokens of same word: same label, but
                    # convert B- to I- so seqeval-side decoding stays consistent
                    lab = labels[wid]
                    if lab.startswith("B-"):
                        lab = "I-" + lab.split("-", 1)[1]
                    aligned.append(label2id[lab])
                prev_word = wid
            item = {k: v.squeeze(0) for k, v in enc.items()}
            item["labels"] = torch.tensor(aligned, dtype=torch.long)
            return item

    return TaggedDataset(ds.sentences, ds.tags)


def predict_to_word_tags(
    trainer,
    eval_dataset,
    raw_ds: NERDataset,
    tokenizer,
    id2label: dict[int, str],
) -> list[list[str]]:
    """Map sub-word predictions back to per-word tags (one tag per
    original word, taken from the first sub-token). Required for
    seqeval, since IOB tags are defined at the word level."""
    import numpy as np

    out = trainer.predict(eval_dataset)
    pred_ids = np.argmax(out.predictions, axis=-1)
    word_tags: list[list[str]] = []
    for sent_idx, words in enumerate(raw_ds.sentences):
        enc = tokenizer(words, is_split_into_words=True, truncation=True,
                        max_length=eval_dataset[0]["input_ids"].shape[0],
                        padding="max_length")
        word_ids = enc.word_ids()
        seen = set()
        per_word: list[str] = ["O"] * len(words)
        for tok_pos, wid in enumerate(word_ids):
            if wid is None or wid in seen or wid >= len(words):
                continue
            seen.add(wid)
            per_word[wid] = id2label[int(pred_ids[sent_idx][tok_pos])]
        word_tags.append(per_word)
    return word_tags


def main() -> None:
    args = parse_args()

    import torch  # noqa: F401  (sanity import; clearer error if missing)
    from transformers import (
        AutoTokenizer,
        AutoModelForTokenClassification,
        TrainingArguments,
        Trainer,
        DataCollatorForTokenClassification,
        set_seed,
    )

    set_seed(args.seed)

    train_ds = load_iob(args.train, name="train")
    dev_ds = load_iob(args.dev, name="dev") if args.dev else None
    test_ds = load_iob(args.test, name="test")

    labels = sorted({t for ts in train_ds.tags for t in ts})
    if "O" in labels:
        labels = ["O"] + [l for l in labels if l != "O"]
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(
        args.model, num_labels=len(labels), id2label=id2label, label2id=label2id,
    )

    train_enc = encode_split(train_ds, tokenizer, label2id, args.max_length)
    dev_enc = encode_split(dev_ds, tokenizer, label2id, args.max_length) if dev_ds else None
    test_enc = encode_split(test_ds, tokenizer, label2id, args.max_length)

    targs = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        eval_strategy="epoch" if dev_enc is not None else "no",
        save_strategy="epoch",
        logging_steps=50,
        seed=args.seed,
        report_to=[],
        load_best_model_at_end=dev_enc is not None,
        metric_for_best_model="eval_loss" if dev_enc is not None else None,
        output_attentions=False,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_enc,
        eval_dataset=dev_enc,
        tokenizer=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer),
    )
    trainer.train()

    word_pred = predict_to_word_tags(trainer, test_enc, test_ds, tokenizer, id2label)
    metrics = evaluate(test_ds.tags, word_pred, print_report=True)

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(
            {
                "model": args.model,
                "seed": args.seed,
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
            },
            f, indent=2,
        )
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    if args.save_attention:
        _dump_attention(model, tokenizer, test_ds, args.output_dir, args.max_length)


def _dump_attention(model, tokenizer, test_ds: NERDataset, out_dir: str, max_length: int) -> None:
    """Save mean last-layer attention per (sentence, predicted-token, source-token).

    Used by interpretability_compare.py to compare attention-based
    explanations against LIME and NB feature weights.
    """
    import numpy as np
    import torch

    model.eval()
    device = next(model.parameters()).device
    rows: list[dict[str, Any]] = []
    for sent_idx, words in enumerate(test_ds.sentences):
        enc = tokenizer(
            words, is_split_into_words=True, truncation=True,
            max_length=max_length, padding="max_length", return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc, output_attentions=True)
        # last layer, mean over heads: (1, seq, seq)
        attn = out.attentions[-1].mean(dim=1).squeeze(0).cpu().numpy()
        rows.append({"sentence_idx": sent_idx, "attention_last_layer": attn.tolist()})
    np.save(os.path.join(out_dir, "test_attention.npy"), np.array(rows, dtype=object))


if __name__ == "__main__":
    main()
