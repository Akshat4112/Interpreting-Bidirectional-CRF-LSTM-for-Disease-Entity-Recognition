"""
Disease NER training entry point.

This is now a CLI wrapper. The previous version trained on the train
split, validated on a 20% slice of training, never touched the official
NCBI Disease test set, and reported Keras `accuracy` as if it were F1.
The new version:

  - reads NCBI Disease (or BC5CDR) via `data_loader`
  - trains BiLSTM or BiLSTM-CRF (`--model {bilstm,bilstm_crf}`)
  - evaluates on the held-out test.iob with seqeval entity-level metrics

Usage:
    python main_NN.py --model bilstm     --epochs 20 --seed 42
    python main_NN.py --model bilstm_crf --epochs 20 --seed 42
    python main_NN.py --model bilstm     --dataset bc5cdr --epochs 20

For BioBERT / PubMedBERT, use `train_biobert.py` (separate to keep the
TF and PyTorch deps independent).

For multi-seed runs:
    python multi_seed_runner.py --model bilstm_crf --seeds 41 42 43
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["bilstm", "bilstm_crf"], default="bilstm")
    p.add_argument("--dataset", choices=["ncbi", "bc5cdr"], default="ncbi")
    p.add_argument("--data-root", default=None,
                   help="Override default data root for the chosen dataset.")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="../models/last_run_metrics.json")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    from data_loader import load_iob, load_ncbi_disease, load_bc5cdr
    from Train import train_and_eval

    if args.dataset == "ncbi":
        root = args.data_root or "../data/ner-disease"
        splits = load_ncbi_disease(root)
    else:
        root = args.data_root or "../data/bc5cdr"
        splits = load_bc5cdr(root)

    print(f"Loaded {args.dataset}: train={len(splits['train'])} sents, "
          f"test={len(splits['test'])} sents, "
          f"labels={splits['train'].label_set()}")

    metrics = train_and_eval(
        splits["train"], splits["test"],
        model_kind=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(
            {k: metrics[k] for k in ("model_kind", "seed", "precision", "recall", "f1", "weights_path")},
            f, indent=2,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
