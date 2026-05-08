"""
Run a model across multiple seeds and report mean +/- std.

Reviewers expect this on every metric in 2026; the original repo
reported a single number from one run.

Example:
    python multi_seed_runner.py --model bilstm --seeds 41 42 43 --epochs 20
    python multi_seed_runner.py --model bilstm_crf --seeds 41 42 43 --epochs 20
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

from data_loader import load_iob


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["bilstm", "bilstm_crf"], default="bilstm")
    p.add_argument("--train", default="../data/ner-disease/train.iob")
    p.add_argument("--test", default="../data/ner-disease/test.iob")
    p.add_argument("--seeds", type=int, nargs="+", default=[41, 42, 43])
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--out", default="../models/multiseed_results.json")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    from Train import train_and_eval  # local import (TF heavy)

    train_ds = load_iob(args.train)
    test_ds = load_iob(args.test)

    runs: list[dict] = []
    for s in args.seeds:
        m = train_and_eval(
            train_ds, test_ds,
            model_kind=args.model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            seed=s,
        )
        runs.append({"seed": s, "precision": m["precision"], "recall": m["recall"], "f1": m["f1"]})

    summary = {
        "model": args.model,
        "n_seeds": len(args.seeds),
        "seeds": args.seeds,
        "runs": runs,
    }
    for k in ("precision", "recall", "f1"):
        vals = [r[k] for r in runs]
        summary[f"{k}_mean"] = statistics.fmean(vals)
        summary[f"{k}_std"] = statistics.pstdev(vals) if len(vals) > 1 else 0.0

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== {args.model}  ({len(args.seeds)} seeds) ===")
    print(f"precision: {summary['precision_mean']:.4f} +/- {summary['precision_std']:.4f}")
    print(f"recall   : {summary['recall_mean']:.4f} +/- {summary['recall_std']:.4f}")
    print(f"f1       : {summary['f1_mean']:.4f} +/- {summary['f1_std']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
