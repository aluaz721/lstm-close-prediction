#!/usr/bin/env python3
"""Compare the classical LSTM and QLSTM under walk-forward validation.

This replaces the original notebook's single 67/33 train/test split (see
`Stock Prediction Draft 3.ipynb`) with rolling-origin walk-forward
validation: the model is retrained from scratch and scored out-of-sample on
several successive blocks of the series, so the reported RMSE/MAE is a
mean +/- std across folds rather than one (possibly lucky) split.

Usage:
    python scripts/run_walk_forward_comparison.py
    python scripts/run_walk_forward_comparison.py --n-splits 5 --num-epochs 20
    python scripts/run_walk_forward_comparison.py --models lstm --output results.json
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qlstm_stock.evaluation import aggregate_fold_results, run_walk_forward_validation

DATA_PATH = Path(__file__).resolve().parent.parent / "dataset_MRK_prediction.csv"
TARGET = "Close_lead1"

# Hyperparameters as chosen (by experimentation) in the original notebook.
MODEL_CONFIGS = {
    "lstm": dict(
        model_kwargs={"hidden_units": 16},
        optimizer_cls=torch.optim.Adam,
        optimizer_kwargs={"lr": 0.0001},
    ),
    "qlstm": dict(
        model_kwargs={"hidden_units": 16, "n_qubits": 4},
        optimizer_cls=torch.optim.Adagrad,
        optimizer_kwargs={"lr": 0.05},
    ),
}


def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df.drop(["Date", "Unnamed: 0"], axis=1)
    features = list(df.columns.difference(["Close", TARGET]))
    return df, features


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=["lstm", "qlstm"], choices=["lstm", "qlstm"])
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--min-train-fraction", type=float, default=0.5)
    parser.add_argument("--gap", type=int, default=0)
    parser.add_argument("--sequence-length", type=int, default=3)
    parser.add_argument("--num-epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--output", type=Path, default=None, help="Optional path to dump results as JSON.")
    args = parser.parse_args()

    df, features = load_data()
    print(f"Loaded {len(df)} rows, {len(features)} features, target={TARGET!r}")

    all_results = {}
    for model_name in args.models:
        config = MODEL_CONFIGS[model_name]
        print(f"\n=== {model_name} ===")
        fold_results = run_walk_forward_validation(
            df,
            target=TARGET,
            features=features,
            model_name=model_name,
            model_kwargs=config["model_kwargs"],
            optimizer_cls=config["optimizer_cls"],
            optimizer_kwargs=config["optimizer_kwargs"],
            sequence_length=args.sequence_length,
            num_epochs=args.num_epochs,
            n_splits=args.n_splits,
            min_train_fraction=args.min_train_fraction,
            gap=args.gap,
            seed=args.seed,
            verbose=args.verbose,
        )
        for r in fold_results:
            print(f"  fold {r.fold}: n_train={r.n_train} n_test={r.n_test} RMSE={r.rmse:.4f} MAE={r.mae:.4f}")

        summary = aggregate_fold_results(fold_results)
        print(
            f"  -> RMSE mean={summary['rmse']['mean']:.4f} std={summary['rmse']['std']:.4f} | "
            f"MAE mean={summary['mae']['mean']:.4f} std={summary['mae']['std']:.4f}"
        )

        all_results[model_name] = {
            "folds": [vars(r) for r in fold_results],
            "summary": summary,
        }

    if args.output:
        args.output.write_text(json.dumps(all_results, indent=2))
        print(f"\nWrote results to {args.output}")


if __name__ == "__main__":
    main()
