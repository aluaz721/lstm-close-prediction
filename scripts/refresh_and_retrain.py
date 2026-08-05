#!/usr/bin/env python3
"""Fetch live MRK data, snapshot it, and retrain + register both models in
MLflow. Run this by hand whenever you want to refresh the pipeline
(scheduled automation is deliberately not wired up yet -- see README).

Usage:
    python scripts/refresh_and_retrain.py
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.refresh import refresh_and_retrain


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start", default="2009-07-27", help="Earliest date to fetch.")
    parser.add_argument("--end", default=None, help="Latest date to fetch (default: today).")
    parser.add_argument("--n-splits", type=int, default=5)
    args = parser.parse_args()

    summary = refresh_and_retrain(start=args.start, end=args.end, n_splits=args.n_splits)
    print(f"Snapshot: {summary['snapshot_path']} ({summary['n_rows']} rows)")

    for name, result in summary["results"].items():
        wf = result["walk_forward"]
        outcome = "promoted to champion" if result["promoted"] else "did not beat champion"
        print(
            f"{name}: RMSE={wf['rmse']['mean']:.4f}+/-{wf['rmse']['std']:.4f} "
            f"MAE={wf['mae']['mean']:.4f}+/-{wf['mae']['std']:.4f} -> {outcome} "
            f"(run {result['run_id']}, v{result['registered_version']})"
        )


if __name__ == "__main__":
    main()
