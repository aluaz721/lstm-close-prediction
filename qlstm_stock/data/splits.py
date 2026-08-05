"""Walk-forward (rolling-origin) cross-validation splits for time series.

The original notebook used a single chronological 67/33 split, which reports
exactly one out-of-sample score. That number is noisy (it depends heavily on
which market regime happened to fall in the last third of the series) and
gives no sense of variance -- a common flaw in quick stock-prediction
notebooks that inflates apparent performance. Walk-forward validation trains
on an ever-growing history and evaluates on several subsequent, non-overlapping
blocks, so the model is scored out-of-sample multiple times across different
periods.
"""

from dataclasses import dataclass

import pandas as pd


@dataclass
class WalkForwardFold:
    fold: int
    train: pd.DataFrame
    test: pd.DataFrame


def walk_forward_splits(df: pd.DataFrame, n_splits=5, min_train_fraction=0.5, gap=0):
    """Generate expanding-window walk-forward folds over a chronologically
    ordered dataframe.

    The first `min_train_fraction` of rows seeds the initial training
    window. The remaining rows are divided into `n_splits` equal-sized test
    blocks. For fold i, the model trains on everything before that block
    (a window that grows fold over fold) and is evaluated only on that
    block, so no test-period row is ever visible during that fold's
    training. `gap` (in rows) can be used to leave a buffer between the end
    of training and the start of testing.
    """
    n = len(df)
    initial_train_end = int(n * min_train_fraction)
    remaining = n - initial_train_end - gap
    if n_splits < 1:
        raise ValueError("n_splits must be >= 1")
    if remaining < n_splits:
        raise ValueError(
            f"Not enough rows ({n}) for {n_splits} walk-forward splits "
            f"with min_train_fraction={min_train_fraction} and gap={gap}."
        )
    test_size = remaining // n_splits

    folds = []
    for i in range(n_splits):
        train_end = initial_train_end + i * test_size
        test_start = train_end + gap
        test_end = n if i == n_splits - 1 else test_start + test_size

        train = df.iloc[:train_end].copy()
        test = df.iloc[test_start:test_end].copy()
        folds.append(WalkForwardFold(fold=i, train=train, test=test))
    return folds
