"""Walk-forward (rolling-origin) validation harness.

Replaces the original notebook's single chronological 67/33 split, which
reports exactly one out-of-sample score -- a number that depends heavily on
whatever market regime happened to fall in the final third of the series
and gives no sense of variance. This harness trains on an ever-growing
history and scores the model out-of-sample on several subsequent blocks, so
results are reported as mean +/- std across folds instead of a single
(possibly lucky or unlucky) number.
"""

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader

from qlstm_stock.data.dataset import SequenceDataset, Standardizer
from qlstm_stock.data.splits import walk_forward_splits
from qlstm_stock.evaluation.metrics import mae, rmse, summarize
from qlstm_stock.models.registry import build_model
from qlstm_stock.training.loop import fit, predict


@dataclass
class FoldResult:
    fold: int
    n_train: int
    n_test: int
    train_loss_history: list
    test_loss_history: list  # standardized-scale loss per epoch (index 0 = pre-training)
    rmse: float  # original price scale
    mae: float  # original price scale


def run_walk_forward_validation(
    df,
    target,
    features,
    model_name,
    model_kwargs=None,
    optimizer_cls=torch.optim.Adam,
    optimizer_kwargs=None,
    loss_function=None,
    sequence_length=3,
    batch_size=1,
    num_epochs=20,
    n_splits=5,
    min_train_fraction=0.5,
    gap=0,
    seed=101,
    verbose=False,
    on_fold_start=None,
    on_fold_end=None,
):
    """Run leakage-safe walk-forward validation for a single model.

    For every fold, standardization statistics are fit on that fold's
    training slice only (never on its test slice), and the model is
    (re)trained from scratch so later folds don't get an unfair head start
    from earlier folds' weights. Each fold yields a genuinely out-of-sample
    RMSE/MAE on the original price scale.

    `on_fold_start(fold_idx, n_train, n_test)` and `on_fold_end(FoldResult)`
    are optional progress hooks -- useful for callers (e.g. the live
    pipeline) that want fold-level progress output without the per-epoch
    detail `verbose` turns on for every fold.
    """
    model_kwargs = dict(model_kwargs or {})
    optimizer_kwargs = dict(optimizer_kwargs or {"lr": 1e-3})
    loss_function = loss_function or torch.nn.MSELoss()

    fold_results = []
    for wf_fold in walk_forward_splits(
        df, n_splits=n_splits, min_train_fraction=min_train_fraction, gap=gap
    ):
        if on_fold_start is not None:
            on_fold_start(wf_fold.fold, len(wf_fold.train), len(wf_fold.test))

        torch.manual_seed(seed)

        scaler = Standardizer().fit(wf_fold.train)
        train_std = scaler.transform(wf_fold.train)
        test_std = scaler.transform(wf_fold.test)

        train_dataset = SequenceDataset(
            train_std, target=target, features=features, sequence_length=sequence_length
        )
        test_dataset = SequenceDataset(
            test_std, target=target, features=features, sequence_length=sequence_length
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        eval_test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        model = build_model(model_name, num_sensors=len(features), **model_kwargs)
        optimizer = optimizer_cls(model.parameters(), **optimizer_kwargs)

        history = fit(
            train_loader, test_loader, model, loss_function, optimizer, num_epochs, verbose=verbose
        )

        predictions_std = predict(eval_test_loader, model).numpy()
        actual_std = test_dataset.y.numpy()

        predictions = predictions_std * scaler.std_[target] + scaler.mean_[target]
        actual = actual_std * scaler.std_[target] + scaler.mean_[target]

        result = FoldResult(
            fold=wf_fold.fold,
            n_train=len(wf_fold.train),
            n_test=len(wf_fold.test),
            train_loss_history=history["train_loss"],
            test_loss_history=history["test_loss"],
            rmse=rmse(actual, predictions),
            mae=mae(actual, predictions),
        )
        fold_results.append(result)
        if on_fold_end is not None:
            on_fold_end(result)
        if verbose:
            print(f"[{model_name}] fold {result.fold}: RMSE={result.rmse:.4f} MAE={result.mae:.4f}")

    return fold_results


def aggregate_fold_results(fold_results):
    """Mean/std RMSE and MAE across folds -- the headline numbers to report
    instead of a single train/test split's score."""
    return {
        "rmse": summarize([f.rmse for f in fold_results]),
        "mae": summarize([f.mae for f in fold_results]),
    }
