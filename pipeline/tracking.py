"""MLflow-integrated training + registration for a single model.

Kept out of qlstm_stock so the core ML package stays tracking-backend
agnostic. Both the CLI (scripts/refresh_and_retrain.py) and the FastAPI
backend call `train_and_track` directly, so retraining logic lives in
exactly one place.
"""

import mlflow
import mlflow.pytorch
import torch
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from torch.utils.data import DataLoader

from pipeline.config import (
    CHAMPION_ALIAS,
    MLFLOW_ARTIFACT_ROOT,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    REGISTRY_NAME,
)
from qlstm_stock.data.dataset import SequenceDataset, Standardizer
from qlstm_stock.evaluation import aggregate_fold_results, run_walk_forward_validation
from qlstm_stock.models import build_model
from qlstm_stock.training import fit

TARGET = "Close_lead1"
SEQUENCE_LENGTH = 3
NUM_EPOCHS = 20
VALIDATION_TAIL = 60  # rows held out from the end for the deployable model's own val loss

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

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


def _ensure_experiment():
    experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
    if experiment is None:
        mlflow.create_experiment(
            MLFLOW_EXPERIMENT_NAME, artifact_location=f"file://{MLFLOW_ARTIFACT_ROOT}"
        )
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)


def train_and_track(model_name, df, features, n_splits=5, min_train_fraction=0.5):
    """Walk-forward-validate `model_name` on `df`, then train + register a
    deployable model, promoting it to the `champion` alias if it beats the
    currently registered champion's walk-forward RMSE (or if there is none
    yet). Returns a summary dict.
    """
    config = MODEL_CONFIGS[model_name]
    # Date isn't a model input and Standardizer/SequenceDataset assume it's
    # already gone (same convention as the notebooks) -- a raw Timestamp
    # column would also fail to JSON-serialize when the scaler is logged below.
    trainable = df.dropna(subset=[TARGET]).drop(columns=["Date"]).reset_index(drop=True)

    _ensure_experiment()
    with mlflow.start_run(run_name=model_name) as run:
        mlflow.log_params(
            {
                "model": model_name,
                "n_splits": n_splits,
                "min_train_fraction": min_train_fraction,
                "sequence_length": SEQUENCE_LENGTH,
                "num_epochs": NUM_EPOCHS,
                "n_rows": len(trainable),
                "optimizer": config["optimizer_cls"].__name__,
                **{f"model__{k}": v for k, v in config["model_kwargs"].items()},
                **{f"optimizer__{k}": v for k, v in config["optimizer_kwargs"].items()},
            }
        )

        fold_results = run_walk_forward_validation(
            trainable,
            target=TARGET,
            features=features,
            model_name=model_name,
            sequence_length=SEQUENCE_LENGTH,
            num_epochs=NUM_EPOCHS,
            n_splits=n_splits,
            min_train_fraction=min_train_fraction,
            **config,
        )
        for r in fold_results:
            mlflow.log_metric("fold_rmse", r.rmse, step=r.fold)
            mlflow.log_metric("fold_mae", r.mae, step=r.fold)

        summary = aggregate_fold_results(fold_results)
        mlflow.log_metrics(
            {
                "walk_forward_rmse_mean": summary["rmse"]["mean"],
                "walk_forward_rmse_std": summary["rmse"]["std"],
                "walk_forward_mae_mean": summary["mae"]["mean"],
                "walk_forward_mae_std": summary["mae"]["std"],
            }
        )

        model, scaler = _fit_deployable_model(model_name, config, trainable, features)

        # Persisted alongside the model so inference can standardize inputs
        # (and inverse-transform predictions) the exact same way training did.
        mlflow.log_dict(
            {"mean": scaler.mean_.to_dict(), "std": scaler.std_.to_dict()}, "scaler.json"
        )

        # Pin pickle serialization explicitly: newer MLflow defaults to the
        # 'pt2' traced-graph format, which needs an input_example and traces
        # model.forward via torch.export -- QLSTM's forward pass runs actual
        # quantum-circuit simulation through Pennylane's TorchLayer, which
        # isn't the kind of code torch.export tracing reliably handles.
        mlflow.pytorch.log_model(model, artifact_path="model", serialization_format="pickle")
        model_uri = f"runs:/{run.info.run_id}/model"
        registry_name = REGISTRY_NAME[model_name]
        registered = mlflow.register_model(model_uri, registry_name)

        client = MlflowClient()
        client.set_model_version_tag(
            registry_name, registered.version, "walk_forward_rmse_mean", str(summary["rmse"]["mean"])
        )
        promoted = _maybe_promote(client, registry_name, registered.version, summary["rmse"]["mean"])

    return {
        "model_name": model_name,
        "run_id": run.info.run_id,
        "walk_forward": summary,
        "registered_version": registered.version,
        "promoted": promoted,
    }


def _fit_deployable_model(model_name, config, trainable, features):
    val_start = len(trainable) - VALIDATION_TAIL
    train_df = trainable.iloc[:val_start].copy()
    val_df = trainable.iloc[val_start:].copy()

    scaler = Standardizer().fit(train_df)
    train_std = scaler.transform(train_df)
    val_std = scaler.transform(val_df)

    train_dataset = SequenceDataset(train_std, target=TARGET, features=features, sequence_length=SEQUENCE_LENGTH)
    val_dataset = SequenceDataset(val_std, target=TARGET, features=features, sequence_length=SEQUENCE_LENGTH)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    model = build_model(model_name, num_sensors=len(features), **config["model_kwargs"])
    optimizer = config["optimizer_cls"](model.parameters(), **config["optimizer_kwargs"])
    fit(train_loader, val_loader, model, torch.nn.MSELoss(), optimizer, NUM_EPOCHS, verbose=False)
    return model, scaler


def _maybe_promote(client, registry_name, new_version, new_rmse):
    try:
        champion = client.get_model_version_by_alias(registry_name, CHAMPION_ALIAS)
    except MlflowException:
        champion = None

    if champion is None:
        client.set_registered_model_alias(registry_name, CHAMPION_ALIAS, new_version)
        return True

    champion_rmse = float(champion.tags.get("walk_forward_rmse_mean", "inf"))
    if new_rmse < champion_rmse:
        client.set_registered_model_alias(registry_name, CHAMPION_ALIAS, new_version)
        return True
    return False
