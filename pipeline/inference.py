"""Load champion models (+ the scaler each was trained with) and run
inference. Shared by the FastAPI backend so no prediction logic lives in
the HTTP layer itself.
"""

import mlflow
import mlflow.artifacts
import mlflow.pytorch
import pandas as pd
import torch
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from torch.utils.data import DataLoader

from pipeline.config import CHAMPION_ALIAS, MLFLOW_TRACKING_URI, REGISTRY_NAME
from qlstm_stock.data.dataset import SequenceDataset, Standardizer
from qlstm_stock.training import predict as run_predict

TARGET = "Close_lead1"
SEQUENCE_LENGTH = 3

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

_model_cache = {}
_scaler_cache = {}


class NoChampionError(RuntimeError):
    """Raised when a model has never been registered/promoted yet."""


def load_champion(model_name):
    if model_name in _model_cache:
        return _model_cache[model_name], _scaler_cache[model_name]

    registry_name = REGISTRY_NAME[model_name]
    try:
        version = MlflowClient().get_model_version_by_alias(registry_name, CHAMPION_ALIAS)
    except MlflowException as e:
        raise NoChampionError(f"No champion registered yet for {model_name!r}") from e

    model = mlflow.pytorch.load_model(f"models:/{registry_name}@{CHAMPION_ALIAS}")

    scaler_dict = mlflow.artifacts.load_dict(f"runs:/{version.run_id}/scaler.json")
    scaler = Standardizer()
    scaler.mean_ = pd.Series(scaler_dict["mean"])
    scaler.std_ = pd.Series(scaler_dict["std"])

    _model_cache[model_name] = model
    _scaler_cache[model_name] = scaler
    return model, scaler


def clear_cache():
    """Call after a retrain/promotion so the next request loads the new champion."""
    _model_cache.clear()
    _scaler_cache.clear()


def predict_next_close(model_name, df, features):
    """Predict tomorrow's close from the most recent SEQUENCE_LENGTH rows of
    `df` (including today's just-closed row, whose own Close_lead1 is
    unknown -- that's exactly the row we're predicting for)."""
    model, scaler = load_champion(model_name)

    recent = df[features].tail(SEQUENCE_LENGTH)
    recent_std = scaler.transform(recent)
    x = torch.tensor(recent_std[features].values).float().unsqueeze(0)

    model.eval()
    with torch.no_grad():
        pred_std = model(x).item()

    return pred_std * scaler.std_[TARGET] + scaler.mean_[TARGET]


def predict_history(model_name, df, features, days=90):
    """Actual vs. predicted close over the most recent `days` rows with a
    known target. Only pulls in the small extra window of history each
    prediction's sequence needs, not the whole dataset -- QLSTM inference
    through the Pennylane simulator is too slow to run over full history on
    every request.
    """
    model, scaler = load_champion(model_name)

    trainable = df.dropna(subset=[TARGET]).reset_index(drop=True)
    window = trainable.tail(days + SEQUENCE_LENGTH - 1).reset_index(drop=True)
    window_std = scaler.transform(window[features + [TARGET]])

    dataset = SequenceDataset(window_std, target=TARGET, features=features, sequence_length=SEQUENCE_LENGTH)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    preds_std = run_predict(loader, model).numpy()
    preds = preds_std * scaler.std_[TARGET] + scaler.mean_[TARGET]

    return {
        "dates": window["Date"].dt.strftime("%Y-%m-%d").tolist()[-days:],
        "actual": window[TARGET].tolist()[-days:],
        "predicted": preds.tolist()[-days:],
    }
