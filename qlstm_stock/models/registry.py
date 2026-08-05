"""Factory for building a `SequenceRegressor` by name, so training and
evaluation code can swap the classical LSTM and QLSTM interchangeably."""

from qlstm_stock.models.lstm import ShallowRegressionLSTM
from qlstm_stock.models.qlstm import QShallowRegressionLSTM

MODEL_REGISTRY = {
    "lstm": ShallowRegressionLSTM,
    "qlstm": QShallowRegressionLSTM,
}


def build_model(name: str, **kwargs):
    try:
        model_cls = MODEL_REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"Unknown model '{name}'. Available models: {list(MODEL_REGISTRY)}"
        )
    return model_cls(**kwargs)
