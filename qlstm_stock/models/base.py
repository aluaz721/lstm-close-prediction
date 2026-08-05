"""Common interface shared by the classical LSTM and QLSTM regressors."""

from abc import ABC, abstractmethod

from torch import nn


class SequenceRegressor(nn.Module, ABC):
    """A model that maps a (batch, seq_len, num_features) window of past
    observations to a (batch,) vector of single-step-ahead predictions.

    Both `ShallowRegressionLSTM` and `QShallowRegressionLSTM` implement this
    interface, so training/evaluation code and scripts can swap one for the
    other without any special-casing.
    """

    def __init__(self, num_sensors: int, hidden_units: int):
        super().__init__()
        self.num_sensors = num_sensors
        self.hidden_units = hidden_units

    @abstractmethod
    def forward(self, x):
        ...
