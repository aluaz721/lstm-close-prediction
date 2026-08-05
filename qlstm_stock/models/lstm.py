"""Classical LSTM regressor, ported from the original Factory.py."""

import torch

from qlstm_stock.models.base import SequenceRegressor


class ShallowRegressionLSTM(SequenceRegressor):
    def __init__(self, num_sensors, hidden_units):
        super().__init__(num_sensors, hidden_units)
        self.num_layers = 1

        self.lstm = torch.nn.LSTM(
            input_size=num_sensors,
            hidden_size=hidden_units,
            batch_first=True,
            num_layers=self.num_layers,
        )
        self.linear = torch.nn.Linear(in_features=self.hidden_units, out_features=1)

    def forward(self, x):
        batch_size = x.shape[0]
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).requires_grad_()
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).requires_grad_()

        _, (hn, _) = self.lstm(x, (h0, c0))
        out = self.linear(hn[0]).flatten()  # hn's first dim is num_layers (== 1 above)

        return out
