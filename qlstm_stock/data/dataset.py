"""Windowed sequence dataset and leakage-safe standardization.

Ported from the original Factory.py, with standardization pulled out into an
explicit fit/transform object so each walk-forward fold can be standardized
using only statistics from that fold's own training slice.
"""

import torch
from torch.utils.data import Dataset


class SequenceDataset(Dataset):
    def __init__(self, dataframe, target, features, sequence_length=5):
        self.features = features
        self.target = target
        self.sequence_length = sequence_length
        self.y = torch.tensor(dataframe[self.target].values).float()
        self.X = torch.tensor(dataframe[self.features].values).float()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        if i >= self.sequence_length - 1:
            i_start = i - self.sequence_length + 1
            x = self.X[i_start:(i + 1), :]
        else:
            padding = self.X[0].repeat(self.sequence_length - i - 1, 1)
            x = self.X[0:(i + 1), :]
            x = torch.cat((padding, x), 0)

        return x, self.y[i]


class Standardizer:
    """Per-column z-score standardization fit on a training slice only.

    Fitting must always happen on the training portion of a given
    walk-forward fold, never on validation/test data, to avoid leaking
    future distribution information into the past.
    """

    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, df):
        self.mean_ = df.mean()
        self.std_ = df.std()
        return self

    def transform(self, df):
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Standardizer must be fit before transform.")
        return (df - self.mean_) / self.std_

    def fit_transform(self, df):
        return self.fit(df).transform(df)

    def inverse_transform_column(self, series, column):
        return series * self.std_[column] + self.mean_[column]
