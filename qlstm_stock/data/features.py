"""Technical-indicator feature engineering, ported from the original helper.py."""

import datetime

import numpy as np


def date_parser(x):
    return datetime.datetime.strptime(x, "%Y-%m-%d")


def get_technical_indicators(dataset, target_col):
    # 7- and 21-day moving average
    dataset["ma7"] = dataset[target_col].rolling(window=7).mean()
    dataset["ma21"] = dataset[target_col].rolling(window=21).mean()

    # MACD: exponentially weighted moving averages
    dataset["26ema"] = dataset[target_col].ewm(span=26).mean()
    dataset["12ema"] = dataset[target_col].ewm(span=12).mean()
    dataset["MACD"] = dataset["12ema"] - dataset["26ema"]

    # Bollinger Bands
    dataset["20sd"] = dataset[target_col].rolling(20).std()
    dataset["upper_band"] = dataset["ma21"] + dataset["20sd"] * 2
    dataset["lower_band"] = dataset["ma21"] - dataset["20sd"] * 2

    # Exponential moving average
    dataset["ema"] = dataset[target_col].ewm(com=0.5).mean()

    # Momentum
    dataset["momentum"] = dataset[target_col] - 1
    dataset["log_momentum"] = np.log(dataset[target_col] - 1)
    return dataset
