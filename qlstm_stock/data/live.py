"""Live MRK + correlated-asset data ingestion via yfinance.

Kept as a standalone submodule (not re-exported from qlstm_stock.data's
__init__) since yfinance is a network-dependent, optional extra -- the rest
of qlstm_stock.data works fine against a static CSV without it.
"""

from datetime import date
from pathlib import Path

import pandas as pd
import yfinance as yf

from qlstm_stock.data.features import get_technical_indicators

# dataset column name -> yfinance symbol, matching example/Data_collection.ipynb
CORRELATED_TICKERS = {
    "AMGN": "AMGN",
    "BMY": "BMY",
    "GSK": "GSK",
    "JNJ": "JNJ",
    "LLY": "LLY",
    "NVS": "NVS",
    "PFE": "PFE",
    "SNY": "SNY",
    "VTRS": "VTRS",
    "VZ": "VZ",
    "SNP": "^GSPC",
    "NYSE": "^NYA",
    "NASDAQ": "^IXIC",
}

TARGET = "Close_lead1"


def _fetch_close(symbol, start, end):
    history = yf.Ticker(symbol).history(start=start, end=end)
    if history.index.tz is not None:
        history.index = history.index.tz_localize(None)
    return history["Close"].rename(symbol)


def fetch_live_dataset(start="2009-07-27", end=None):
    """Fetch MRK OHLCV plus correlated-asset closes via yfinance, and
    engineer the same technical-indicator feature set used throughout
    qlstm_stock.

    Returns (df, features). The last row's target (`Close_lead1`) is NaN by
    design -- that row is the one used for "predict tomorrow's close"
    inference; training/evaluation code should drop NaN-target rows.

    Correlated-asset closes are joined onto MRK's dates rather than
    positionally assigned (the original notebook's approach), so a ticker
    with a different set of trading days can't silently misalign the data.
    """
    mrk = yf.Ticker("MRK").history(start=start, end=end)
    if mrk.index.tz is not None:
        mrk.index = mrk.index.tz_localize(None)
    df = mrk[["Close", "Volume"]].copy()

    for column, symbol in CORRELATED_TICKERS.items():
        closes = _fetch_close(symbol, start, end).rename(column)
        df = df.join(closes, how="left")

    df = df.ffill().dropna()

    df = get_technical_indicators(df, "Close")
    df = df.iloc[20:].copy()  # drop rolling-window warmup rows (matches the original notebook)

    df[TARGET] = df["Close"].shift(-1)

    df = df.reset_index().rename(columns={"index": "Date"})
    return df, feature_columns(df)


def feature_columns(df):
    return list(df.columns.difference(["Close", TARGET, "Date"]))


def snapshot_dataset(df, snapshot_dir="data/snapshots"):
    """Write a timestamped parquet snapshot and refresh data/latest.parquet."""
    snapshot_dir = Path(snapshot_dir)
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    snapshot_path = snapshot_dir / f"mrk_{date.today():%Y%m%d}.parquet"
    df.to_parquet(snapshot_path, index=False)

    latest_path = snapshot_dir.parent / "latest.parquet"
    df.to_parquet(latest_path, index=False)

    return snapshot_path
