"""Orchestrates a full refresh: fetch live data, snapshot it, and retrain +
register both models."""

from qlstm_stock.data.live import fetch_live_dataset, snapshot_dataset

from pipeline.config import MODEL_NAMES, SNAPSHOT_DIR
from pipeline.tracking import train_and_track


def refresh_and_retrain(start="2009-07-27", end=None, n_splits=5):
    df, features = fetch_live_dataset(start=start, end=end)
    snapshot_path = snapshot_dataset(df, snapshot_dir=SNAPSHOT_DIR)

    results = {
        name: train_and_track(name, df, features, n_splits=n_splits) for name in MODEL_NAMES
    }

    return {
        "snapshot_path": str(snapshot_path),
        "n_rows": len(df),
        "results": results,
    }
