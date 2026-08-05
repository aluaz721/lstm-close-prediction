"""FastAPI backend for the MRK close-price dashboard.

Thin HTTP layer only -- all business logic lives in `pipeline/` and
`qlstm_stock/`. Route handlers are plain `def` (not `async def`) since the
work underneath (pandas, torch, Pennylane, MLflow, yfinance) is all
blocking; FastAPI runs sync handlers in a threadpool so this doesn't block
the event loop.
"""

import os
import uuid

import mlflow
import pandas as pd
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from mlflow.tracking import MlflowClient

from pipeline.config import LATEST_SNAPSHOT_PATH, MLFLOW_EXPERIMENT_NAME, MODEL_NAMES, SNAPSHOT_DIR
from pipeline.inference import NoChampionError, clear_cache, predict_history, predict_next_close
from pipeline.refresh import refresh_and_retrain
from qlstm_stock.data.live import fetch_live_dataset, feature_columns, snapshot_dataset

app = FastAPI(title="MRK Close Prediction API")

_allowed_origins = os.environ.get("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

_jobs: dict[str, dict] = {}


@app.get("/health")
def health():
    return {"status": "ok"}


def _load_latest_df() -> pd.DataFrame:
    if not LATEST_SNAPSHOT_PATH.exists():
        raise HTTPException(404, "No data snapshot yet -- POST /api/refresh first.")
    df = pd.read_parquet(LATEST_SNAPSHOT_PATH)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


@app.get("/api/latest")
def get_latest():
    df = _load_latest_df()
    last = df.iloc[-1]
    return {
        "as_of_date": last["Date"].strftime("%Y-%m-%d"),
        "close": float(last["Close"]),
        "snapshot_fetched_at": pd.Timestamp(LATEST_SNAPSHOT_PATH.stat().st_mtime, unit="s").isoformat(),
    }


@app.get("/api/predictions")
def get_predictions():
    df = _load_latest_df()
    features = feature_columns(df)

    predictions = {}
    for name in MODEL_NAMES:
        try:
            predictions[name] = predict_next_close(name, df, features)
        except NoChampionError:
            predictions[name] = None

    return {"last_close": float(df.iloc[-1]["Close"]), "predictions": predictions}


@app.get("/api/history")
def get_history(days: int = 90):
    df = _load_latest_df()
    features = feature_columns(df)

    result = {}
    for name in MODEL_NAMES:
        try:
            result[name] = predict_history(name, df, features, days=days)
        except NoChampionError:
            result[name] = None

    return result


@app.get("/api/metrics")
def get_metrics():
    experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
    if experiment is None:
        return {name: None for name in MODEL_NAMES}

    result = {}
    for name in MODEL_NAMES:
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"tags.`mlflow.runName` = '{name}'",
            order_by=["start_time DESC"],
            max_results=1,
        )
        if runs.empty:
            result[name] = None
            continue
        row = runs.iloc[0]
        result[name] = {
            "rmse_mean": row.get("metrics.walk_forward_rmse_mean"),
            "rmse_std": row.get("metrics.walk_forward_rmse_std"),
            "mae_mean": row.get("metrics.walk_forward_mae_mean"),
            "mae_std": row.get("metrics.walk_forward_mae_std"),
            "run_id": row["run_id"],
            "start_time": str(row["start_time"]),
        }
    return result


@app.get("/api/runs")
def get_runs(limit: int = 20):
    experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
    if experiment is None:
        return []

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id], order_by=["start_time DESC"], max_results=limit
    )
    if runs.empty:
        return []

    columns = {
        "run_id": "run_id",
        "tags.mlflow.runName": "model",
        "start_time": "start_time",
        "metrics.walk_forward_rmse_mean": "rmse_mean",
        "metrics.walk_forward_mae_mean": "mae_mean",
    }
    available = [c for c in columns if c in runs.columns]
    table = runs[available].rename(columns={c: columns[c] for c in available})
    table["start_time"] = table["start_time"].astype(str)
    return table.fillna("").to_dict(orient="records")


@app.post("/api/refresh")
def post_refresh():
    df, _ = fetch_live_dataset()
    path = snapshot_dataset(df, snapshot_dir=SNAPSHOT_DIR)
    return {"snapshot_path": str(path), "n_rows": len(df)}


def _run_retrain_job(job_id: str):
    _jobs[job_id] = {"status": "running"}
    try:
        summary = refresh_and_retrain()
        clear_cache()  # next prediction request should pick up any newly promoted champion
        _jobs[job_id] = {"status": "done", "result": summary}
    except Exception as e:  # background task -- must record failure, not just vanish
        _jobs[job_id] = {"status": "error", "error": str(e)}


@app.post("/api/retrain")
def post_retrain(background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    _jobs[job_id] = {"status": "pending"}
    background_tasks.add_task(_run_retrain_job, job_id)
    return {"job_id": job_id}


@app.get("/api/retrain/status/{job_id}")
def get_retrain_status(job_id: str):
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "Unknown job id")
    return job
