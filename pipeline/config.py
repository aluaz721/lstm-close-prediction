"""Shared configuration for the pipeline package.

Explicit, absolute paths throughout -- both the CLI script and the FastAPI
backend need to hit the same MLflow store and data files regardless of
their working directory when invoked.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = REPO_ROOT / "data"
SNAPSHOT_DIR = DATA_DIR / "snapshots"
LATEST_SNAPSHOT_PATH = DATA_DIR / "latest.parquet"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# mlflow.db AND mlruns/ (artifacts) both live under DATA_DIR, not REPO_ROOT.
# Render attaches exactly one persistent disk per service -- so everything
# that needs to survive a redeploy (the registry DB, which model version is
# `champion`, model artifact files, data snapshots) has to sit under a
# single mount point. See backend/Dockerfile.
MLFLOW_TRACKING_URI = f"sqlite:///{DATA_DIR / 'mlflow.db'}"
MLFLOW_ARTIFACT_ROOT = DATA_DIR / "mlruns"
MLFLOW_EXPERIMENT_NAME = "mrk-close-prediction"

MODEL_NAMES = ["lstm", "qlstm"]
REGISTRY_NAME = {"lstm": "mrk-close-lstm", "qlstm": "mrk-close-qlstm"}
CHAMPION_ALIAS = "champion"
