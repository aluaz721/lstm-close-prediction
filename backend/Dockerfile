FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY qlstm_stock/ qlstm_stock/
COPY pipeline/ pipeline/
COPY backend/ backend/

# Render mounts its one persistent disk here: mlflow.db (the registry),
# model artifacts (mlruns/), and data snapshots all live under data/.
VOLUME ["/app/data"]

EXPOSE 8000
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
