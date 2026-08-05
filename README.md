# SoftServe_QLSTM
## Introduction

Stock price prediction is one of the most rewarding problems in modern finance, where the accurate forecasting of future stock prices can yield significant profit and reduce the risks. LSTM (Long Short-Term Memory) is a recurrent Neural Network (RNN) applicable to a broad range of problems aiming to analyze or classify sequential data. Therefore, many people have used LSTM to predict the future stock price based on the historical data sequences with great success.

On the other hand, recent studies have shown that the LSTM's efficiency and trainability can be improved by replacing some of the layers in the LSTM with variational quantum layers, thus making the classical LSTM a quantum-classical hybrid model, which we will call QLSTM for Quantum LSTM. A recent study done by Samuel Yen-Chi Chen, Shinjae Yoo, and Yao-Lung L. Fang shows that QLSTM offers better trainability compared to its classical counterpart as it proved to learn significantly more information after the first training epoch than its classical counterpart and learned the local features better, all while having a comparable number of parameters. Inspired by these recent results, we proceed to test this variational quantum-classical hybrid neural network technique on stock price predictions.

In this submission, we provide a proof of concept that QLSTM can be used to predict stock prices on a particular stock (Merck and Co. Inc (MRK)), and that the results of its prediction is comparable, and perhaps even arguably better in terms of loss, to its classical counter part. We demonstrate that it has a higher trainability as the loss decreases faster with the QLSTM per epoch, and that the results were achieved using much less parameters in QLSTM as compared to the classical LSTM.

## Submission

For the implementation of QLSTM and its comparison to the classical LSTM, please refer to the notebook <b> example/Stock Prediction Draft 3 </b>. It provides a proof of concept that QLSTM can be used to perform stock price predictions by training the model to predict the stock prices of Merck and Co. Inc (MRK), and that it has comparable results for its prediction to its classical counterpart while requiring much fewer parameters. Furthermore, we show that trainability of QLSTM is arguably better than LSTM.

For more in depth view of our data collection process, please refer to the notebook <b> example/Data Collection </b>. It lists our data collection decisions that provided us a relevant csv file to train the QLSTM and LSTM for the above submission.

Disclaimer: As of right now, we are testing QLSTM on the Pennylane simulator, but the technique is technology agnostic and can work on any gate based device, be it IBM's Qiskit or AWS Braket's gate based devices. We are planning to test its viability on NISQ era devices soon. 

## Outline of GitHub

- <b> qlstm_stock/ </b>: Importable package with the LSTM/QLSTM models, data pipeline (including live yfinance ingestion), training loop, and walk-forward evaluation harness (see below). This is the source of truth for all model, data-processing, and evaluation code.
- <b> pipeline/ </b>: MLflow-integrated orchestration on top of `qlstm_stock` — training + registration (`tracking.py`), the fetch→train→register flow (`refresh.py`), and inference for the currently deployed models (`inference.py`). Shared by the CLI script and the FastAPI backend so retraining logic lives in exactly one place.
- <b> backend/ </b>: FastAPI app serving live predictions, history, walk-forward metrics, and MLflow run data to the dashboard, plus refresh/retrain endpoints. See "Live pipeline" below.
- <b> frontend/ </b>: Next.js dashboard (TypeScript + Tailwind + Recharts) showing current MRK price, next-close predictions from both models, actual-vs-predicted history, and the walk-forward comparison.
- <b> example/Stock Prediction Draft 3.ipynb </b>: Walks through training and comparing the classical LSTM and QLSTM on MRK stock prices, and the complexity analysis, using the `qlstm_stock` package rather than duplicating that logic locally.
- <b> example/Data_collection.ipynb </b>: Describes the data collection process (technical indicators, Fourier transforms, ARIMA, correlated assets, sentiment analysis) and the relevant decisions made during that process, producing `dataset_MRK_prediction.csv`.
- <b> scripts/run_walk_forward_comparison.py </b>: Runs the LSTM vs. QLSTM comparison on the static historical CSV, non-interactively, with walk-forward validation.
- <b> scripts/refresh_and_retrain.py </b>: Runs the *live* pipeline by hand — fetch current MRK data, walk-forward validate, retrain, and register/promote both models in MLflow.
- <b> dataset_MRK_prediction.csv </b>: Main csv used for <b> example/Stock Prediction Draft 3.ipynb </b>
- <b> MRK.csv </b>: MRK data, used in <b> example/Data_collection.ipynb </b>
- Other files are just images used in descriptions in <b> example/Stock Prediction Draft 3.ipynb </b>

## `qlstm_stock` package

The original hackathon notebook's logic has been refactored into the importable package `qlstm_stock/`, so the LSTM and QLSTM can be trained, evaluated, and compared as a script or from a notebook without duplicating model/data/training code in either place:

- `qlstm_stock/data/` — feature engineering (`features.py`), the windowed `SequenceDataset` and a leakage-safe `Standardizer` (`dataset.py`), and walk-forward split generation (`splits.py`).
- `qlstm_stock/models/` — `ShallowRegressionLSTM` and `QShallowRegressionLSTM` behind a shared `SequenceRegressor` interface (`base.py`), plus a `build_model(name, ...)` registry (`registry.py`) so a model is chosen by string name.
- `qlstm_stock/training/` — the train/test-epoch loop and a `fit()` helper.
- `qlstm_stock/evaluation/` — `run_walk_forward_validation(...)`, metrics (RMSE/MAE), and aggregation across folds.

Run the LSTM vs. QLSTM comparison with:

```bash
pip install -r requirements.txt
python scripts/run_walk_forward_comparison.py
```

### Evaluation methodology caveat (fixed)

The original hackathon notebook split the series with:

```python
df_train = df.loc[:size].copy()
df_test = df.loc[size:].copy()
```

Because `.loc` slicing is inclusive on both ends, the row at index `size` appeared in **both** `df_train` and `df_test` — the model was trained and tested on that row. It also reported a single out-of-sample score from one chronological split, which for financial time series is noisy: performance depends heavily on which market regime happened to land in the final third of the series.

`example/Stock Prediction Draft 3.ipynb` now uses position-based (`.iloc`) slicing, which drops the shared row, for its main walkthrough. More importantly, its new *Walk-forward validation* section (and `qlstm_stock.evaluation.run_walk_forward_validation` / `scripts/run_walk_forward_comparison.py` generally) retrains each model from scratch and scores it out-of-sample across `n_splits` expanding-window folds (`qlstm_stock/data/splits.py`), reporting RMSE/MAE as mean ± std across folds instead of a single number. Standardization statistics are refit per fold on that fold's training slice only, so no fold's test-period distribution leaks into training.

## Live pipeline: MLflow + dashboard

On top of the static-CSV package above, there's a live pipeline that fetches current MRK data, tracks experiments in MLflow, and serves predictions through a dashboard.

**Live feature set.** `qlstm_stock/data/live.py` fetches MRK OHLCV plus 13 correlated tickers/indices via `yfinance`, joined on date (not the original notebook's positional list-assignment, which silently breaks if a ticker has different trading days), and applies the same `get_technical_indicators`. The original dataset's Fourier transform (FT3/FT6/FT9), ARIMA, and sentiment features are deliberately **not** replicated live: the Fourier features are a single FFT over the *entire* historical series (so early rows leak information from future closes — a real methodology flaw, not just an inconvenience), ARIMA(5,1,0) refits from scratch at every day of the test period (too slow to run in a retraining loop), and sentiment needs FinBERT run over a live MRK headline feed that doesn't exist (the original used a static, uncommitted Kaggle CSV). The live feature set is technical indicators + correlated-asset closes only — both fully causal and cheap to compute.

**MLflow.** `pipeline/tracking.py` walk-forward validates each model, logs params/fold metrics/aggregate RMSE-MAE to MLflow, trains a deployable model on the full live dataset, and registers it under `mrk-close-lstm` / `mrk-close-qlstm`. A version is only promoted to the `champion` alias if its walk-forward RMSE beats the current champion's (or there is none yet) — a bad retrain can never silently replace a working model. The fitted `Standardizer` is logged alongside each model (`scaler.json`) so inference reproduces the exact same standardization used at training time.

Run it:

```bash
pip install -r requirements.txt
python scripts/refresh_and_retrain.py
mlflow ui   # inspect runs + the model registry at http://localhost:5000
```

**Backend + dashboard.** `backend/main.py` (FastAPI) serves `/api/latest`, `/api/predictions`, `/api/history`, `/api/metrics`, and `/api/runs` to the `frontend/` Next.js dashboard, plus `POST /api/refresh` (fetch + snapshot) and `POST /api/retrain` (kicked off as a background job, polled via `/api/retrain/status/{job_id}` — retraining, especially QLSTM through the Pennylane simulator, can take several minutes).

```bash
uvicorn backend.main:app --reload          # http://localhost:8000
cd frontend && cp .env.example .env.local && npm install && npm run dev   # http://localhost:3000
```

**Automation.** Deliberately not wired up yet — `scripts/refresh_and_retrain.py` is a manual/on-demand entrypoint for now (see "Deployment" below for where a scheduled trigger would plug in later).

**Deployment.** Frontend → Vercel (native Next.js support). Backend → Render, as a Dockerized web service (`backend/Dockerfile`) — not a serverless function, since PyTorch/Pennylane are too heavy and MLflow's local file store needs a real persistent disk. Render's persistent disk is mounted across `data/` (holds `mlflow.db`, the registry, plus data snapshots) and `mlruns/` (model artifacts), so state survives redeploys. Set `ALLOWED_ORIGINS` (backend, comma-separated) to the deployed Vercel domain, and `NEXT_PUBLIC_API_BASE_URL` (frontend) to the deployed Render URL.

