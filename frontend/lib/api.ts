const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

export const MODEL_NAMES = ["lstm", "qlstm"] as const;
export type ModelName = (typeof MODEL_NAMES)[number];

export interface LatestResponse {
  as_of_date: string;
  close: number;
  snapshot_fetched_at: string;
}

export interface PredictionsResponse {
  last_close: number;
  predictions: Record<ModelName, number | null>;
}

export interface HistorySeries {
  dates: string[];
  actual: number[];
  predicted: number[];
}

export type HistoryResponse = Record<ModelName, HistorySeries | null>;

export interface ModelMetrics {
  rmse_mean: number;
  rmse_std: number;
  mae_mean: number;
  mae_std: number;
  run_id: string;
  start_time: string;
}

export type MetricsResponse = Record<ModelName, ModelMetrics | null>;

export interface RunRow {
  run_id: string;
  model: string;
  start_time: string;
  rmse_mean: number | string;
  mae_mean: number | string;
}

export interface RetrainStatus {
  status: "pending" | "running" | "done" | "error";
  result?: unknown;
  error?: string;
}

class ApiError extends Error {
  constructor(
    public path: string,
    public status: number,
  ) {
    super(`${path} failed: ${status}`);
  }
}

async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE_URL}${path}`, { ...init, cache: "no-store" });
  if (!res.ok) {
    throw new ApiError(path, res.status);
  }
  return res.json() as Promise<T>;
}

export { ApiError };

export function getLatest() {
  return apiFetch<LatestResponse>("/api/latest");
}

export function getPredictions() {
  return apiFetch<PredictionsResponse>("/api/predictions");
}

export function getHistory(days = 90) {
  return apiFetch<HistoryResponse>(`/api/history?days=${days}`);
}

export function getMetrics() {
  return apiFetch<MetricsResponse>("/api/metrics");
}

export function getRuns(limit = 20) {
  return apiFetch<RunRow[]>(`/api/runs?limit=${limit}`);
}

export function postRefresh() {
  return apiFetch<{ snapshot_path: string; n_rows: number }>("/api/refresh", { method: "POST" });
}

export function postRetrain() {
  return apiFetch<{ job_id: string }>("/api/retrain", { method: "POST" });
}

export function getRetrainStatus(jobId: string) {
  return apiFetch<RetrainStatus>(`/api/retrain/status/${jobId}`);
}
