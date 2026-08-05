"use client";

import {
  Bar,
  BarChart,
  CartesianGrid,
  ErrorBar,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { MODEL_NAMES, type MetricsResponse } from "@/lib/api";
import { MODEL_META } from "@/lib/models";

export default function MetricsComparison({ metrics }: { metrics: MetricsResponse }) {
  const hasAny = MODEL_NAMES.some((name) => metrics[name]);
  if (!hasAny) {
    return (
      <section>
        <h2 className="text-sm font-medium text-[var(--text-secondary)] mb-2">
          Walk-forward validation
        </h2>
        <p className="text-sm text-[var(--text-muted)]">No walk-forward runs logged yet.</p>
      </section>
    );
  }

  const rows = [
    {
      metric: "RMSE",
      lstm: metrics.lstm?.rmse_mean ?? null,
      lstmStd: metrics.lstm?.rmse_std ?? 0,
      qlstm: metrics.qlstm?.rmse_mean ?? null,
      qlstmStd: metrics.qlstm?.rmse_std ?? 0,
    },
    {
      metric: "MAE",
      lstm: metrics.lstm?.mae_mean ?? null,
      lstmStd: metrics.lstm?.mae_std ?? 0,
      qlstm: metrics.qlstm?.mae_mean ?? null,
      qlstmStd: metrics.qlstm?.mae_std ?? 0,
    },
  ];

  return (
    <section>
      <h2 className="text-sm font-medium text-[var(--text-secondary)] mb-2">
        Walk-forward validation &mdash; mean &plusmn; std across folds
      </h2>
      <div className="h-64 w-full" style={{ background: "var(--surface)" }}>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={rows} margin={{ top: 8, right: 16, bottom: 0, left: 0 }} barGap={6}>
            <CartesianGrid stroke="var(--chart-grid)" vertical={false} />
            <XAxis
              dataKey="metric"
              stroke="var(--chart-axis)"
              tick={{ fill: "var(--text-muted)", fontSize: 12 }}
              tickLine={false}
              axisLine={{ stroke: "var(--chart-axis)" }}
            />
            <YAxis
              stroke="var(--chart-axis)"
              tick={{ fill: "var(--text-muted)", fontSize: 12 }}
              tickLine={false}
              axisLine={false}
              width={48}
            />
            <Tooltip
              contentStyle={{
                background: "var(--surface)",
                border: "1px solid var(--border)",
                borderRadius: 6,
                fontSize: 12,
              }}
              formatter={(value) => (typeof value === "number" ? value.toFixed(3) : value)}
            />
            <Legend wrapperStyle={{ fontSize: 12, color: "var(--text-secondary)" }} />
            <Bar dataKey="lstm" name={MODEL_META.lstm.label} fill={MODEL_META.lstm.colorVar} radius={[3, 3, 0, 0]}>
              <ErrorBar dataKey="lstmStd" stroke="var(--text-muted)" width={4} />
            </Bar>
            <Bar dataKey="qlstm" name={MODEL_META.qlstm.label} fill={MODEL_META.qlstm.colorVar} radius={[3, 3, 0, 0]}>
              <ErrorBar dataKey="qlstmStd" stroke="var(--text-muted)" width={4} />
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </section>
  );
}
