"use client";

import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { MODEL_NAMES, type HistoryResponse } from "@/lib/api";
import { MODEL_META } from "@/lib/models";

interface Row {
  date: string;
  actual: number | null;
  lstm: number | null;
  qlstm: number | null;
}

function mergeSeries(history: HistoryResponse): Row[] {
  const anySeries = history.lstm ?? history.qlstm;
  if (!anySeries) return [];

  return anySeries.dates.map((date, i) => ({
    date,
    actual: anySeries.actual[i] ?? null,
    lstm: history.lstm?.predicted[i] ?? null,
    qlstm: history.qlstm?.predicted[i] ?? null,
  }));
}

export default function HistoryChart({ history }: { history: HistoryResponse }) {
  const rows = mergeSeries(history);

  if (rows.length === 0) {
    return (
      <section>
        <h2 className="text-sm font-medium text-[var(--text-secondary)] mb-2">
          Actual vs. predicted close
        </h2>
        <p className="text-sm text-[var(--text-muted)]">No champion model registered yet.</p>
      </section>
    );
  }

  return (
    <section>
      <h2 className="text-sm font-medium text-[var(--text-secondary)] mb-2">
        Actual vs. predicted close &mdash; last {rows.length} trading days
      </h2>
      <div className="h-72 w-full" style={{ background: "var(--surface)" }}>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={rows} margin={{ top: 8, right: 16, bottom: 0, left: 0 }}>
            <CartesianGrid stroke="var(--chart-grid)" vertical={false} />
            <XAxis
              dataKey="date"
              stroke="var(--chart-axis)"
              tick={{ fill: "var(--text-muted)", fontSize: 12 }}
              tickLine={false}
              axisLine={{ stroke: "var(--chart-axis)" }}
              minTickGap={40}
            />
            <YAxis
              stroke="var(--chart-axis)"
              tick={{ fill: "var(--text-muted)", fontSize: 12 }}
              tickLine={false}
              axisLine={false}
              width={56}
              domain={["auto", "auto"]}
              tickFormatter={(v: number) => `$${v.toFixed(0)}`}
            />
            <Tooltip
              contentStyle={{
                background: "var(--surface)",
                border: "1px solid var(--border)",
                borderRadius: 6,
                fontSize: 12,
              }}
              formatter={(value) => (typeof value === "number" ? `$${value.toFixed(2)}` : value)}
            />
            <Legend wrapperStyle={{ fontSize: 12, color: "var(--text-secondary)" }} />
            <Line
              type="monotone"
              dataKey="actual"
              name="Actual"
              stroke="var(--series-actual)"
              strokeWidth={2}
              strokeDasharray="4 3"
              dot={false}
              connectNulls
            />
            {MODEL_NAMES.map((name) => (
              <Line
                key={name}
                type="monotone"
                dataKey={name}
                name={MODEL_META[name].label}
                stroke={MODEL_META[name].colorVar}
                strokeWidth={2}
                dot={false}
                connectNulls
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </section>
  );
}
