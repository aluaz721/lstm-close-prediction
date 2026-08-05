import { MODEL_NAMES, type PredictionsResponse } from "@/lib/api";
import { MODEL_META } from "@/lib/models";

export default function PredictionCards({ predictions }: { predictions: PredictionsResponse }) {
  return (
    <section aria-label="Next-close predictions" className="grid grid-cols-1 sm:grid-cols-2 gap-4">
      {MODEL_NAMES.map((name) => {
        const meta = MODEL_META[name];
        const value = predictions.predictions[name];
        const delta = value === null || value === undefined ? null : value - predictions.last_close;

        return (
          <div
            key={name}
            className="rounded-lg border p-4"
            style={{ borderColor: "var(--border)", background: "var(--surface)" }}
          >
            <div className="flex items-center gap-2 mb-1">
              <span
                aria-hidden
                className="inline-block h-2.5 w-2.5 rounded-full"
                style={{ background: meta.colorVar }}
              />
              <span className="text-sm font-medium text-[var(--text-secondary)]">
                {meta.label} &mdash; predicted next close
              </span>
            </div>

            {value === null || value === undefined ? (
              <p className="text-sm text-[var(--text-muted)]">
                No champion model registered yet &mdash; run a retrain first.
              </p>
            ) : (
              <div className="flex items-baseline gap-2">
                <span className="text-2xl font-semibold tabular-nums">${value.toFixed(2)}</span>
                <span className="text-sm tabular-nums text-[var(--text-secondary)]">
                  {delta !== null && (delta >= 0 ? "▲ " : "▼ ")}
                  {delta !== null && `$${Math.abs(delta).toFixed(2)}`}
                </span>
              </div>
            )}
          </div>
        );
      })}
    </section>
  );
}
