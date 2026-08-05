import type { RunRow } from "@/lib/api";

export default function RunsTable({ runs }: { runs: RunRow[] }) {
  return (
    <section>
      <h2 className="text-sm font-medium text-[var(--text-secondary)] mb-2">Recent MLflow runs</h2>
      {runs.length === 0 ? (
        <p className="text-sm text-[var(--text-muted)]">No runs yet.</p>
      ) : (
        <div className="overflow-x-auto rounded-lg border" style={{ borderColor: "var(--border)" }}>
          <table className="w-full text-sm tabular-nums">
            <thead>
              <tr className="text-left text-[var(--text-muted)] text-xs uppercase tracking-wide">
                <th className="px-3 py-2 font-medium">Model</th>
                <th className="px-3 py-2 font-medium">Started</th>
                <th className="px-3 py-2 font-medium text-right">RMSE (mean)</th>
                <th className="px-3 py-2 font-medium text-right">MAE (mean)</th>
                <th className="px-3 py-2 font-medium">Run ID</th>
              </tr>
            </thead>
            <tbody>
              {runs.map((run) => (
                <tr key={run.run_id} className="border-t" style={{ borderColor: "var(--border)" }}>
                  <td className="px-3 py-2">{run.model || "—"}</td>
                  <td className="px-3 py-2 text-[var(--text-secondary)]">
                    {run.start_time ? new Date(run.start_time).toLocaleString() : "—"}
                  </td>
                  <td className="px-3 py-2 text-right">
                    {typeof run.rmse_mean === "number" ? run.rmse_mean.toFixed(3) : "—"}
                  </td>
                  <td className="px-3 py-2 text-right">
                    {typeof run.mae_mean === "number" ? run.mae_mean.toFixed(3) : "—"}
                  </td>
                  <td className="px-3 py-2 text-[var(--text-muted)] font-mono text-xs">
                    {run.run_id.slice(0, 8)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}
