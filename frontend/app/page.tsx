import { getHistory, getLatest, getMetrics, getPredictions, getRuns } from "@/lib/api";
import PriceHeader from "@/components/PriceHeader";
import PredictionCards from "@/components/PredictionCards";
import HistoryChart from "@/components/HistoryChart";
import MetricsComparison from "@/components/MetricsComparison";
import RunsTable from "@/components/RunsTable";
import RefreshControls from "@/components/RefreshControls";

export const dynamic = "force-dynamic";

export default async function Home() {
  const latest = await getLatest().catch(() => null);

  if (!latest) {
    return (
      <main className="flex-1 flex items-center justify-center p-8">
        <div className="text-center max-w-sm">
          <h1 className="text-xl font-semibold mb-2">MRK Close Prediction</h1>
          <p className="text-sm text-[var(--text-secondary)] mb-6">
            No data yet. Fetch the latest MRK data to get started.
          </p>
          <div className="flex justify-center">
            <RefreshControls />
          </div>
        </div>
      </main>
    );
  }

  const [predictions, history, metrics, runs] = await Promise.all([
    getPredictions(),
    getHistory(90),
    getMetrics(),
    getRuns(20),
  ]);

  return (
    <main className="flex-1 w-full max-w-4xl mx-auto p-6 space-y-8">
      <PriceHeader latest={latest} />
      <RefreshControls />
      <PredictionCards predictions={predictions} />
      <HistoryChart history={history} />
      <MetricsComparison metrics={metrics} />
      <RunsTable runs={runs} />
    </main>
  );
}
