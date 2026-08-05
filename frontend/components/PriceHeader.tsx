import type { LatestResponse } from "@/lib/api";

export default function PriceHeader({ latest }: { latest: LatestResponse }) {
  const fetchedAt = new Date(latest.snapshot_fetched_at);

  return (
    <header className="flex flex-wrap items-baseline justify-between gap-x-6 gap-y-2">
      <div>
        <h1 className="text-xl font-semibold tracking-tight">MRK &mdash; Merck &amp; Co.</h1>
        <p className="text-sm text-[var(--text-secondary)]">
          Close as of {latest.as_of_date}
        </p>
      </div>
      <div className="text-right">
        <p className="text-3xl font-semibold tabular-nums">${latest.close.toFixed(2)}</p>
        <p className="text-xs text-[var(--text-muted)]">
          Data fetched {fetchedAt.toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" })}
        </p>
      </div>
    </header>
  );
}
