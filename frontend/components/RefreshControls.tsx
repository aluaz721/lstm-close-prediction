"use client";

import { useRef, useState } from "react";
import { useRouter } from "next/navigation";

import { getRetrainStatus, postRefresh, postRetrain, type RetrainStatus } from "@/lib/api";

type Phase = "idle" | "refreshing" | "retraining" | "error";

export default function RefreshControls() {
  const router = useRouter();
  const [phase, setPhase] = useState<Phase>("idle");
  const [message, setMessage] = useState<string | null>(null);
  const pollTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  async function handleRefresh() {
    setPhase("refreshing");
    setMessage(null);
    try {
      const result = await postRefresh();
      setMessage(`Fetched ${result.n_rows} rows.`);
      setPhase("idle");
      router.refresh();
    } catch (e) {
      setPhase("error");
      setMessage(e instanceof Error ? e.message : "Refresh failed.");
    }
  }

  function pollStatus(jobId: string) {
    getRetrainStatus(jobId)
      .then((status: RetrainStatus) => {
        if (status.status === "done") {
          setPhase("idle");
          setMessage("Retrain complete -- champion updated if a model improved.");
          router.refresh();
        } else if (status.status === "error") {
          setPhase("error");
          setMessage(status.error ?? "Retrain failed.");
        } else {
          pollTimer.current = setTimeout(() => pollStatus(jobId), 4000);
        }
      })
      .catch((e) => {
        setPhase("error");
        setMessage(e instanceof Error ? e.message : "Lost track of the retrain job.");
      });
  }

  async function handleRetrain() {
    setPhase("retraining");
    setMessage("Retraining started -- this can take several minutes, especially for QLSTM.");
    try {
      const { job_id } = await postRetrain();
      pollStatus(job_id);
    } catch (e) {
      setPhase("error");
      setMessage(e instanceof Error ? e.message : "Could not start retrain.");
    }
  }

  const busy = phase === "refreshing" || phase === "retraining";

  return (
    <div className="flex flex-wrap items-center gap-3">
      <button
        onClick={handleRefresh}
        disabled={busy}
        className="rounded-md border px-3 py-1.5 text-sm font-medium transition-colors disabled:opacity-50"
        style={{ borderColor: "var(--border)" }}
      >
        {phase === "refreshing" ? "Refreshing…" : "Refresh data"}
      </button>
      <button
        onClick={handleRetrain}
        disabled={busy}
        className="rounded-md border px-3 py-1.5 text-sm font-medium transition-colors disabled:opacity-50"
        style={{ borderColor: "var(--border)" }}
      >
        {phase === "retraining" ? "Retraining…" : "Retrain now"}
      </button>
      {message && (
        <span
          className="text-sm"
          style={{ color: phase === "error" ? "var(--status-critical)" : "var(--text-secondary)" }}
        >
          {message}
        </span>
      )}
    </div>
  );
}
