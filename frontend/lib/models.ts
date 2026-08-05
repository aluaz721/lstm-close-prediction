import type { ModelName } from "./api";

export const MODEL_META: Record<ModelName, { label: string; colorVar: string }> = {
  lstm: { label: "LSTM", colorVar: "var(--series-lstm)" },
  qlstm: { label: "QLSTM", colorVar: "var(--series-qlstm)" },
};
