import { Prediction } from "./types";

export function filterByTier(predictions: Prediction[], tier: string | null): Prediction[] {
  if (!tier || tier === "all") return predictions;
  const maxRank = parseInt(tier, 10);
  if (isNaN(maxRank)) return predictions;
  return predictions.filter((p) => {
    const bestRank = Math.min(p.t1_rank ?? 999, p.t2_rank ?? 999);
    return bestRank <= maxRank;
  });
}
