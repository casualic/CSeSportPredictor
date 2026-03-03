import { Suspense } from "react";
import { supabase } from "@/lib/supabase";
import { Bet } from "@/lib/types";
import StatsCard from "@/components/StatsCard";
import PnlChart from "@/components/PnlChart";
import TierFilter from "@/components/TierFilter";

export const revalidate = 0;

export default async function PnL({ searchParams }: { searchParams: Promise<{ tier?: string }> }) {
  const { tier } = await searchParams;
  const { data: bets } = await supabase
    .from("bets")
    .select("*, predictions(t1_rank, t2_rank)")
    .order("created_at", { ascending: true });

  let items = (bets ?? []) as (Bet & { predictions: { t1_rank: number | null; t2_rank: number | null } | null })[];

  /* Tier filter */
  if (tier && tier !== "all") {
    const maxRank = parseInt(tier, 10);
    if (!isNaN(maxRank)) {
      items = items.filter((b) => {
        const p = b.predictions;
        if (!p) return false;
        const bestRank = Math.min(p.t1_rank ?? 999, p.t2_rank ?? 999);
        return bestRank <= maxRank;
      });
    }
  }

  /* Strategy filter: only bets where edge >= 5% (model_prob - implied_prob >= 0.05) */
  const strategyBets = items.filter((b) => b.edge >= 0.05);

  const totalBets = strategyBets.length;
  const wins = strategyBets.filter((b) => b.won).length;
  const winRate = totalBets > 0 ? wins / totalBets : 0;
  const totalPnl = strategyBets.reduce((acc, b) => acc + b.pnl, 0);
  const totalStaked = strategyBets.reduce((acc, b) => acc + b.stake, 0);
  const roi = totalStaked > 0 ? (totalPnl / totalStaked) * 100 : 0;

  /* Cumulative P&L for chart */
  const cumulative: number[] = [];
  let running = 0;
  for (const b of strategyBets) {
    running += b.pnl;
    cumulative.push(Math.round(running * 100) / 100);
  }

  const reversed = [...strategyBets].reverse();

  return (
    <>
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:justify-between sm:items-end gap-3 mb-5 sm:mb-7">
        <div>
          <h1 className="font-serif text-[22px] sm:text-[26px] font-semibold text-slate-900 tracking-[-0.5px] mb-[3px]">
            Profit &amp; Loss
          </h1>
          <p className="text-[13px] text-slate-400">
            Strategy: bet when model edge &ge; 5% — positive or negative — vs bookmaker odds
          </p>
        </div>
        <Suspense>
          <TierFilter />
        </Suspense>
      </div>

      {/* Stats row */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5 sm:gap-3.5 mb-5 sm:mb-7">
        <StatsCard title="Total Bets" value={String(totalBets)} subtitle="within strategy" />
        <StatsCard
          title="Win Rate"
          value={`${totalBets ? (winRate * 100).toFixed(1) : 0}%`}
          subtitle={`${wins}/${totalBets}`}
          color="blue"
        />
        <StatsCard
          title="Total P&L"
          value={`$${totalPnl.toFixed(2)}`}
          subtitle="net profit/loss"
          color={totalPnl > 0 ? "green" : totalPnl < 0 ? "red" : "default"}
        />
        <StatsCard
          title="ROI"
          value={`${roi.toFixed(1)}%`}
          subtitle="return on investment"
          color={roi > 0 ? "green" : roi < 0 ? "red" : "default"}
        />
      </div>

      {/* Cumulative P&L chart */}
      {cumulative.length > 0 && (
        <div className="bg-white border border-slate-400 rounded-[6px] p-3 sm:p-5 mb-5 sm:mb-7">
          <h4 className="text-[11px] font-semibold text-slate-400 uppercase tracking-[0.5px] mb-3">
            Cumulative P&amp;L
          </h4>
          <PnlChart cumulative={cumulative} />
        </div>
      )}

      {/* Bet history table */}
      {reversed.length > 0 ? (
        <>
          <h3 className="font-serif text-base sm:text-lg font-semibold text-slate-900 mb-3">Bet History</h3>
          <div className="bg-white border border-slate-400 rounded-[6px] overflow-x-auto -mx-4 sm:mx-0 border-x-0 sm:border-x rounded-none sm:rounded-[6px]">
            <table className="w-full border-collapse min-w-[650px]">
              <thead>
                <tr>
                  {["#", "Team", "Odds", "Model Prob", "Edge", "Stake", "Result", "P&L"].map(
                    (header) => (
                      <th
                        key={header}
                        className="text-[11px] font-semibold text-slate-400 uppercase tracking-[0.5px] py-3 px-3 sm:px-[18px] text-left border-b-2 border-slate-300 bg-slate-50"
                      >
                        {header}
                      </th>
                    )
                  )}
                </tr>
              </thead>
              <tbody>
                {reversed.map((b, i) => (
                  <tr
                    key={b.id}
                    className="border-b border-slate-300 last:border-b-0 hover:bg-blue-50 transition-colors duration-100 animate-[rowIn_0.3s_ease_both]"
                  >
                    <td className="py-3.5 px-3 sm:px-[18px] align-middle font-mono text-[13px] text-slate-400">
                      {reversed.length - i}
                    </td>
                    <td className="py-3.5 px-3 sm:px-[18px] align-middle text-sm font-medium text-slate-900">
                      {b.bet_team}
                    </td>
                    <td className="py-3.5 px-3 sm:px-[18px] align-middle font-mono text-[13px] text-slate-600">
                      {b.bet_odds?.toFixed(2) ?? "-"}
                    </td>
                    <td className="py-3.5 px-3 sm:px-[18px] align-middle font-mono text-[13px] text-slate-600">
                      {b.model_prob ? `${(b.model_prob * 100).toFixed(1)}%` : "-"}
                    </td>
                    <td className="py-3.5 px-3 sm:px-[18px] align-middle font-mono text-[13px] font-medium text-green-600">
                      {b.edge ? `+${(b.edge * 100).toFixed(1)}%` : "-"}
                    </td>
                    <td className="py-3.5 px-3 sm:px-[18px] align-middle font-mono text-[13px] text-slate-600">
                      ${b.stake.toFixed(2)}
                    </td>
                    <td className="py-3.5 px-3 sm:px-[18px] align-middle">
                      {b.won ? (
                        <span className="text-[11px] font-semibold px-2 py-[2px] rounded-[3px] bg-green-50 text-green-600">
                          Won
                        </span>
                      ) : (
                        <span className="text-[11px] font-semibold px-2 py-[2px] rounded-[3px] bg-red-50 text-red-600">
                          Lost
                        </span>
                      )}
                    </td>
                    <td
                      className={`py-3.5 px-3 sm:px-[18px] align-middle font-mono text-[13px] font-medium ${
                        b.pnl > 0 ? "text-green-600" : "text-red-600"
                      }`}
                    >
                      {b.pnl > 0 ? "+" : ""}${b.pnl.toFixed(2)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      ) : (
        <div className="text-center text-slate-400 py-16">
          <p className="text-lg">No bets matching the strategy yet.</p>
          <p className="text-sm mt-2">
            Bets with edge &ge; 5% will appear here when resolved.
          </p>
        </div>
      )}
    </>
  );
}
