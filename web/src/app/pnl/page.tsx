import { Suspense } from "react";
import { supabase } from "@/lib/supabase";
import { Bet, Prediction } from "@/lib/types";
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

  const totalBets = items.length;
  const wins = items.filter((b) => b.won).length;
  const winRate = totalBets > 0 ? wins / totalBets : 0;
  const totalPnl = items.reduce((acc, b) => acc + b.pnl, 0);
  const totalStaked = items.reduce((acc, b) => acc + b.stake, 0);
  const roi = totalStaked > 0 ? (totalPnl / totalStaked) * 100 : 0;

  const cumulative: number[] = [];
  let running = 0;
  for (const b of items) {
    running += b.pnl;
    cumulative.push(Math.round(running * 100) / 100);
  }

  const reversed = [...items].reverse();

  return (
    <>
      <div className="flex justify-between items-center mb-6">
        <h2 className="text-2xl font-bold">Profit &amp; Loss</h2>
        <Suspense><TierFilter /></Suspense>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-6">
        <StatsCard title="Total Bets" value={String(totalBets)} />
        <StatsCard
          title="Win Rate"
          value={`${totalBets ? (winRate * 100).toFixed(1) : 0}%`}
          subtitle={`${wins}/${totalBets}`}
        />
        <StatsCard
          title="Total P&L"
          value={`$${totalPnl.toFixed(2)}`}
          color={totalPnl > 0 ? "green" : totalPnl < 0 ? "red" : "default"}
        />
        <StatsCard
          title="ROI"
          value={`${roi.toFixed(1)}%`}
          color={roi > 0 ? "green" : roi < 0 ? "red" : "default"}
        />
      </div>

      {cumulative.length > 0 && (
        <div className="bg-gray-900 rounded-lg border border-gray-800 p-4 mb-6">
          <PnlChart cumulative={cumulative} />
        </div>
      )}

      {reversed.length > 0 ? (
        <>
          <h3 className="text-lg font-semibold mb-3">Bet History</h3>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-gray-800 text-gray-400">
                  <th className="text-left py-3 px-2 font-medium">#</th>
                  <th className="text-left py-3 px-2 font-medium">Team</th>
                  <th className="text-left py-3 px-2 font-medium">Odds</th>
                  <th className="text-left py-3 px-2 font-medium">Model Prob</th>
                  <th className="text-left py-3 px-2 font-medium">Edge</th>
                  <th className="text-left py-3 px-2 font-medium">Result</th>
                  <th className="text-left py-3 px-2 font-medium">P&amp;L</th>
                </tr>
              </thead>
              <tbody>
                {reversed.map((b, i) => (
                  <tr key={b.id} className="border-b border-gray-800/50 hover:bg-gray-900/50">
                    <td className="py-2 px-2 text-gray-500">{reversed.length - i}</td>
                    <td className="py-2 px-2">{b.bet_team}</td>
                    <td className="py-2 px-2">{b.bet_odds?.toFixed(2) ?? "-"}</td>
                    <td className="py-2 px-2">{b.model_prob ? `${(b.model_prob * 100).toFixed(1)}%` : "-"}</td>
                    <td className="py-2 px-2">{b.edge ? `${(b.edge * 100).toFixed(1)}%` : "-"}</td>
                    <td className="py-2 px-2">
                      {b.won ? (
                        <span className="text-xs bg-green-500/20 text-green-400 px-2 py-0.5 rounded">Won</span>
                      ) : (
                        <span className="text-xs bg-red-500/20 text-red-400 px-2 py-0.5 rounded">Lost</span>
                      )}
                    </td>
                    <td className={`py-2 px-2 font-medium ${b.pnl > 0 ? "text-green-500" : "text-red-500"}`}>
                      ${b.pnl.toFixed(2)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      ) : (
        <div className="text-center text-gray-500 py-16">
          <p className="text-lg">No bets recorded yet.</p>
          <p className="text-sm mt-2">Bets are created when predictions with positive edge are resolved.</p>
        </div>
      )}
    </>
  );
}
