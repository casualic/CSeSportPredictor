import { Suspense } from "react";
import Link from "next/link";
import { supabase } from "@/lib/supabase";
import { Prediction } from "@/lib/types";
import { filterByTier } from "@/lib/tier";
import StatsCard from "@/components/StatsCard";
import TierFilter from "@/components/TierFilter";

export const revalidate = 0;

export default async function Dashboard({ searchParams }: { searchParams: Promise<{ tier?: string }> }) {
  const { tier } = await searchParams;
  const { data: predictions } = await supabase
    .from("predictions")
    .select("*")
    .is("actual_winner", null)
    .order("created_at", { ascending: false });

  const items = filterByTier((predictions ?? []) as Prediction[], tier ?? null);

  /* Compute stats */
  const active = items.length;
  const modelsAgree = items.filter((p) => p.models_agree).length;

  return (
    <>
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:justify-between sm:items-end gap-3 mb-5 sm:mb-7">
        <div>
          <h1 className="font-serif text-[22px] sm:text-[26px] font-semibold text-slate-900 tracking-[-0.5px] mb-[3px]">
            Upcoming Predictions
          </h1>
          <p className="text-[13px] text-slate-400">
            Unresolved match forecasts from the ensemble model
          </p>
        </div>
        <Suspense>
          <TierFilter />
        </Suspense>
      </div>

      {/* Stats row */}
      <div className="grid grid-cols-2 gap-2.5 sm:gap-3.5 mb-5 sm:mb-7">
        <StatsCard title="Active" value={String(active)} subtitle="predictions pending" />
        <StatsCard
          title="Models Agree"
          value={String(modelsAgree)}
          subtitle={`of ${active} total`}
          color="green"
        />
      </div>

      {/* Table */}
      {items.length > 0 ? (
        <div className="bg-white border border-slate-400 rounded-[6px] overflow-x-auto -mx-4 sm:mx-0 border-x-0 sm:border-x rounded-none sm:rounded-[6px]">
          <table className="w-full border-collapse min-w-[700px]">
            <thead>
              <tr>
                <th className="text-[11px] font-semibold text-slate-400 uppercase tracking-[0.5px] py-3 px-3 sm:px-[18px] text-left border-b-2 border-slate-300 bg-slate-50">
                  Match
                </th>
                <th className="text-[11px] font-semibold text-slate-400 uppercase tracking-[0.5px] py-3 px-3 sm:px-[18px] text-left border-b-2 border-slate-300 bg-slate-50">
                  Event
                </th>
                <th className="text-[11px] font-semibold text-slate-400 uppercase tracking-[0.5px] py-3 px-3 sm:px-[18px] text-left border-b-2 border-slate-300 bg-slate-50">
                  Probability
                </th>
                <th className="text-[11px] font-semibold text-slate-400 uppercase tracking-[0.5px] py-3 px-3 sm:px-[18px] text-center border-b-2 border-slate-300 bg-slate-50">
                  FSVM
                </th>
                <th className="text-[11px] font-semibold text-slate-400 uppercase tracking-[0.5px] py-3 px-3 sm:px-[18px] text-center border-b-2 border-slate-300 bg-slate-50">
                  XGB
                </th>
                <th className="text-[11px] font-semibold text-slate-400 uppercase tracking-[0.5px] py-3 px-3 sm:px-[18px] text-center border-b-2 border-slate-300 bg-slate-50">
                  Agree
                </th>
                <th className="text-[11px] font-semibold text-slate-400 uppercase tracking-[0.5px] py-3 px-3 sm:px-[18px] text-center border-b-2 border-slate-300 bg-slate-50">
                  Edge
                </th>
                <th className="text-[11px] font-semibold text-slate-400 uppercase tracking-[0.5px] py-3 px-3 sm:px-[18px] text-left border-b-2 border-slate-300 bg-slate-50" />
              </tr>
            </thead>
            <tbody>
              {items.map((p) => {
                const t1Pct = (p.t1_win_prob * 100).toFixed(1);
                const t2Pct = ((1 - p.t1_win_prob) * 100).toFixed(1);
                const edgeVal = p.edge ?? 0;
                const edgePct = (edgeVal * 100).toFixed(1);
                const edgeColor =
                  edgeVal > 0.05
                    ? "text-green-600"
                    : edgeVal > 0
                      ? "text-amber-600"
                      : "text-red-600";

                return (
                  <tr
                    key={p.id}
                    className="border-b border-slate-300 last:border-b-0 hover:bg-blue-50 transition-colors duration-100 animate-[rowIn_0.3s_ease_both]"
                  >
                    {/* Match */}
                    <td className="py-3.5 px-3 sm:px-[18px] align-middle">
                      <div className="flex items-center gap-2">
                        <span
                          className={`font-semibold text-sm ${
                            p.predicted_winner === p.team1 ? "text-blue-600" : "text-slate-900"
                          }`}
                        >
                          {p.team1}
                        </span>
                        {p.t1_rank && (
                          <span className="font-mono text-[11px] text-slate-400">
                            #{p.t1_rank}
                          </span>
                        )}
                        <span className="text-[11px] text-slate-400 font-medium">vs</span>
                        <span
                          className={`font-semibold text-sm ${
                            p.predicted_winner === p.team2 ? "text-blue-600" : "text-slate-900"
                          }`}
                        >
                          {p.team2}
                        </span>
                        {p.t2_rank && (
                          <span className="font-mono text-[11px] text-slate-400">
                            #{p.t2_rank}
                          </span>
                        )}
                      </div>
                    </td>

                    {/* Event */}
                    <td className="py-3.5 px-3 sm:px-[18px] align-middle text-slate-600 text-[13px]">
                      {p.event || ""}
                      {p.bo_format && (
                        <span className="inline-block text-[11px] font-semibold text-slate-400 bg-slate-50 border border-slate-200 px-2 py-[1px] rounded-[3px] ml-2">
                          {p.bo_format}
                        </span>
                      )}
                    </td>

                    {/* Probability bar */}
                    <td className="py-3.5 px-3 sm:px-[18px] align-middle">
                      <div className="w-[120px] sm:w-[180px]">
                        <div className="flex h-1.5 rounded-[3px] overflow-hidden ring-1 ring-blue-600">
                          <div
                            className="bg-emerald-500 rounded-l-[3px] transition-[width] duration-600"
                            style={{ width: `${t1Pct}%` }}
                          />
                          <div className="bg-red-400 flex-1 rounded-r-[3px]" />
                        </div>
                        <div className="flex justify-between mt-1 font-mono text-[11px]">
                          <span className="text-emerald-600 font-medium">{t1Pct}%</span>
                          <span className="text-red-500">{t2Pct}%</span>
                        </div>
                      </div>
                    </td>

                    {/* FSVM */}
                    <td className="py-3.5 px-3 sm:px-[18px] align-middle font-mono text-[13px] text-slate-600 text-center">
                      {(p.fsvm_prob * 100).toFixed(1)}%
                    </td>

                    {/* XGB */}
                    <td className="py-3.5 px-3 sm:px-[18px] align-middle font-mono text-[13px] text-slate-600 text-center">
                      {(p.xgb_prob * 100).toFixed(1)}%
                    </td>

                    {/* Agree */}
                    <td className="py-3.5 px-3 sm:px-[18px] align-middle text-center">
                      <span
                        className={`text-[11px] font-semibold px-2 py-[2px] rounded-[3px] ${
                          p.models_agree
                            ? "bg-green-50 text-green-600"
                            : "bg-red-50 text-red-600"
                        }`}
                      >
                        {p.models_agree ? "Yes" : "No"}
                      </span>
                    </td>

                    {/* Edge */}
                    <td
                      className={`py-3.5 px-3 sm:px-[18px] align-middle text-center font-mono text-[13px] font-medium ${edgeColor}`}
                    >
                      {edgeVal >= 0 ? "+" : ""}
                      {edgePct}%
                    </td>

                    {/* Action */}
                    <td className="py-3.5 px-3 sm:px-[18px] align-middle">
                      <Link
                        href={`/match/${p.id}`}
                        className="text-xs font-medium text-blue-600 no-underline px-2.5 py-1 border border-blue-100 rounded-[4px] hover:bg-blue-50 hover:border-blue-500 transition-all duration-150"
                      >
                        Details
                      </Link>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      ) : (
        <div className="text-center text-slate-400 py-16">
          <p className="text-lg">No upcoming predictions yet.</p>
          <p className="text-sm mt-2">
            Run the scraper to add predictions from HLTV.
          </p>
        </div>
      )}
    </>
  );
}
