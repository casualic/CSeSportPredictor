import { Suspense } from "react";
import Link from "next/link";
import { supabase } from "@/lib/supabase";
import { Prediction } from "@/lib/types";
import { filterByTier } from "@/lib/tier";
import StatsCard from "@/components/StatsCard";
import TierFilter from "@/components/TierFilter";
import { RankingsProvider, RankingsButton, RankingsDropdown } from "@/components/RankingsPanel";

export const revalidate = 0;

export default async function Results({ searchParams }: { searchParams: Promise<{ tier?: string }> }) {
  const { tier } = await searchParams;
  const { data: predictions } = await supabase
    .from("predictions")
    .select("*")
    .not("actual_winner", "is", null)
    .order("resolved_at", { ascending: false });

  const items = filterByTier((predictions ?? []) as Prediction[], tier ?? "30");

  const total = items.length;
  const correct = items.filter((p) => p.prediction_correct).length;
  const accuracy = total > 0 ? correct / total : 0;
  const agreed = items.filter((p) => p.models_agree).length;
  const agreedCorrect = items.filter((p) => p.prediction_correct && p.models_agree).length;
  const agreedAccuracy = agreed > 0 ? agreedCorrect / agreed : 0;

  return (
    <RankingsProvider>
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:justify-between sm:items-end gap-3 mb-5 sm:mb-7">
        <div>
          <h1 className="font-serif text-[22px] sm:text-[26px] font-semibold text-slate-900 tracking-[-0.5px] mb-[3px]">
            Prediction Results
          </h1>
          <p className="text-[13px] text-slate-400">
            Resolved match outcomes and model accuracy
          </p>
        </div>
        <div className="flex items-center gap-2">
          <RankingsButton />
          <Suspense>
            <TierFilter defaultTier="30" />
          </Suspense>
        </div>
      </div>

      <RankingsDropdown />

      {/* Stats row */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5 sm:gap-3.5 mb-5 sm:mb-7">
        <StatsCard title="Total Resolved" value={String(total)} subtitle="matches completed" />
        <StatsCard title="Correct" value={String(correct)} subtitle={`of ${total} total`} color="green" />
        <StatsCard
          title="Accuracy"
          value={`${(accuracy * 100).toFixed(1)}%`}
          subtitle="overall hit rate"
          color="blue"
        />
        <StatsCard
          title="Models Agree Acc"
          value={`${(agreedAccuracy * 100).toFixed(1)}%`}
          subtitle={`${agreedCorrect}/${agreed}`}
          color="green"
        />
      </div>

      {/* Table */}
      {items.length > 0 ? (
        <div className="bg-white border border-slate-400 rounded-[6px] overflow-x-auto -mx-4 sm:mx-0 border-x-0 sm:border-x rounded-none sm:rounded-[6px]">
          <table className="w-full border-collapse min-w-[650px]">
            <thead>
              <tr>
                {["Match", "Event", "Predicted", "Actual", "Model Prob", "Agree", "Result"].map(
                  (header) => (
                    <th
                      key={header}
                      className="text-[11px] font-semibold text-slate-400 uppercase tracking-[0.5px] py-3 px-3 sm:px-[18px] text-left border-b-2 border-slate-300 bg-slate-50"
                    >
                      {header}
                    </th>
                  )
                )}
                <th className="text-[11px] font-semibold text-slate-400 uppercase tracking-[0.5px] py-3 px-3 sm:px-[18px] text-left border-b-2 border-slate-300 bg-slate-50" />
              </tr>
            </thead>
            <tbody>
              {items.map((p) => (
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
                        <span className="font-mono text-[11px] text-slate-400">#{p.t1_rank}</span>
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
                        <span className="font-mono text-[11px] text-slate-400">#{p.t2_rank}</span>
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

                  {/* Predicted */}
                  <td className="py-3.5 px-3 sm:px-[18px] align-middle text-sm font-medium text-slate-900">
                    {p.predicted_winner}
                  </td>

                  {/* Actual */}
                  <td className="py-3.5 px-3 sm:px-[18px] align-middle text-sm font-medium text-slate-900">
                    {p.actual_winner}
                  </td>

                  {/* Model Prob */}
                  <td className="py-3.5 px-3 sm:px-[18px] align-middle font-mono text-[13px] text-slate-600">
                    {(p.t1_win_prob * 100).toFixed(1)}%
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

                  {/* Result */}
                  <td className="py-3.5 px-3 sm:px-[18px] align-middle">
                    {p.prediction_correct ? (
                      <span className="text-[11px] font-semibold px-2 py-[2px] rounded-[3px] bg-green-50 text-green-600">
                        Correct
                      </span>
                    ) : (
                      <span className="text-[11px] font-semibold px-2 py-[2px] rounded-[3px] bg-red-50 text-red-600">
                        Wrong
                      </span>
                    )}
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
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <div className="text-center text-slate-400 py-16">
          <p className="text-lg">No resolved predictions yet.</p>
        </div>
      )}
    </RankingsProvider>
  );
}
