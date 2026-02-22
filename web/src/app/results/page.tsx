import { Suspense } from "react";
import Link from "next/link";
import { supabase } from "@/lib/supabase";
import { Prediction } from "@/lib/types";
import { filterByTier } from "@/lib/tier";
import StatsCard from "@/components/StatsCard";
import TierFilter from "@/components/TierFilter";

export const revalidate = 0;

export default async function Results({ searchParams }: { searchParams: Promise<{ tier?: string }> }) {
  const { tier } = await searchParams;
  const { data: predictions } = await supabase
    .from("predictions")
    .select("*")
    .not("actual_winner", "is", null)
    .order("resolved_at", { ascending: false });

  const items = filterByTier((predictions ?? []) as Prediction[], tier ?? null);

  const total = items.length;
  const correct = items.filter((p) => p.prediction_correct).length;
  const accuracy = total > 0 ? correct / total : 0;
  const agreed = items.filter((p) => p.models_agree).length;
  const agreedCorrect = items.filter((p) => p.prediction_correct && p.models_agree).length;
  const agreedAccuracy = agreed > 0 ? agreedCorrect / agreed : 0;

  return (
    <>
      <div className="flex justify-between items-center mb-6">
        <h2 className="text-2xl font-bold">Prediction Results</h2>
        <Suspense><TierFilter /></Suspense>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-6">
        <StatsCard title="Total Resolved" value={String(total)} />
        <StatsCard title="Correct" value={String(correct)} color="green" />
        <StatsCard title="Accuracy" value={`${(accuracy * 100).toFixed(1)}%`} />
        <StatsCard
          title="Models Agree Acc"
          value={`${(agreedAccuracy * 100).toFixed(1)}%`}
          subtitle={`${agreedCorrect}/${agreed}`}
        />
      </div>

      {items.length > 0 ? (
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-gray-800 text-gray-400">
                <th className="text-left py-3 px-2 font-medium">Match</th>
                <th className="text-left py-3 px-2 font-medium">Event</th>
                <th className="text-left py-3 px-2 font-medium">Predicted</th>
                <th className="text-left py-3 px-2 font-medium">Actual</th>
                <th className="text-left py-3 px-2 font-medium">T1 Prob</th>
                <th className="text-left py-3 px-2 font-medium">Agree</th>
                <th className="text-left py-3 px-2 font-medium">Result</th>
              </tr>
            </thead>
            <tbody>
              {items.map((p) => (
                <tr key={p.id} className="border-b border-gray-800/50 hover:bg-gray-900/50">
                  <td className="py-2 px-2">
                    <Link href={`/match/${p.id}`} className="text-blue-400 hover:text-blue-300">
                      {p.team1} vs {p.team2}
                    </Link>
                  </td>
                  <td className="py-2 px-2 text-gray-500 text-xs">{p.event || ""}</td>
                  <td className="py-2 px-2">{p.predicted_winner}</td>
                  <td className="py-2 px-2">{p.actual_winner}</td>
                  <td className="py-2 px-2">{(p.t1_win_prob * 100).toFixed(1)}%</td>
                  <td className="py-2 px-2">{p.models_agree ? "Yes" : "No"}</td>
                  <td className="py-2 px-2">
                    {p.prediction_correct ? (
                      <span className="text-xs bg-green-500/20 text-green-400 px-2 py-0.5 rounded">
                        Correct
                      </span>
                    ) : (
                      <span className="text-xs bg-red-500/20 text-red-400 px-2 py-0.5 rounded">
                        Wrong
                      </span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <div className="text-center text-gray-500 py-16">
          <p className="text-lg">No resolved predictions yet.</p>
        </div>
      )}
    </>
  );
}
