import Link from "next/link";
import { notFound } from "next/navigation";
import { supabase } from "@/lib/supabase";
import { Prediction } from "@/lib/types";
import ProbabilityBar from "@/components/ProbabilityBar";

export const revalidate = 0;

export default async function MatchDetail({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  const { data, error } = await supabase
    .from("predictions")
    .select("*")
    .eq("id", id)
    .single();

  if (error || !data) {
    notFound();
  }

  const p = data as Prediction;

  return (
    <>
      <nav className="text-sm text-gray-500 mb-4">
        <Link href="/" className="hover:text-white">Dashboard</Link>
        <span className="mx-2">/</span>
        <span className="text-gray-300">{p.team1} vs {p.team2}</span>
      </nav>

      <div className="bg-gray-900 rounded-lg border border-gray-800">
        <div className="px-6 py-3 border-b border-gray-800 flex justify-between items-center">
          <span className="text-gray-300">{p.event || "Match"}</span>
          <div className="flex items-center gap-2">
            <span className="text-xs text-gray-500">
              {(() => {
                const dateStr = p.match_date || p.created_at;
                const date = dateStr ? new Date(dateStr).toLocaleDateString("en-US", { month: "short", day: "numeric" }) : "";
                return date ? `${date}${p.match_time ? ` · ${p.match_time}` : ""}` : p.match_time || "";
              })()}
            </span>
            <span className="text-xs bg-gray-800 text-gray-300 px-2 py-0.5 rounded">{p.bo_format || "BO3"}</span>
          </div>
        </div>

        <div className="p-6">
          <div className="grid grid-cols-5 items-center text-center mb-6">
            <div className="col-span-2">
              <h3 className={`text-2xl font-bold ${p.predicted_winner === p.team1 ? "text-green-500" : "text-white"}`}>
                {p.team1}
              </h3>
              {p.odds_t1 && <div className="text-gray-500 text-sm">Odds: {p.odds_t1.toFixed(2)}</div>}
            </div>
            <div className="text-xl text-gray-500">vs</div>
            <div className="col-span-2">
              <h3 className={`text-2xl font-bold ${p.predicted_winner === p.team2 ? "text-green-500" : "text-white"}`}>
                {p.team2}
              </h3>
              {p.odds_t2 && <div className="text-gray-500 text-sm">Odds: {p.odds_t2.toFixed(2)}</div>}
            </div>
          </div>

          <ProbabilityBar t1Prob={p.t1_win_prob} team1={p.team1} team2={p.team2} height="h-10" />

          <div className="grid md:grid-cols-2 gap-6 mt-6">
            <div>
              <table className="w-full text-sm">
                <tbody>
                  <Row label="Predicted Winner" value={p.predicted_winner || "-"} />
                  <Row label="FSVM Probability (T1)" value={`${(p.fsvm_prob * 100).toFixed(1)}%`} />
                  <Row label="XGB Probability (T1)" value={`${(p.xgb_prob * 100).toFixed(1)}%`} />
                  <Row label="Ensemble (T1)" value={`${(p.t1_win_prob * 100).toFixed(1)}%`} />
                  <Row label="Models Agree" value={p.models_agree ? "Yes" : "No"} />
                </tbody>
              </table>
            </div>
            <div>
              <table className="w-full text-sm">
                <tbody>
                  {p.odds_t1 && (
                    <>
                      <Row
                        label="Implied Prob T1"
                        value={p.implied_prob_t1 ? `${(p.implied_prob_t1 * 100).toFixed(1)}%` : "-"}
                      />
                      <Row
                        label="Implied Prob T2"
                        value={p.implied_prob_t2 ? `${(p.implied_prob_t2 * 100).toFixed(1)}%` : "-"}
                      />
                      <Row
                        label="Edge"
                        value={p.edge !== null ? `${(p.edge * 100).toFixed(1)}%` : "-"}
                        valueClass={p.edge && p.edge > 0 ? "text-green-500" : p.edge ? "text-red-500" : ""}
                      />
                    </>
                  )}
                  {p.actual_winner && (
                    <>
                      <Row label="Actual Winner" value={p.actual_winner} />
                      <Row
                        label="Result"
                        value={p.prediction_correct ? "Correct" : "Wrong"}
                        valueClass={p.prediction_correct ? "text-green-500" : "text-red-500"}
                      />
                    </>
                  )}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>

      {p.match_url && (
        <a
          href={p.match_url}
          target="_blank"
          rel="noopener noreferrer"
          className="inline-block mt-4 text-sm text-gray-400 border border-gray-700 px-3 py-1.5 rounded hover:text-white hover:border-gray-500 transition-colors"
        >
          View on HLTV
        </a>
      )}
    </>
  );
}

function Row({ label, value, valueClass = "" }: { label: string; value: string; valueClass?: string }) {
  return (
    <tr className="border-b border-gray-800/50">
      <th className="text-left py-2 text-gray-400 font-medium">{label}</th>
      <td className={`text-right py-2 ${valueClass}`}>{value}</td>
    </tr>
  );
}
