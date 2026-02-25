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

  const dateStr = p.match_date || p.created_at;
  const dateFormatted = dateStr
    ? new Date(dateStr).toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" })
    : "";
  const timeStr = p.match_time ? ` · ${p.match_time}` : "";

  return (
    <>
      {/* Breadcrumb */}
      <nav className="text-sm text-slate-400 mb-5">
        <Link href="/" className="hover:text-blue-600 transition-colors">
          Dashboard
        </Link>
        <span className="mx-2">/</span>
        <span className="text-slate-600">
          {p.team1} vs {p.team2}
        </span>
      </nav>

      {/* Result banner if resolved */}
      {p.actual_winner && (
        <div
          className={`rounded-[6px] border px-3 sm:px-5 py-3 mb-5 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-2 sm:gap-0 ${
            p.prediction_correct
              ? "bg-green-50 border-green-200"
              : "bg-red-50 border-red-200"
          }`}
        >
          <div className="flex items-center gap-3">
            <span
              className={`text-[11px] font-semibold px-2.5 py-[2px] rounded-[3px] ${
                p.prediction_correct
                  ? "bg-white text-green-600"
                  : "bg-white text-red-600"
              }`}
            >
              {p.prediction_correct ? "Correct" : "Wrong"}
            </span>
            <span className="text-sm text-slate-600">
              Winner: <strong className="text-slate-900">{p.actual_winner}</strong>
            </span>
          </div>
          <span className="text-xs text-slate-400">
            {p.resolved_at
              ? new Date(p.resolved_at).toLocaleDateString("en-US", {
                  month: "short",
                  day: "numeric",
                })
              : ""}
          </span>
        </div>
      )}

      {/* Header card */}
      <div className="bg-white border border-slate-200 rounded-[6px] overflow-hidden">
        {/* Event header */}
        <div className="px-4 sm:px-6 py-3 border-b border-slate-200 flex flex-col sm:flex-row justify-between items-start sm:items-center gap-1 sm:gap-0 bg-slate-50">
          <span className="text-slate-600 text-sm font-medium">
            {p.event || "Match"}
          </span>
          <div className="flex items-center gap-2">
            <span className="text-xs text-slate-400">
              {dateFormatted}{timeStr}
            </span>
            <span className="text-[11px] font-semibold text-slate-400 bg-white border border-slate-200 px-2 py-[1px] rounded-[3px]">
              {p.bo_format || "BO3"}
            </span>
          </div>
        </div>

        <div className="p-4 sm:p-6">
          {/* Two-column grid */}
          <div className="grid md:grid-cols-2 gap-6 sm:gap-8">
            {/* Left: Team matchup + probability */}
            <div>
              <div className="flex items-center justify-between mb-5">
                <div className="text-center flex-1">
                  <h3
                    className={`text-xl sm:text-2xl font-bold font-serif ${
                      p.predicted_winner === p.team1 ? "text-blue-600" : "text-slate-900"
                    }`}
                  >
                    {p.team1}
                  </h3>
                  {p.t1_rank && (
                    <span className="font-mono text-[11px] text-slate-400">#{p.t1_rank}</span>
                  )}
                  {p.odds_t1 && (
                    <div className="text-slate-400 text-xs mt-1">
                      Odds: {p.odds_t1.toFixed(2)}
                    </div>
                  )}
                </div>
                <div className="text-lg text-slate-400 font-medium px-4">vs</div>
                <div className="text-center flex-1">
                  <h3
                    className={`text-xl sm:text-2xl font-bold font-serif ${
                      p.predicted_winner === p.team2 ? "text-blue-600" : "text-slate-900"
                    }`}
                  >
                    {p.team2}
                  </h3>
                  {p.t2_rank && (
                    <span className="font-mono text-[11px] text-slate-400">#{p.t2_rank}</span>
                  )}
                  {p.odds_t2 && (
                    <div className="text-slate-400 text-xs mt-1">
                      Odds: {p.odds_t2.toFixed(2)}
                    </div>
                  )}
                </div>
              </div>

              <ProbabilityBar t1Prob={p.t1_win_prob} team1={p.team1} team2={p.team2} height="h-10" />

              <div className="mt-3 text-center text-[13px] text-slate-400">
                Predicted winner:{" "}
                <span className="text-blue-600 font-semibold">{p.predicted_winner}</span>
              </div>
            </div>

            {/* Right: Model outputs table */}
            <div>
              <h4 className="text-[11px] font-semibold text-slate-400 uppercase tracking-[0.5px] mb-3">
                Model Outputs
              </h4>
              <div className="border border-slate-200 rounded-[6px] overflow-hidden">
                <table className="w-full border-collapse text-sm">
                  <tbody>
                    <Row label="FSVM Probability (T1)" value={`${(p.fsvm_prob * 100).toFixed(1)}%`} />
                    <Row label="XGB Probability (T1)" value={`${(p.xgb_prob * 100).toFixed(1)}%`} />
                    <Row label="Ensemble (T1)" value={`${(p.t1_win_prob * 100).toFixed(1)}%`} highlight />
                    <Row
                      label="Models Agree"
                      value={p.models_agree ? "Yes" : "No"}
                      valueClass={p.models_agree ? "text-green-600" : "text-red-600"}
                    />
                    <Row label="Confidence" value={`${(p.confidence * 100).toFixed(1)}%`} />
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          {/* Bookmaker data */}
          {p.odds_t1 && (
            <div className="mt-8">
              <h4 className="text-[11px] font-semibold text-slate-400 uppercase tracking-[0.5px] mb-3">
                Bookmaker &amp; Edge Analysis
              </h4>
              <div className="grid md:grid-cols-2 gap-4">
                <div className="border border-slate-200 rounded-[6px] overflow-hidden">
                  <table className="w-full border-collapse text-sm">
                    <tbody>
                      <Row label={`${p.team1} Odds`} value={p.odds_t1?.toFixed(2) ?? "-"} />
                      <Row label={`${p.team2} Odds`} value={p.odds_t2?.toFixed(2) ?? "-"} />
                      <Row
                        label="Implied Prob T1"
                        value={p.implied_prob_t1 ? `${(p.implied_prob_t1 * 100).toFixed(1)}%` : "-"}
                      />
                      <Row
                        label="Implied Prob T2"
                        value={p.implied_prob_t2 ? `${(p.implied_prob_t2 * 100).toFixed(1)}%` : "-"}
                      />
                    </tbody>
                  </table>
                </div>
                <div className="border border-slate-200 rounded-[6px] overflow-hidden">
                  <table className="w-full border-collapse text-sm">
                    <tbody>
                      <Row
                        label="Edge"
                        value={p.edge !== null ? `${(p.edge * 100).toFixed(1)}%` : "-"}
                        valueClass={
                          p.edge && p.edge > 0.05
                            ? "text-green-600 font-semibold"
                            : p.edge && p.edge > 0
                              ? "text-amber-600"
                              : p.edge
                                ? "text-red-600"
                                : ""
                        }
                      />
                      <Row
                        label="Edge Signal"
                        value={
                          p.edge !== null
                            ? p.edge > 0.05
                              ? "Strong positive"
                              : p.edge > 0
                                ? "Weak positive"
                                : "Negative"
                            : "-"
                        }
                      />
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* HLTV link */}
      {p.match_url && (
        <a
          href={p.match_url}
          target="_blank"
          rel="noopener noreferrer"
          className="inline-block mt-4 text-xs font-medium text-blue-600 border border-blue-100 px-3 py-1.5 rounded-[4px] hover:bg-blue-50 hover:border-blue-500 transition-all duration-150 no-underline"
        >
          View on HLTV
        </a>
      )}
    </>
  );
}

function Row({
  label,
  value,
  valueClass = "",
  highlight = false,
}: {
  label: string;
  value: string;
  valueClass?: string;
  highlight?: boolean;
}) {
  return (
    <tr className="border-b border-slate-100 last:border-b-0">
      <th className="text-left py-2.5 px-3 sm:px-4 text-slate-600 font-medium text-[12px] sm:text-[13px]">
        {label}
      </th>
      <td
        className={`text-right py-2.5 px-3 sm:px-4 font-mono text-[12px] sm:text-[13px] ${
          highlight ? "text-blue-600 font-semibold" : valueClass || "text-slate-900"
        }`}
      >
        {value}
      </td>
    </tr>
  );
}
