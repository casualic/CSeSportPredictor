import Link from "next/link";
import { Prediction } from "@/lib/types";
import ProbabilityBar from "./ProbabilityBar";

interface PredictionCardProps {
  prediction: Prediction;
}

export default function PredictionCard({ prediction: p }: PredictionCardProps) {
  const edgePct = p.edge ? (p.edge * 100).toFixed(1) : null;

  return (
    <div className="bg-gray-900 rounded-lg border border-gray-800 hover:-translate-y-0.5 transition-transform">
      <div className="px-4 py-2 border-b border-gray-800 flex justify-between items-center">
        <span className="text-sm text-gray-400 truncate">{p.event || "Unknown Event"}</span>
        <span className="text-xs bg-gray-800 text-gray-300 px-2 py-0.5 rounded">
          {p.bo_format || "BO3"}
        </span>
      </div>
      <div className="p-4">
        <div className="flex justify-between items-center mb-3">
          <div className="text-center flex-1">
            <div className={`font-bold text-lg ${p.predicted_winner === p.team1 ? "text-green-500" : "text-white"}`}>
              {p.team1}
            </div>
            {p.odds_t1 && <span className="text-xs text-gray-500">{p.odds_t1.toFixed(2)}</span>}
          </div>
          <div className="px-3 text-gray-500">vs</div>
          <div className="text-center flex-1">
            <div className={`font-bold text-lg ${p.predicted_winner === p.team2 ? "text-green-500" : "text-white"}`}>
              {p.team2}
            </div>
            {p.odds_t2 && <span className="text-xs text-gray-500">{p.odds_t2.toFixed(2)}</span>}
          </div>
        </div>

        <ProbabilityBar t1Prob={p.t1_win_prob} team1={p.team1} team2={p.team2} />

        <div className="grid grid-cols-3 text-center text-xs mt-3">
          <div>
            <div className="text-gray-500">FSVM</div>
            <div className="text-gray-300">{(p.fsvm_prob * 100).toFixed(1)}%</div>
          </div>
          <div>
            <div className="text-gray-500">XGB</div>
            <div className="text-gray-300">{(p.xgb_prob * 100).toFixed(1)}%</div>
          </div>
          <div>
            <div className="text-gray-500">Agree</div>
            <div className="text-gray-300">{p.models_agree ? "Yes" : "No"}</div>
          </div>
        </div>

        {edgePct && (
          <div className="flex justify-center mt-3">
            <span
              className={`text-xs px-2 py-0.5 rounded font-medium ${
                p.edge! > 0.05
                  ? "bg-green-500/20 text-green-400"
                  : p.edge! > 0
                    ? "bg-yellow-500/20 text-yellow-400"
                    : "bg-red-500/20 text-red-400"
              }`}
            >
              Edge: {edgePct}%
            </span>
          </div>
        )}
      </div>
      <div className="px-4 py-2 border-t border-gray-800 flex justify-between items-center">
        <span className="text-xs text-gray-500">
          {(() => {
            const dateStr = p.match_date || p.created_at;
            const date = dateStr ? new Date(dateStr).toLocaleDateString("en-US", { month: "short", day: "numeric" }) : "";
            return date ? `${date}${p.match_time ? ` · ${p.match_time}` : ""}` : p.match_time || "";
          })()}
        </span>
        <Link
          href={`/match/${p.id}`}
          className="text-xs text-gray-400 hover:text-white border border-gray-700 px-2 py-1 rounded transition-colors"
        >
          Details
        </Link>
      </div>
    </div>
  );
}
