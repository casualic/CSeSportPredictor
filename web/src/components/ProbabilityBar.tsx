interface ProbabilityBarProps {
  t1Prob: number;
  team1: string;
  team2: string;
  height?: string;
}

export default function ProbabilityBar({ t1Prob, team1, team2, height = "h-7" }: ProbabilityBarProps) {
  const t1Pct = (t1Prob * 100).toFixed(1);
  const t2Pct = ((1 - t1Prob) * 100).toFixed(1);

  return (
    <div className={`flex ${height} rounded-md overflow-hidden text-xs font-semibold`}>
      <div
        className="bg-gradient-to-r from-green-600 to-green-500 flex items-center justify-center text-white min-w-[30px]"
        style={{ width: `${t1Pct}%` }}
      >
        {team1} {t1Pct}%
      </div>
      <div
        className="bg-gradient-to-r from-red-500 to-red-600 flex items-center justify-center text-white min-w-[30px]"
        style={{ width: `${t2Pct}%` }}
      >
        {t2Pct}% {team2}
      </div>
    </div>
  );
}
