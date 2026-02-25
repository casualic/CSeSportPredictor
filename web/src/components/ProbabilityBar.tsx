interface ProbabilityBarProps {
  t1Prob: number;
  team1: string;
  team2: string;
  height?: string;
}

export default function ProbabilityBar({ t1Prob, team1, team2, height }: ProbabilityBarProps) {
  const t1Pct = (t1Prob * 100).toFixed(1);
  const t2Pct = ((1 - t1Prob) * 100).toFixed(1);

  if (height) {
    /* Large variant used on match detail page */
    return (
      <div>
        <div className={`flex ${height} rounded-[6px] overflow-hidden text-xs font-semibold ring-2 ring-blue-600`}>
          <div
            className="bg-emerald-500 flex items-center justify-center text-white min-w-[40px]"
            style={{ width: `${t1Pct}%` }}
          >
            {team1} {t1Pct}%
          </div>
          <div
            className="bg-red-400 flex items-center justify-center text-white min-w-[40px]"
            style={{ width: `${t2Pct}%` }}
          >
            {t2Pct}% {team2}
          </div>
        </div>
      </div>
    );
  }

  /* Compact bar for table cells */
  return (
    <div className="w-[180px]">
      <div className="flex h-1.5 rounded-[3px] overflow-hidden ring-1 ring-blue-600">
        <div
          className="bg-emerald-500 rounded-l-[3px] transition-[width] duration-600 ease-out"
          style={{ width: `${t1Pct}%` }}
        />
        <div className="bg-red-400 flex-1 rounded-r-[3px]" />
      </div>
      <div className="flex justify-between mt-1 font-mono text-[11px]">
        <span className="text-emerald-600 font-medium">{t1Pct}%</span>
        <span className="text-red-500">{t2Pct}%</span>
      </div>
    </div>
  );
}
