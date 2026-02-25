interface StatsCardProps {
  title: string;
  value: string;
  subtitle?: string;
  color?: "default" | "green" | "red" | "blue";
}

export default function StatsCard({ title, value, subtitle, color = "default" }: StatsCardProps) {
  const colorClass =
    color === "green"
      ? "text-green-600"
      : color === "red"
        ? "text-red-600"
        : color === "blue"
          ? "text-blue-600"
          : "text-slate-900";

  return (
    <div className="bg-white border border-slate-200 rounded-[6px] px-5 py-4 animate-[fadeIn_0.35s_ease_both]">
      <div className="text-[11px] font-medium text-slate-400 uppercase tracking-[0.5px] mb-1.5">
        {title}
      </div>
      <div className={`font-serif text-[28px] font-semibold leading-none ${colorClass}`}>
        {value}
      </div>
      {subtitle && (
        <div className="text-xs text-slate-400 mt-[3px]">{subtitle}</div>
      )}
    </div>
  );
}
