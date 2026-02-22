interface StatsCardProps {
  title: string;
  value: string;
  subtitle?: string;
  color?: "default" | "green" | "red";
}

export default function StatsCard({ title, value, subtitle, color = "default" }: StatsCardProps) {
  const colorClass =
    color === "green" ? "text-green-500" : color === "red" ? "text-red-500" : "text-white";

  return (
    <div className="bg-gray-900 rounded-lg border border-gray-800 p-4 text-center">
      <div className="text-sm text-gray-400 mb-1">{title}</div>
      <div className={`text-3xl font-bold ${colorClass}`}>{value}</div>
      {subtitle && <div className="text-xs text-gray-500 mt-1">{subtitle}</div>}
    </div>
  );
}
