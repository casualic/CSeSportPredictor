"use client";

import { useRouter, useSearchParams, usePathname } from "next/navigation";

const TIERS = [
  { label: "All", value: "all" },
  { label: "Top 15", value: "15" },
  { label: "Top 30", value: "30" },
  { label: "Top 50", value: "50" },
];

export default function TierFilter() {
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();
  const current = searchParams.get("tier") || "all";

  function setTier(value: string) {
    const params = new URLSearchParams(searchParams.toString());
    if (value === "all") {
      params.delete("tier");
    } else {
      params.set("tier", value);
    }
    router.push(`${pathname}?${params.toString()}`);
  }

  return (
    <div className="flex items-center gap-1">
      <span className="text-xs text-gray-500 mr-1">Tier:</span>
      {TIERS.map((t) => (
        <button
          key={t.value}
          onClick={() => setTier(t.value)}
          className={`text-xs px-2 py-1 rounded transition-colors ${
            current === t.value
              ? "bg-gray-700 text-white"
              : "text-gray-400 hover:text-white hover:bg-gray-800/50"
          }`}
        >
          {t.label}
        </button>
      ))}
    </div>
  );
}
