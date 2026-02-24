"use client";

import { useRouter, useSearchParams, usePathname } from "next/navigation";

const TIERS = [
  { label: "All", value: "all" },
  { label: "Top 15", value: "15" },
  { label: "Top 30", value: "30" },
  { label: "Top 50", value: "50" },
];

export default function TierFilter({ defaultTier = "all" }: { defaultTier?: string }) {
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();
  const current = searchParams.get("tier") || defaultTier;

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
    <div className="flex items-center gap-1.5">
      <span className="text-sm text-gray-400 font-medium mr-1">Filter:</span>
      {TIERS.map((t) => (
        <button
          key={t.value}
          onClick={() => setTier(t.value)}
          className={`text-sm px-4 py-2 rounded-md font-medium transition-colors ${
            current === t.value
              ? "bg-emerald-600 text-white ring-2 ring-emerald-400/50"
              : "text-gray-400 hover:text-white bg-gray-800/50 hover:bg-gray-700"
          }`}
        >
          {t.label}
        </button>
      ))}
    </div>
  );
}
