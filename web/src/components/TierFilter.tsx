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
    params.set("tier", value);
    router.push(`${pathname}?${params.toString()}`);
  }

  return (
    <div className="flex items-center gap-1.5">
      {TIERS.map((t) => (
        <button
          key={t.value}
          onClick={() => setTier(t.value)}
          className={`text-xs font-medium px-3.5 py-1.5 rounded-[4px] border transition-all duration-150 cursor-pointer ${
            current === t.value
              ? "bg-blue-600 border-blue-600 text-white"
              : "bg-white border-slate-200 text-slate-600 hover:border-blue-600 hover:text-blue-600"
          }`}
        >
          {t.label}
        </button>
      ))}
    </div>
  );
}
