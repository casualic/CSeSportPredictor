"use client";

import { createContext, useContext, useEffect, useState, type ReactNode } from "react";
import { supabase } from "@/lib/supabase";

interface Ranking {
  team: string;
  rank: number;
  points: number;
  date: string;
}

const RankingsContext = createContext<{
  open: boolean;
  toggle: () => void;
  rankings: Ranking[];
  loading: boolean;
  rankDate: string;
}>({ open: false, toggle: () => {}, rankings: [], loading: false, rankDate: "" });

export function RankingsProvider({ children }: { children: ReactNode }) {
  const [open, setOpen] = useState(false);
  const [rankings, setRankings] = useState<Ranking[]>([]);
  const [loading, setLoading] = useState(false);
  const [rankDate, setRankDate] = useState("");

  useEffect(() => {
    if (!open || rankings.length > 0) return;
    setLoading(true);

    (async () => {
      const { data: latest } = await supabase
        .from("rankings")
        .select("date")
        .order("date", { ascending: false })
        .limit(1);

      if (!latest || latest.length === 0) {
        setLoading(false);
        return;
      }

      const latestDate = latest[0].date;
      setRankDate(latestDate);

      const { data } = await supabase
        .from("rankings")
        .select("team, rank, points, date")
        .eq("date", latestDate)
        .lte("rank", 50)
        .order("rank", { ascending: true });

      setRankings((data as Ranking[]) ?? []);
      setLoading(false);
    })();
  }, [open, rankings.length]);

  return (
    <RankingsContext.Provider value={{ open, toggle: () => setOpen((v) => !v), rankings, loading, rankDate }}>
      {children}
    </RankingsContext.Provider>
  );
}

export function RankingsButton() {
  const { open, toggle } = useContext(RankingsContext);

  return (
    <button
      onClick={toggle}
      className={`inline-flex items-center gap-1.5 text-[12px] sm:text-[13px] font-semibold px-3 sm:px-3.5 py-1.5 rounded-[5px] border transition-all duration-150 cursor-pointer ${
        open
          ? "bg-blue-600 text-white border-blue-600 shadow-sm"
          : "bg-blue-50 text-blue-600 border-blue-200 hover:bg-blue-100 hover:border-blue-300"
      }`}
    >
      <svg
        width="14"
        height="14"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="2.5"
        strokeLinecap="round"
        strokeLinejoin="round"
      >
        <line x1="4" y1="9" x2="20" y2="9" />
        <line x1="4" y1="15" x2="20" y2="15" />
        <line x1="10" y1="3" x2="8" y2="21" />
        <line x1="16" y1="3" x2="14" y2="21" />
      </svg>
      Check Rankings
    </button>
  );
}

export function RankingsDropdown() {
  const { open, toggle, rankings, loading, rankDate } = useContext(RankingsContext);

  if (!open) return null;

  return (
    <div className="bg-white border border-slate-300 rounded-[6px] animate-[fadeIn_0.25s_ease] mb-5 sm:mb-7">
      <div className="flex items-center justify-between px-3 sm:px-4 py-2.5 border-b border-slate-200">
        <div className="flex items-baseline gap-2">
          <h3 className="font-serif text-sm sm:text-[15px] font-semibold text-slate-900">
            HLTV Rankings
          </h3>
          {rankDate && (
            <span className="font-mono text-[11px] text-slate-400">
              {rankDate}
            </span>
          )}
        </div>
        <button
          onClick={toggle}
          className="text-slate-400 hover:text-slate-600 transition-colors p-0.5 cursor-pointer"
        >
          <svg
            width="14"
            height="14"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2.5"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <line x1="18" y1="6" x2="6" y2="18" />
            <line x1="6" y1="6" x2="18" y2="18" />
          </svg>
        </button>
      </div>

      {loading ? (
        <div className="py-8 text-center text-slate-400 text-sm">
          Loading rankings...
        </div>
      ) : rankings.length === 0 ? (
        <div className="py-8 text-center text-slate-400 text-sm">
          No ranking data available.
        </div>
      ) : (
        <div className="columns-2 sm:columns-3 md:columns-5 gap-0">
          {rankings.map((r) => (
            <div
              key={r.rank}
              className="flex items-center gap-2 px-3 py-1.5 border-b border-r border-slate-100 break-inside-avoid"
            >
              <span className="font-mono text-[11px] text-slate-600 font-bold w-5 text-right shrink-0">
                {r.rank}
              </span>
              <span className="text-[12px] sm:text-[13px] text-slate-700 font-medium truncate">
                {r.team}
              </span>
              <span className="font-mono text-[10px] text-slate-400 ml-auto shrink-0">
                {r.points}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
