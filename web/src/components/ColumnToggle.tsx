"use client";

import { useState } from "react";

interface ColumnToggleProps {
  children: React.ReactNode;
}

export default function ColumnToggle({ children }: ColumnToggleProps) {
  const [cols, setCols] = useState<1 | 2>(2);

  return (
    <>
      <div className="flex justify-end mb-3">
        <div className="flex items-center gap-1">
          <button
            onClick={() => setCols(2)}
            className={`p-2 rounded-[4px] transition-colors border ${
              cols === 2
                ? "bg-blue-600 text-white border-blue-600"
                : "text-slate-400 bg-white border-slate-200 hover:text-slate-600 hover:border-blue-600"
            }`}
            aria-label="Two columns"
          >
            <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
              <rect x="1" y="1" width="6" height="6" rx="1" fill="currentColor" />
              <rect x="9" y="1" width="6" height="6" rx="1" fill="currentColor" />
              <rect x="1" y="9" width="6" height="6" rx="1" fill="currentColor" />
              <rect x="9" y="9" width="6" height="6" rx="1" fill="currentColor" />
            </svg>
          </button>
          <button
            onClick={() => setCols(1)}
            className={`p-2 rounded-[4px] transition-colors border ${
              cols === 1
                ? "bg-blue-600 text-white border-blue-600"
                : "text-slate-400 bg-white border-slate-200 hover:text-slate-600 hover:border-blue-600"
            }`}
            aria-label="One column"
          >
            <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
              <rect x="1" y="1" width="14" height="4" rx="1" fill="currentColor" />
              <rect x="1" y="7" width="14" height="4" rx="1" fill="currentColor" />
              <rect x="1" y="13" width="14" height="2" rx="1" fill="currentColor" />
            </svg>
          </button>
        </div>
      </div>
      <div className={`grid gap-4 ${cols === 2 ? "grid-cols-1 md:grid-cols-2" : "grid-cols-1"}`}>
        {children}
      </div>
    </>
  );
}
