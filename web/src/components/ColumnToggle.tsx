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
            className={`p-2 rounded-md transition-colors ${
              cols === 2
                ? "bg-emerald-600 text-white ring-2 ring-emerald-400/50"
                : "text-gray-400 bg-gray-800/50 hover:text-white hover:bg-gray-700"
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
            className={`p-2 rounded-md transition-colors ${
              cols === 1
                ? "bg-emerald-600 text-white ring-2 ring-emerald-400/50"
                : "text-gray-400 bg-gray-800/50 hover:text-white hover:bg-gray-700"
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
