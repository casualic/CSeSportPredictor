"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const links = [
  { href: "/", label: "Dashboard" },
  { href: "/results", label: "Results" },
  { href: "/pnl", label: "P&L" },
  { href: "/about", label: "About" },
];

export default function Navbar() {
  const pathname = usePathname();

  return (
    <nav className="bg-white border-b border-slate-200 sticky top-0 z-50">
      <div className="max-w-[1120px] mx-auto px-3 sm:px-7 h-12 sm:h-14 flex items-center justify-between gap-2">
        <Link
          href="/"
          className="font-serif text-[15px] sm:text-lg font-semibold text-slate-900 no-underline tracking-[-0.3px] shrink-0 whitespace-nowrap"
        >
          CS2 <span className="text-blue-600">Predictor</span>
        </Link>

        <div className="flex gap-[1px] bg-slate-100 rounded-[6px] p-[3px]">
          {links.map((link) => (
            <Link
              key={link.href}
              href={link.href}
              className={`text-[11px] sm:text-[13px] font-medium px-2 sm:px-4 py-1 sm:py-1.5 rounded-[4px] no-underline transition-all duration-150 whitespace-nowrap ${
                pathname === link.href
                  ? "text-blue-600 bg-white shadow-[0_1px_2px_rgba(0,0,0,0.06)]"
                  : "text-slate-400 hover:text-slate-600"
              }`}
            >
              {link.label}
            </Link>
          ))}
        </div>

        <div className="hidden sm:flex font-mono text-xs text-slate-400 items-center gap-1.5">
          FSVM+XGB &middot; <span className="text-blue-600 font-medium">66.5%</span> WF
        </div>
      </div>
    </nav>
  );
}
