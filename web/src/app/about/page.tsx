import fs from "fs";
import path from "path";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { Components } from "react-markdown";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "About — CS2 Predictor",
};

/* Map image paths from the markdown to /writeup/ public folder */
const imageMap: Record<string, string> = {
  "screenshot-dashboard.png": "/writeup/screenshot-dashboard.png",
  "results/upset_accuracy_by_tier.png": "/writeup/upset_accuracy_by_tier.png",
  "results/tier_comparison_16_vs_18.png":
    "/writeup/tier_comparison_16_vs_18.png",
};

const components: Components = {
  img: ({ src, alt, ...props }) => {
    const srcStr = typeof src === "string" ? src : undefined;
    const mapped = srcStr ? imageMap[srcStr] ?? srcStr : srcStr;
    return (
      // eslint-disable-next-line @next/next/no-img-element
      <img
        src={mapped}
        alt={alt ?? ""}
        className="rounded-lg border border-slate-200 shadow-sm"
        {...props}
      />
    );
  },
};

export default function WriteupPage() {
  const md = fs.readFileSync(
    path.join(process.cwd(), "PROJECT_WRITEUP.md"),
    "utf-8"
  );

  return (
    <div className="max-w-[820px] mx-auto px-4 sm:px-6 py-8 sm:py-12">
      <article className="prose prose-slate prose-sm sm:prose-base max-w-none prose-headings:font-serif prose-headings:tracking-tight prose-h1:text-2xl prose-h1:sm:text-3xl prose-a:text-blue-600 prose-img:mx-auto prose-table:text-xs sm:prose-table:text-sm prose-th:px-3 prose-th:py-1.5 prose-td:px-3 prose-td:py-1.5">
        <ReactMarkdown remarkPlugins={[remarkGfm]} components={components}>
          {md}
        </ReactMarkdown>
      </article>
    </div>
  );
}
