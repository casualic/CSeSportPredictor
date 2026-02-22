import { Suspense } from "react";
import { supabase } from "@/lib/supabase";
import { Prediction } from "@/lib/types";
import { filterByTier } from "@/lib/tier";
import PredictionCard from "@/components/PredictionCard";
import TierFilter from "@/components/TierFilter";

export const revalidate = 0;

export default async function Dashboard({ searchParams }: { searchParams: Promise<{ tier?: string }> }) {
  const { tier } = await searchParams;
  const { data: predictions } = await supabase
    .from("predictions")
    .select("*")
    .is("actual_winner", null)
    .order("created_at", { ascending: false });

  const items = filterByTier((predictions ?? []) as Prediction[], tier ?? null);

  return (
    <>
      <div className="flex justify-between items-center mb-6">
        <h2 className="text-2xl font-bold">Upcoming Predictions</h2>
        <Suspense><TierFilter /></Suspense>
      </div>

      {items.length > 0 ? (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {items.map((p) => (
            <PredictionCard key={p.id} prediction={p} />
          ))}
        </div>
      ) : (
        <div className="text-center text-gray-500 py-16">
          <p className="text-lg">No upcoming predictions yet.</p>
          <p className="text-sm mt-2">
            Run the scraper to add predictions from HLTV.
          </p>
        </div>
      )}
    </>
  );
}
