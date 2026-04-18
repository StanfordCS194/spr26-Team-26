import Header from "@/components/Header";
import Sidebar from "@/components/Sidebar";
import FeatureSection from "@/components/FeatureSection";
import TypesSection from "@/components/TypesSection";
import { features } from "@/content/spec";

export default function Home() {
  const totalFunctions = features.reduce((n, f) => n + f.functions.length, 0);

  return (
    <div className="min-h-screen flex flex-col">
      <Header />
      <div className="flex flex-1">
        <Sidebar />
        <main className="flex-1 min-w-0 px-4 md:px-8 py-8 max-w-4xl">
          {/* Summary bar */}
          <div className="mb-8 flex flex-wrap gap-4 text-sm">
            <div className="border border-neutral-800 rounded-lg px-4 py-2.5 bg-neutral-900/40">
              <p className="text-xs text-neutral-500 mb-0.5">Features</p>
              <p className="font-semibold text-white">{features.length}</p>
            </div>
            <div className="border border-neutral-800 rounded-lg px-4 py-2.5 bg-neutral-900/40">
              <p className="text-xs text-neutral-500 mb-0.5">Total Functions</p>
              <p className="font-semibold text-white">{totalFunctions}</p>
            </div>
            <div className="border border-neutral-800 rounded-lg px-4 py-2.5 bg-neutral-900/40 flex-1 min-w-48">
              <p className="text-xs text-neutral-500 mb-0.5">Edit spec</p>
              <p className="text-xs text-neutral-400 font-mono">spec-site/content/spec.ts</p>
            </div>
          </div>

          <div className="space-y-12">
            <TypesSection />
            {features.map((feature) => (
              <FeatureSection key={feature.id} feature={feature} />
            ))}
          </div>

          <footer className="mt-12 pt-6 border-t border-neutral-800 text-center text-xs text-neutral-700">
            CS194 · Team 26 · Stanford University · Spring 2026
          </footer>
        </main>
      </div>
    </div>
  );
}
