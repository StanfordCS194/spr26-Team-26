import Header from "@/components/Header";
import Sidebar from "@/components/Sidebar";
import SectionCard from "@/components/SectionCard";
import { sections } from "@/content/spec";

export default function Home() {
  return (
    <div className="min-h-screen flex flex-col">
      <Header />
      <div className="flex flex-1">
        <Sidebar />
        <main className="flex-1 min-w-0 px-4 md:px-8 py-8 max-w-4xl mx-auto w-full">
          {/* Hero banner */}
          <div className="mb-10 rounded-xl border border-violet-800/50 bg-gradient-to-br from-violet-950/60 via-indigo-950/40 to-neutral-900 p-6 md:p-8">
            <div className="flex items-center gap-2 mb-3">
              <span className="text-xs font-mono text-violet-400 tracking-widest uppercase">
                Stanford University · CS194 · Spring 2026
              </span>
            </div>
            <h1 className="text-2xl md:text-3xl font-bold text-white mb-3 leading-tight">
              Autonomous ML Training Agent
            </h1>
            <p className="text-neutral-300 text-base leading-relaxed max-w-2xl">
              An end-to-end system that takes a plain-English prompt and a hard budget cap,
              then automatically handles data discovery, model selection, hyperparameter tuning,
              and training orchestration — with zero infrastructure setup required.
            </p>
            <div className="mt-5 flex flex-wrap gap-2">
              {[
                "Manager Agent",
                "Data Generator",
                "Decision Engine",
                "AutoResearch Loop",
                "Cost Manager",
                "Observability",
              ].map((f) => (
                <span
                  key={f}
                  className="text-xs bg-white/5 border border-white/10 text-neutral-300 px-2.5 py-1 rounded-full"
                >
                  {f}
                </span>
              ))}
            </div>
          </div>

          {/* Spec sections */}
          <div className="space-y-6">
            {sections.map((section) => (
              <SectionCard key={section.id} section={section} />
            ))}
          </div>

          <footer className="mt-12 pt-6 border-t border-neutral-800 text-center text-xs text-neutral-600">
            CS194 · Team 26 · Stanford University · Spring 2026
          </footer>
        </main>
      </div>
    </div>
  );
}
