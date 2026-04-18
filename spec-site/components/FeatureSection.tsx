import FunctionCard from "./FunctionCard";
import { FeatureSpec } from "@/content/spec";

const OWNER_COLORS: Record<string, string> = {
  "Sid Potti": "bg-blue-900/40 text-blue-300 border-blue-800",
  "Matthew Torre": "bg-green-900/40 text-green-300 border-green-800",
  "Ron Polonsky": "bg-orange-900/40 text-orange-300 border-orange-800",
  "Angel Raychev": "bg-rose-900/40 text-rose-300 border-rose-800",
  "Hayley Antczak": "bg-yellow-900/40 text-yellow-300 border-yellow-800",
  "Team": "bg-violet-900/40 text-violet-300 border-violet-800",
  "Matthew Torre, Hayley Antczak": "bg-teal-900/40 text-teal-300 border-teal-800",
  "Ron Polonsky, Angel Raychev": "bg-amber-900/40 text-amber-300 border-amber-800",
};

export default function FeatureSection({ feature }: { feature: FeatureSpec }) {
  const ownerClass = OWNER_COLORS[feature.owner] ?? "bg-neutral-800 text-neutral-300 border-neutral-700";

  return (
    <section id={feature.id} className="scroll-mt-8 space-y-4">
      {/* Feature header */}
      <div className="border border-neutral-800 rounded-xl p-5 bg-neutral-900/60">
        <div className="flex flex-wrap items-start justify-between gap-3 mb-2">
          <h2 className="text-lg font-bold text-white">{feature.title}</h2>
          <span className={`inline-flex items-center gap-1.5 text-xs font-medium px-2.5 py-1 rounded-full border ${ownerClass}`}>
            <svg className="w-3 h-3" fill="currentColor" viewBox="0 0 20 20">
              <path d="M10 9a3 3 0 100-6 3 3 0 000 6zm-7 9a7 7 0 1114 0H3z" />
            </svg>
            {feature.owner}
          </span>
        </div>
        <p className="text-sm text-neutral-400 leading-relaxed">{feature.description}</p>
      </div>

      {/* Architecture / control flow */}
      <div className="border border-neutral-800 rounded-xl overflow-hidden">
        <div className="bg-neutral-900/80 px-4 py-2.5 border-b border-neutral-800">
          <span className="text-xs font-semibold uppercase tracking-widest text-neutral-400">How it works</span>
        </div>
        <div className="p-5 space-y-4 bg-neutral-950/40">
          <p className="text-sm text-neutral-300 leading-relaxed whitespace-pre-wrap">{feature.architecture}</p>
          {feature.flowDiagram && (
            <div>
              <p className="text-xs font-semibold uppercase tracking-widest text-neutral-500 mb-2">Control Flow</p>
              <pre className="text-xs text-neutral-300 font-mono leading-relaxed overflow-x-auto bg-neutral-900 border border-neutral-800 rounded-lg p-4">
                {feature.flowDiagram}
              </pre>
            </div>
          )}
        </div>
      </div>

      {/* Function reference */}
      <div>
        <p className="text-xs font-semibold uppercase tracking-widest text-neutral-500 mb-3 px-1">
          Functions ({feature.functions.length})
        </p>
        <div className="space-y-3 pl-3 border-l-2 border-neutral-800">
          {feature.functions.map((fn) => (
            <div key={fn.name} id={`fn-${fn.name}`} className="scroll-mt-8">
              <div className="flex items-center gap-2 mb-1.5">
                <span className="text-xs font-mono font-semibold text-neutral-300">{fn.name}</span>
              </div>
              <FunctionCard fn={fn} />
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
