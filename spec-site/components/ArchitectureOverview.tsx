import { systemArchitecture } from "@/content/spec";

export default function ArchitectureOverview() {
  return (
    <section id="architecture" className="scroll-mt-8 space-y-4">
      {/* Header */}
      <div className="border border-violet-800/50 rounded-xl p-5 bg-gradient-to-br from-violet-950/40 to-neutral-900">
        <h2 className="text-lg font-bold text-white mb-2">System Architecture</h2>
        <p className="text-sm text-neutral-300 leading-relaxed">{systemArchitecture.overview}</p>
      </div>

      {/* Flow diagram */}
      <div className="border border-neutral-800 rounded-xl overflow-hidden">
        <div className="bg-neutral-900 px-4 py-2.5 border-b border-neutral-800">
          <span className="text-xs font-semibold uppercase tracking-widest text-neutral-400">End-to-End Control Flow</span>
        </div>
        <pre className="p-5 text-xs text-neutral-300 font-mono leading-relaxed overflow-x-auto bg-neutral-950">
          {systemArchitecture.flowDiagram}
        </pre>
      </div>

      {/* Data contracts */}
      <div className="border border-neutral-800 rounded-xl overflow-hidden">
        <div className="bg-neutral-900 px-4 py-2.5 border-b border-neutral-800">
          <span className="text-xs font-semibold uppercase tracking-widest text-neutral-400">Inter-Agent Data Contracts</span>
        </div>
        <div className="divide-y divide-neutral-800">
          {systemArchitecture.keyContracts.map((c) => (
            <div key={c.from} className="px-4 py-3 flex flex-col sm:flex-row sm:items-start gap-2">
              <code className="text-xs font-mono text-violet-300 flex-shrink-0 w-56">{c.from}</code>
              <span className="text-xs text-neutral-400">{c.data}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Observability note */}
      <div className="border border-sky-900/40 rounded-xl px-4 py-3 bg-sky-950/20 text-xs text-sky-300">
        <span className="font-semibold">Observability: </span>
        {systemArchitecture.observabilityNote}
      </div>
    </section>
  );
}
