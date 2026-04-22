import { systemArchitecture } from "@/content/spec";

export default function ArchitectureOverview() {
  return (
    <section id="architecture" className="scroll-mt-8 space-y-4">
      <div className="border border-neutral-800 rounded-xl p-5 bg-neutral-900/60">
        <h2 className="text-lg font-bold text-white mb-1">Overall Architecture</h2>
        <p className="text-sm text-neutral-400 leading-relaxed whitespace-pre-wrap">{systemArchitecture.overview}</p>
      </div>

      <div className="border border-neutral-800 rounded-xl overflow-hidden">
        <div className="bg-neutral-900/80 px-4 py-2.5 border-b border-neutral-800">
          <span className="text-xs font-semibold uppercase tracking-widest text-neutral-400">System Flow</span>
        </div>
        <pre className="p-5 text-xs text-neutral-300 font-mono leading-relaxed overflow-x-auto bg-neutral-950/50">
          {systemArchitecture.flowDiagram}
        </pre>
      </div>

      <div className="border border-neutral-800 rounded-xl p-5 bg-neutral-950/30 space-y-3">
        <p className="text-xs font-semibold uppercase tracking-widest text-neutral-500">Interface Contracts</p>
        <div className="space-y-2">
          {systemArchitecture.keyContracts.map((contract) => (
            <div key={contract.from} className="border border-neutral-800 rounded-lg p-3 bg-neutral-900/40">
              <p className="text-xs text-neutral-500 mb-1">{contract.from}</p>
              <p className="text-sm text-neutral-300">{contract.data}</p>
            </div>
          ))}
        </div>
        <p className="text-xs text-neutral-400">{systemArchitecture.observabilityNote}</p>
      </div>
    </section>
  );
}
