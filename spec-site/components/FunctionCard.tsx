import { FunctionSpec } from "@/content/spec";

export default function FunctionCard({ fn }: { fn: FunctionSpec }) {
  return (
    <div className="border border-neutral-800 rounded-lg overflow-hidden">
      {/* Signature bar */}
      <div className="bg-neutral-900 px-4 py-3 border-b border-neutral-800">
        <code className="text-sm text-violet-300 font-mono break-all">{fn.signature}</code>
      </div>

      <div className="p-4 space-y-4 bg-neutral-950/50">
        {/* Description */}
        <p className="text-sm text-neutral-300 leading-relaxed">{fn.description}</p>

        {/* Params table */}
        {fn.params.length > 0 && (
          <div>
            <p className="text-xs font-semibold uppercase tracking-widest text-neutral-500 mb-2">Parameters</p>
            <div className="rounded-md border border-neutral-800 overflow-hidden text-sm">
              <table className="w-full">
                <thead>
                  <tr className="bg-neutral-900">
                    <th className="text-left px-3 py-2 text-xs font-semibold text-neutral-400 w-32">Name</th>
                    <th className="text-left px-3 py-2 text-xs font-semibold text-neutral-400 w-40">Type</th>
                    <th className="text-left px-3 py-2 text-xs font-semibold text-neutral-400">Description</th>
                  </tr>
                </thead>
                <tbody>
                  {fn.params.map((p, i) => (
                    <tr key={p.name} className={i % 2 === 0 ? "bg-neutral-950" : "bg-neutral-900/40"}>
                      <td className="px-3 py-2 font-mono text-xs text-sky-300 align-top">
                        {p.name}
                        {p.optional && <span className="ml-1 text-neutral-600 font-sans">?</span>}
                      </td>
                      <td className="px-3 py-2 font-mono text-xs text-amber-300/80 align-top">{p.type}</td>
                      <td className="px-3 py-2 text-xs text-neutral-400 align-top">{p.description}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {fn.params.length === 0 && (
          <p className="text-xs text-neutral-600 italic">No parameters.</p>
        )}

        {/* Returns */}
        <div>
          <p className="text-xs font-semibold uppercase tracking-widest text-neutral-500 mb-2">Returns</p>
          <div className="flex gap-3 items-start rounded-md border border-neutral-800 bg-neutral-900/40 px-3 py-2">
            <code className="text-xs font-mono text-emerald-400 flex-shrink-0">{fn.returns.type}</code>
            <span className="text-xs text-neutral-400">{fn.returns.description}</span>
          </div>
        </div>

        {fn.notes && (
          <p className="text-xs text-yellow-400/70 bg-yellow-950/20 border border-yellow-900/40 rounded px-3 py-2">{fn.notes}</p>
        )}
      </div>
    </div>
  );
}
