import { sharedTypes } from "@/content/spec";

export default function TypesSection() {
  return (
    <section id="types" className="scroll-mt-8 space-y-4">
      <div className="border border-neutral-800 rounded-xl p-5 bg-neutral-900/60">
        <h2 className="text-lg font-bold text-white mb-1">Shared Types</h2>
        <p className="text-sm text-neutral-400">Data structures passed between features. All agents share these schemas.</p>
      </div>

      <div className="grid gap-3 pl-3 border-l-2 border-neutral-800">
        {sharedTypes.map((t) => (
          <div key={t.name} className="border border-neutral-800 rounded-lg overflow-hidden">
            <div className="bg-neutral-900 px-4 py-2.5 border-b border-neutral-800 flex items-center justify-between">
              <code className="text-sm font-mono font-semibold text-emerald-400">{t.name}</code>
              <span className="text-xs text-neutral-500">{t.fields.length} fields</span>
            </div>
            <div className="bg-neutral-950/50">
              <p className="px-4 pt-3 pb-2 text-xs text-neutral-400">{t.description}</p>
              <table className="w-full text-sm">
                <thead>
                  <tr className="bg-neutral-900">
                    <th className="text-left px-4 py-2 text-xs font-semibold text-neutral-500 w-40">Field</th>
                    <th className="text-left px-4 py-2 text-xs font-semibold text-neutral-500 w-40">Type</th>
                    <th className="text-left px-4 py-2 text-xs font-semibold text-neutral-500">Description</th>
                  </tr>
                </thead>
                <tbody>
                  {t.fields.map((f, i) => (
                    <tr key={f.name} className={i % 2 === 0 ? "bg-neutral-950" : "bg-neutral-900/30"}>
                      <td className="px-4 py-2 font-mono text-xs text-sky-300">{f.name}</td>
                      <td className="px-4 py-2 font-mono text-xs text-amber-300/80">{f.type}</td>
                      <td className="px-4 py-2 text-xs text-neutral-400">{f.description}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}
