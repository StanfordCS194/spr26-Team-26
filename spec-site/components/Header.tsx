import { TEAM } from "@/content/spec";

export default function Header() {
  return (
    <header className="border-b border-neutral-800 bg-neutral-950/90 backdrop-blur sticky top-0 z-30">
      <div className="px-6 py-3 flex flex-wrap items-center justify-between gap-3">
        <div>
          <div className="flex items-center gap-2 mb-0.5">
            <span className="text-xs font-mono text-violet-400 bg-violet-900/30 border border-violet-800 px-2 py-0.5 rounded">
              CS194 · Team 26 · Spring 2026
            </span>
          </div>
          <h1 className="text-sm font-semibold text-neutral-100">
            Autonomous ML Training Agent
            <span className="text-neutral-500 font-normal ml-2">— Function Reference</span>
          </h1>
        </div>
        <div className="flex items-center gap-3">
          <div className="hidden md:flex flex-wrap gap-1.5">
            {TEAM.map((m) => (
              <span key={m} className="text-xs text-neutral-500 bg-neutral-800/60 px-2 py-0.5 rounded-full">{m}</span>
            ))}
          </div>
          <a
            href="https://github.com/StanfordCS194/spr26-Team-26/blob/main/spec-site/content/spec.ts"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-1.5 text-xs text-neutral-400 hover:text-neutral-200 border border-neutral-700 hover:border-neutral-500 rounded-lg px-3 py-1.5 transition-colors whitespace-nowrap"
          >
            <svg className="w-3.5 h-3.5" fill="currentColor" viewBox="0 0 24 24">
              <path d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z" />
            </svg>
            Edit spec
          </a>
        </div>
      </div>
    </header>
  );
}
