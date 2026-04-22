import { TEAM } from "@/content/spec";

export default function Header() {
  return (
    <header className="sticky top-0 z-20 border-b border-neutral-800 bg-neutral-950/95 backdrop-blur">
      <div className="px-4 md:px-8 py-4">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <p className="text-xs uppercase tracking-widest text-neutral-500">CS194 Team 26</p>
            <h1 className="text-lg md:text-xl font-bold text-white">ML Agent System Spec</h1>
          </div>
          <div className="flex flex-wrap gap-1.5">
            {TEAM.map((member) => (
              <span
                key={member}
                className="text-xs border border-neutral-700 rounded-full px-2.5 py-1 text-neutral-300 bg-neutral-900/60"
              >
                {member}
              </span>
            ))}
          </div>
        </div>
      </div>
    </header>
  );
}
