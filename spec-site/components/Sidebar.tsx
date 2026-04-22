import { features } from "@/content/spec";

export default function Sidebar() {
  return (
    <aside className="hidden lg:block w-72 shrink-0 border-r border-neutral-800 bg-neutral-950/40">
      <div className="sticky top-[73px] h-[calc(100vh-73px)] overflow-y-auto px-4 py-6">
        <p className="text-xs uppercase tracking-widest text-neutral-500 mb-3">Navigation</p>
        <nav className="space-y-1.5">
          <a
            href="#architecture"
            className="block rounded-md px-3 py-2 text-sm text-neutral-300 hover:bg-neutral-900 hover:text-white transition-colors"
          >
            Overall Architecture
          </a>
          <a
            href="#types"
            className="block rounded-md px-3 py-2 text-sm text-neutral-300 hover:bg-neutral-900 hover:text-white transition-colors"
          >
            Shared Types
          </a>
          {features.map((feature) => (
            <a
              key={feature.id}
              href={`#${feature.id}`}
              className="block rounded-md px-3 py-2 text-sm text-neutral-300 hover:bg-neutral-900 hover:text-white transition-colors"
            >
              {feature.title}
            </a>
          ))}
        </nav>
      </div>
    </aside>
  );
}
