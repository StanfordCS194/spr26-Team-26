import MarkdownRenderer from "./MarkdownRenderer";
import { Section } from "@/content/spec";

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

export default function SectionCard({ section }: { section: Section }) {
  const ownerClass = section.owner
    ? OWNER_COLORS[section.owner] ?? "bg-violet-900/40 text-violet-300 border-violet-800"
    : null;

  return (
    <section
      id={section.id}
      className="scroll-mt-8 border border-neutral-800 rounded-xl p-6 md:p-8 bg-neutral-900/60"
    >
      <div className="flex flex-wrap items-start justify-between gap-3 mb-5">
        <h2 className="text-xl font-bold text-white">{section.title}</h2>
        {ownerClass && (
          <span
            className={`inline-flex items-center gap-1.5 text-xs font-medium px-2.5 py-1 rounded-full border ${ownerClass}`}
          >
            <svg className="w-3 h-3" fill="currentColor" viewBox="0 0 20 20">
              <path d="M10 9a3 3 0 100-6 3 3 0 000 6zm-7 9a7 7 0 1114 0H3z" />
            </svg>
            {section.owner}
          </span>
        )}
      </div>
      <MarkdownRenderer content={section.content} />
    </section>
  );
}
