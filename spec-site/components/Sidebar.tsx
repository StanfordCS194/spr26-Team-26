"use client";

import { useState, useEffect } from "react";
import { sections } from "@/content/spec";

const FEATURE_IDS = new Set([
  "feature-0",
  "feature-1",
  "feature-2",
  "feature-3",
  "feature-4",
  "feature-5",
]);

export default function Sidebar() {
  const [active, setActive] = useState<string>(sections[0].id);
  const [mobileOpen, setMobileOpen] = useState(false);

  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((e) => {
          if (e.isIntersecting) setActive(e.target.id);
        });
      },
      { rootMargin: "-20% 0px -70% 0px" }
    );
    sections.forEach((s) => {
      const el = document.getElementById(s.id);
      if (el) observer.observe(el);
    });
    return () => observer.disconnect();
  }, []);

  const navItems = sections.map((s) => ({
    id: s.id,
    label: FEATURE_IDS.has(s.id)
      ? s.title.replace("Feature ", "F")
      : s.title,
    isFeature: FEATURE_IDS.has(s.id),
    owner: s.owner,
  }));

  const Nav = () => (
    <nav className="space-y-0.5">
      {navItems.map((item) => (
        <a
          key={item.id}
          href={`#${item.id}`}
          onClick={() => setMobileOpen(false)}
          className={`block px-3 py-1.5 rounded-md text-sm transition-colors ${
            active === item.id
              ? "bg-violet-900/60 text-violet-200 font-medium"
              : "text-neutral-400 hover:text-neutral-200 hover:bg-white/5"
          } ${item.isFeature ? "pl-5" : ""}`}
        >
          {item.label}
        </a>
      ))}
    </nav>
  );

  return (
    <>
      {/* Mobile toggle */}
      <button
        className="lg:hidden fixed top-4 left-4 z-50 bg-neutral-800 border border-neutral-700 rounded-lg p-2 text-neutral-300"
        onClick={() => setMobileOpen(!mobileOpen)}
        aria-label="Toggle navigation"
      >
        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          {mobileOpen ? (
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          ) : (
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
          )}
        </svg>
      </button>

      {/* Mobile overlay */}
      {mobileOpen && (
        <div
          className="lg:hidden fixed inset-0 z-40 bg-black/60"
          onClick={() => setMobileOpen(false)}
        />
      )}

      {/* Mobile drawer */}
      <div
        className={`lg:hidden fixed top-0 left-0 z-50 h-full w-64 bg-neutral-900 border-r border-neutral-800 p-4 overflow-y-auto transition-transform ${
          mobileOpen ? "translate-x-0" : "-translate-x-full"
        }`}
      >
        <div className="mb-4 pt-10">
          <p className="text-xs font-semibold uppercase tracking-widest text-neutral-500 mb-3">
            Contents
          </p>
          <Nav />
        </div>
      </div>

      {/* Desktop sidebar */}
      <aside className="hidden lg:block sticky top-0 h-screen w-56 flex-shrink-0 overflow-y-auto border-r border-neutral-800 bg-neutral-950/80 backdrop-blur">
        <div className="p-4 pt-6">
          <p className="text-xs font-semibold uppercase tracking-widest text-neutral-500 mb-3">
            Contents
          </p>
          <Nav />
        </div>
      </aside>
    </>
  );
}
