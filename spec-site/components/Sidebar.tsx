"use client";

import { useState, useEffect } from "react";
import { features } from "@/content/spec";

const ALL_IDS = ["architecture", "types", ...features.map((f) => f.id)];

export default function Sidebar() {
  const [active, setActive] = useState<string>("architecture");
  const [mobileOpen, setMobileOpen] = useState(false);

  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((e) => { if (e.isIntersecting) setActive(e.target.id); });
      },
      { rootMargin: "-15% 0px -75% 0px" }
    );
    ALL_IDS.forEach((id) => {
      const el = document.getElementById(id);
      if (el) observer.observe(el);
    });
    return () => observer.disconnect();
  }, []);

  const linkClass = (id: string) =>
    `block px-3 py-1.5 rounded-md text-sm transition-colors ${
      active === id
        ? "bg-violet-900/60 text-violet-200 font-medium"
        : "text-neutral-400 hover:text-neutral-200 hover:bg-white/5"
    }`;

  const Nav = () => (
    <nav className="space-y-0.5">
      <a href="#architecture" onClick={() => setMobileOpen(false)} className={linkClass("architecture")}>
        System Architecture
      </a>
      <a href="#types" onClick={() => setMobileOpen(false)} className={`${linkClass("types")} mt-1`}>
        Shared Types
      </a>

      <div className="pt-3 mt-1 border-t border-neutral-800">
        <p className="px-3 mb-1 text-xs font-semibold uppercase tracking-wider text-neutral-600">Features</p>
        {features.map((f) => (
          <a
            key={f.id}
            href={`#${f.id}`}
            onClick={() => setMobileOpen(false)}
            className={linkClass(f.id)}
          >
            <span className="block truncate">{f.title}</span>
            <span className="block text-xs text-neutral-600">{f.functions.length} functions</span>
          </a>
        ))}
      </div>
    </nav>
  );

  return (
    <>
      <button
        className="lg:hidden fixed top-4 left-4 z-50 bg-neutral-800 border border-neutral-700 rounded-lg p-2 text-neutral-300"
        onClick={() => setMobileOpen(!mobileOpen)}
        aria-label="Toggle navigation"
      >
        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          {mobileOpen
            ? <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            : <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />}
        </svg>
      </button>

      {mobileOpen && (
        <div className="lg:hidden fixed inset-0 z-40 bg-black/60" onClick={() => setMobileOpen(false)} />
      )}

      <div className={`lg:hidden fixed top-0 left-0 z-50 h-full w-64 bg-neutral-900 border-r border-neutral-800 p-4 pt-14 overflow-y-auto transition-transform ${mobileOpen ? "translate-x-0" : "-translate-x-full"}`}>
        <Nav />
      </div>

      <aside className="hidden lg:block sticky top-0 h-screen w-60 flex-shrink-0 overflow-y-auto border-r border-neutral-800 bg-neutral-950/80 backdrop-blur">
        <div className="p-4 pt-6">
          <Nav />
        </div>
      </aside>
    </>
  );
}
