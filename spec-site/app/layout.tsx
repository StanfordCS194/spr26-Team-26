import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Autonomous ML Training Agent — Technical Spec",
  description: "CS194 Spring 2026 · Team 26 · Stanford University",
  openGraph: {
    title: "Autonomous ML Training Agent — Technical Spec",
    description: "CS194 Spring 2026 · Team 26 · Stanford University",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="scroll-smooth">
      <body suppressHydrationWarning className="antialiased bg-neutral-950 text-neutral-100 min-h-screen">
        {children}
      </body>
    </html>
  );
}
