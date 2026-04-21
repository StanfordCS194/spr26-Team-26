# Team 26 — Technical Spec Site

Public website for the Autonomous ML Training Agent technical specification.
CS194 · Stanford University · Spring 2026

## Deploy to Vercel (one-time setup, ~2 minutes)

1. Push this repo to GitHub (already done)
2. Go to [vercel.com/new](https://vercel.com/new) and sign in with GitHub
3. Import the **`spr26-Team-26`** repo
4. **Set root directory to `spec-site`** (important!)
5. Click **Deploy** — Vercel auto-detects Next.js

Every push to `main` will auto-redeploy the site.

## Editing the spec

All spec content lives in one file:

```
spec-site/content/spec.ts
```

Each section is a `Section` object with:
- `id` — used for anchor links in the sidebar
- `title` — section heading
- `owner` — team member name (shows a badge on the section card)
- `content` — Markdown string

Edit the markdown content and push to `main`. Vercel redeploys in ~30 seconds.

## Local development

```bash
cd spec-site
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).
