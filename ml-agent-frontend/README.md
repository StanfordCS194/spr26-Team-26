# ML Training Agent Frontend

Production-grade dashboard for the CS194 Autonomous ML Training Agent (Team 26, Stanford Spring 2026).

## Setup

```bash
npm install
npm run dev        # http://localhost:5173
npm run build      # production build → dist/
```

By default the UI calls the real Manager API at `/api`; Vite proxies that path
to the local backend during development:

```bash
cd ..
uvicorn src.server.app:app --reload --port 8000

cd ml-agent-frontend
npm run dev
```

To point at a different backend, set `VITE_API_BASE_URL` to the API prefix, for
example `VITE_API_BASE_URL=http://localhost:8000/api npm run dev`.

For static UI demos without a backend, opt into the local simulation:

```bash
VITE_USE_SIMULATION=1 npm run dev
npm run build:simulation
npm run preview:simulation
```

`VITE_*` values are baked into the built bundle. Static previews and Vercel
deployments without a Manager API should use `npm run build:simulation`; otherwise
the app will correctly try to call `/api` and report that the backend is missing.

For a deployed frontend backed by a remote Manager API, build with the API prefix
and allow that frontend origin on the backend:

```bash
VITE_API_BASE_URL=https://manager.example.com/api npm run build
MANAGER_API_CORS_ORIGINS=https://frontend-preview.example.com uvicorn src.server.app:app
```

## Deploy (Vercel)

```bash
vercel --prod
# Build command: npm run build
# Output directory: dist
```

For demo-only Vercel previews that do not have a Manager API attached, use
`npm run build:simulation` as the build command.

Bundle: ~168 KB gzipped.
