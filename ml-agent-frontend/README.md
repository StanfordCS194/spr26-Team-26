# ML Training Agent Frontend

Production-grade dashboard for the CS194 Autonomous ML Training Agent (Team 26, Stanford Spring 2026).

## Setup

```bash
npm install
npm run dev        # http://localhost:5173
npm run build      # production build → dist/
```

By default the UI runs in local simulation mode. To use the real Manager API:

```bash
cd ..
uvicorn src.server.app:app --reload --port 8000

cd ml-agent-frontend
VITE_API_BASE_URL=/api npm run dev
```

## Deploy (Vercel)

```bash
vercel --prod
# Build command: npm run build
# Output directory: dist
```

Bundle: ~168 KB gzipped.
