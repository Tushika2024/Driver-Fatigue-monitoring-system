# Night Watch · frontend

React 18 + Vite. Two runtime dependencies besides React (self-hosted fonts); all charts, the dial, the eye and the cabin diagram are hand-written SVG.

```bash
npm install
npm run dev      # http://localhost:5173, proxies /api and /ws to http://127.0.0.1:8000 (override with VITE_BACKEND)
npm test         # unit tests (vitest)
npm run build    # static files in dist/ (set VITE_API_URL if the API is on another origin)
```

Structure: `src/pages/` (one file per page), `src/components/` (Gauge, Eye, Trace, Ribbon, CabinMap, Charts, Ui),
`src/hooks/` (`useLive`, `usePretrip` own the WebSockets), `src/api.js` (REST client), `src/styles.css` (design tokens at the top).
Design notes: dark "night-drive" cockpit; level is always shown by colour **and** shape/word; `prefers-reduced-motion` is respected;
Space acknowledges an alert; synthetic data is stamped on every page that shows it.
