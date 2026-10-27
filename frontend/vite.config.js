import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// In dev, /api and /ws are proxied to the FastAPI backend, so the browser sees one origin.
const target = process.env.VITE_BACKEND || "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": { target, changeOrigin: true },
      "/ws": { target, ws: true, changeOrigin: true },
    },
  },
  test: { environment: "node" },
});
