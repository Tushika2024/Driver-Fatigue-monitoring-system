// Thin client for the FastAPI backend. In dev, Vite proxies /api and /ws, so BASE is empty.
const BASE = (import.meta.env.VITE_API_URL || "").replace(/\/$/, "");

export function wsUrl(path) {
  if (BASE) return BASE.replace(/^http/, "ws") + path;
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  return `${proto}//${location.host}${path}`;
}

async function req(path, opts) {
  let res;
  try {
    res = await fetch(BASE + path, opts);
  } catch (e) {
    throw new Error("Can't reach the backend. Is it running on port 8000?");
  }
  if (!res.ok) {
    let msg = res.statusText;
    try { msg = (await res.json()).detail || msg; } catch { /* keep statusText */ }
    throw new Error(typeof msg === "string" ? msg : JSON.stringify(msg));
  }
  return res.json();
}

const json = (body, method = "POST") => ({
  method, headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
});

export const api = {
  health: () => req("/api/health"),
  config: () => req("/api/config"),
  profiles: () => req("/api/profiles"),
  scenarios: () => req("/api/scenarios"),
  route: (body) => req("/api/cabin/route", json(body)),
  sessions: () => req("/api/sessions"),
  report: (id) => req(`/api/sessions/${encodeURIComponent(id)}/report`),
  demoShift: (body) => req("/api/sessions/demo", json(body)),
  depot: (source) => req(`/api/depot?source=${source}`),
  evaluation: () => req("/api/evaluation"),
  pretripHistory: (d) => req(`/api/pretrip/history/${encodeURIComponent(d)}`),
  pretripReset: (d) => req(`/api/pretrip/history/${encodeURIComponent(d)}`, { method: "DELETE" }),
};
