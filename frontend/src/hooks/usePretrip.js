import { useCallback, useEffect, useReducer, useRef } from "react";
import { wsUrl } from "../api";

const EMPTY = { status: "idle", info: null, elapsed: 0, phase: "settling", trace: [], result: null, error: null };

function reducer(s, a) {
  if (a.type === "reset") return { ...EMPTY, status: "connecting" };
  if (a.type === "error") return { ...s, status: "error", error: a.message };
  if (a.type !== "msg") return s;
  const m = a.msg;
  if (m.type === "started") return { ...s, status: "running", info: m };
  if (m.type === "progress") {
    const add = (m.samples || []).map(([t, ear, , face]) => [t, face ? ear : null]);
    return { ...s, elapsed: m.elapsed, phase: m.phase, trace: s.trace.concat(add).slice(-300) };
  }
  if (m.type === "result") return { ...s, status: "done", result: m.result, info: { ...s.info, history_n: m.history_n } };
  if (m.type === "error") return { ...s, status: "error", error: m.message };
  return s;
}

export function usePretrip() {
  const [state, dispatch] = useReducer(reducer, EMPTY);
  const ws = useRef(null);
  const start = useCallback((cfg) => {
    try { ws.current?.close(); } catch { /* */ }
    dispatch({ type: "reset" });
    const w = new WebSocket(wsUrl("/ws/pretrip"));
    ws.current = w;
    w.onopen = () => w.send(JSON.stringify(cfg));
    w.onmessage = (e) => { try { dispatch({ type: "msg", msg: JSON.parse(e.data) }); } catch { /* */ } };
    w.onerror = () => dispatch({ type: "error", message: "Can't reach the backend. Is it running on port 8000?" });
  }, []);
  useEffect(() => () => { try { ws.current?.close(); } catch { /* */ } }, []);
  return { state, start };
}
