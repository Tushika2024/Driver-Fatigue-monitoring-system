import { useCallback, useEffect, useReducer, useRef } from "react";
import { wsUrl } from "../api";

const EMPTY = {
  status: "idle",          // idle | connecting | running | ended | error
  info: null, error: null, ended: null,
  t: 0, progress: 0, openLevel: null, eye: null, face: true,
  level: 0, mode: "warmup", score: null, early: null, late: null, reason: "",
  trace: [],               // [t, openness 0..~1.2, face]
  history: [],             // [t, score, level, mode]
  events: [],              // alert actions as they happened
  lastActions: [], lastActionAt: 0, acks: [],
};

function reducer(s, a) {
  switch (a.type) {
    case "reset": return { ...EMPTY, status: "connecting" };
    case "error": return { ...s, status: "error", error: a.message };
    case "closed": return s.status === "running" || s.status === "connecting" ? { ...s, status: "ended" } : s;
    case "msg": {
      const m = a.msg;
      if (m.type === "started") return { ...s, status: "running", info: m };
      if (m.type === "error") return { ...s, status: "error", error: m.message };
      if (m.type === "ended") return { ...s, status: "ended", ended: m };
      if (m.type === "acked") return { ...s, acks: [...s.acks, m.t] };
      if (m.type === "tick") {
        const op = m.open_level;
        let trace = s.trace, eye = s.eye, face = s.face;
        if (m.samples?.length) {
          const add = m.samples.map(([t, ear, , , face]) => [t, face && ear != null && op ? ear / op : null, face]);
          trace = trace.concat(add).slice(-360);       // ~36 s at 10 samples/s
          const vals = add.map((x) => x[1]).filter((v) => v != null);
          face = !!m.samples[m.samples.length - 1][4];
          eye = vals.length ? Math.min(...vals) : null;   // min over the tick so a blink is never averaged away
        }
        return { ...s, eye, face, t: m.t, progress: m.progress ?? s.progress, openLevel: op, trace };
      }
      if (m.type === "update") {
        const history = s.history.concat([[m.t, m.score, m.level, m.mode]]);
        let events = s.events, lastActions = s.lastActions, lastActionAt = s.lastActionAt;
        if (m.actions?.length) {
          events = events.concat(m.actions.map((x) => ({ ...x, t: m.t, reason: m.reason }))).slice(-200);
          lastActions = m.actions; lastActionAt = Date.now();
        }
        return { ...s, level: m.level, mode: m.mode, score: m.score, early: m.early, late: m.late,
                 reason: m.reason, history, events, lastActions, lastActionAt };
      }
      return s;
    }
    default: return s;
  }
}

/** Owns the /ws/live socket. Camera frames are sent as JPEG bytes through sendFrame(). */
export function useLive() {
  const [state, dispatch] = useReducer(reducer, EMPTY);
  const ws = useRef(null);

  const stop = useCallback(() => {
    const w = ws.current;
    if (!w) return;
    try { if (w.readyState === 1) w.send(JSON.stringify({ type: "stop" })); } catch { /* closing */ }
  }, []);

  const start = useCallback((cfg) => {
    if (ws.current) { try { ws.current.close(); } catch { /* */ } }
    dispatch({ type: "reset" });
    const w = new WebSocket(wsUrl("/ws/live"));
    ws.current = w;
    w.onopen = () => w.send(JSON.stringify(cfg));
    w.onmessage = (e) => { try { dispatch({ type: "msg", msg: JSON.parse(e.data) }); } catch { /* ignore */ } };
    w.onerror = () => dispatch({ type: "error", message: "Can't reach the backend. Is it running on port 8000?" });
    w.onclose = () => dispatch({ type: "closed" });
  }, []);

  const ack = useCallback(() => {
    const w = ws.current;
    if (w && w.readyState === 1) w.send(JSON.stringify({ type: "ack" }));
  }, []);
  const setSpeed = useCallback((speed) => {
    const w = ws.current;
    if (w && w.readyState === 1) w.send(JSON.stringify({ type: "speed", speed }));
  }, []);
  const sendFrame = useCallback((blob) => {
    const w = ws.current;
    if (w && w.readyState === 1 && w.bufferedAmount < 400_000) w.send(blob);
  }, []);

  useEffect(() => () => { try { ws.current?.close(); } catch { /* */ } }, []);
  return { state, start, stop, ack, setSpeed, sendFrame };
}
