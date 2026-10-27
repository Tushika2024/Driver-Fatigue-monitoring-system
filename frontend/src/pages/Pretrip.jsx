import { useEffect, useState } from "react";
import { api } from "../api";
import { useAsync } from "../hooks/useAsync";
import { usePretrip } from "../hooks/usePretrip";
import { clamp, num } from "../format";
import Eye from "../components/Eye";
import Trace from "../components/Trace";
import { ZBars } from "../components/Charts";
import { ErrorBox, Field, Panel, Seg, SyntheticBanner, Tag } from "../components/Ui";

const VERDICT = {
  FIT:          { color: "var(--lv0)", icon: "✓", title: "Looks like your usual self", sub: "Nothing stands out against your own recent history." },
  CAUTION:      { color: "var(--lv2)", icon: "!", title: "A little off your usual", sub: "Take extra care, plan breaks early, and consider a short rest first." },
  NOT_FIT:      { color: "var(--lv3)", icon: "■", title: "Well off your usual", sub: "Strongly consider resting before this trip. This is advice, not a lock-out." },
  ENROLLING:    { color: "var(--accent2)", icon: "…", title: "Building your baseline", sub: "A few more checks are needed before a verdict can be given. Nothing is wrong." },
  INCONCLUSIVE: { color: "var(--muted)", icon: "?", title: "Couldn't measure this time", sub: "Not stored, and not counted against you. Try again with better light or a clearer view." },
};
const LABELS = {
  reopen_s: "Eyes re-opening speed", openness: "How open your eyes are", rate_pm: "Blink rate", head_micro: "Head movement",
  reopen: "Eyes re-opening speed", rate: "Blink rate", head: "Head movement",
};

export default function Pretrip() {
  const { state: s, start } = usePretrip();
  const [driver, setDriver] = useState("D07");
  const [day, setDay] = useState("normal");
  const [speed, setSpeed] = useState(6);
  const scen = useAsync(api.scenarios);
  const cfg = useAsync(api.config);
  const hist = useAsync(() => api.pretripHistory(driver), [driver, s.status === "done"]);
  const running = s.status === "connecting" || s.status === "running";
  const cap = s.info?.capture_s ?? 60, disc = s.info?.discard_s ?? 10;
  const prog = clamp(s.elapsed / cap, 0, 1);
  const R = 70, C = 2 * Math.PI * R;
  const last = s.trace.length ? s.trace[s.trace.length - 1][1] : null;
  const seen = s.trace.map((p) => p[1]).filter((v) => v != null).sort((a, b) => a - b);
  const open90 = seen.length ? seen[Math.floor(seen.length * 0.9)] : null;     // preview only: scale to the driver's own open level
  const rel = (v) => (v == null || !open90 ? null : v / open90);
  const norm = rel(last);
  const r = s.result, V = r && (VERDICT[r.verdict] || VERDICT.INCONCLUSIVE);

  const [resetErr, setResetErr] = useState(null);
  const reset = async () => { try { setResetErr(null); await api.pretripReset(driver); hist.reload(); } catch (e) { setResetErr(e.message); } };
  useEffect(() => { setResetErr(null); }, [driver]);

  return (
    <div className="pretrip">
      <SyntheticBanner>A simulated driver stands in for the camera so you can see the whole check. The verdict logic is the real one.</SyntheticBanner>
      <div className="two-col">
        <Panel title="Before you drive · 60-second check" className="pt-main">
          <div className="pt-controls">
            <Field label="Driver ID"><input value={driver} maxLength={32} disabled={running}
                                          onChange={(e) => setDriver(e.target.value.replace(/[^A-Za-z0-9_.-]/g, ""))} /></Field>
            <Field label="Simulate…">
              <select value={day} onChange={(e) => setDay(e.target.value)} disabled={running}>
                {(scen.data?.pretrip || []).map((x) => <option key={x.id} value={x.id}>{x.title}</option>)}
              </select>
            </Field>
            <Field label="Speed"><Seg label="Speed" value={speed} onChange={setSpeed} disabled={running}
                                      options={[{ value: 1.5, label: "Real time" }, { value: 6, label: "6×" }, { value: 20, label: "20×" }]} /></Field>
            <button className="btn primary" disabled={running || !driver} onClick={() => start({ driver_id: driver, state: day, speed })}>
              {running ? "Measuring…" : "Begin check"}
            </button>
          </div>

          <div className="pt-stage">
            <div className="ring-wrap">
              <svg viewBox="0 0 180 180" className="pt-ring" role="img" aria-label={`Check ${Math.round(prog * 100)} percent complete`}>
                <circle cx="90" cy="90" r={R} className="pr-track" />
                <circle cx="90" cy="90" r={R} className="pr-fill" strokeDasharray={C} strokeDashoffset={C * (1 - prog)}
                        stroke={r ? V.color : "var(--lv0)"} />
                <circle cx="90" cy="90" r={R} className="pr-disc" strokeDasharray={`${(disc / cap) * C} ${C}`} />
              </svg>
              <div className="ring-in">
                {r ? <span className="big" style={{ color: V.color }}>{V.icon}</span> : <Eye openness={norm} size={120} />}
                <div className="mono secs">{r ? r.verdict.replace("_", " ") : `${Math.round(s.elapsed)}s / ${cap}s`}</div>
              </div>
            </div>
            <div className="pt-side">
              <div className={"phase" + (s.status === "running" && s.phase === "settling" ? " on" : "")}><b>1</b> Settle in<small>first {disc} s are ignored</small></div>
              <div className={"phase" + (s.status === "running" && s.phase === "measuring" ? " on" : "")}><b>2</b> Look ahead, blink naturally<small>blinks and head movement are measured</small></div>
              <div className={"phase" + (r ? " on" : "")}><b>3</b> Compare with your own history<small>never with other drivers</small></div>
              <Trace data={s.trace.map((p) => [p[0], rel(p[1])])} height={70} closedAt={0.5} />
            </div>
          </div>
          {s.status === "error" && <ErrorBox error={s.error} />}

          {r && (
            <div className="verdict" style={{ "--c": V.color }}>
              <div className="v-head"><span className="v-ic">{V.icon}</span>
                <div><h2>{V.title}</h2><p>{V.sub}</p></div>
                <Tag tone="neutral" title="The check never blocks a shift">advisory only</Tag></div>
              {r.verdict !== "INCONCLUSIVE" && r.verdict !== "ENROLLING" ? (
                <>
                  <div className="v-idx"><span>Fatigue index vs your usual</span><b className="mono">{num(r.index, 2)}</b>
                    <small>CAUTION from {num(cfg.data?.pretrip?.caution_at, 1)} · NOT FIT from {num(cfg.data?.pretrip?.not_fit_at, 1)} (configurable)</small></div>
                  <p className="v-reason">{r.reason}</p>
                  <ZBars z={r.z} labels={LABELS} worse={{ reopen: 1, openness: -1, rate: 0, head: 1 }} />
                  <div className="v-meta">
                    <Tag>{r.n_blinks} blinks</Tag><Tag>{Math.round(r.duration_s)} s</Tag>
                    <Tag tone={r.history_used === "none" ? "warn" : "good"}>baseline: {r.history_used.replace(/_/g, " ")}</Tag>
                    <Tag>{r.stored ? "saved to your history" : "not saved"}</Tag>
                  </div>
                  {r.warnings?.map((w, i) => <p key={i} className="warn-line">⚠ {w}</p>)}
                </>
              ) : <p className="v-reason">{r.reason}</p>}
            </div>
          )}
        </Panel>

        <Panel title={`History for ${driver || "…"}`} aside={<button className="btn ghost sm" onClick={reset} disabled={running}>Reset</button>}>
          <p className="muted">Each completed check is compared with this driver's own last sessions (same time of day when there are enough). Inconclusive checks are never stored. There is no face recognition: the driver picks their own ID.</p>
          {resetErr && <ErrorBox error={resetErr} />}
          {hist.error ? <ErrorBox error={hist.error} onRetry={hist.reload} /> : (
            <HistoryList sessions={hist.data?.sessions || []} />
          )}
          <h4>How the verdict is built</h4>
          <ul className="plain">
            <li>Four measures: eyes re-opening speed, openness, blink rate, small head movements.</li>
            <li>Each is a robust distance (median / IQR) from this driver's own history.</li>
            <li>They are combined with fixed weights into one index; thresholds are in the config.</li>
            <li>No history yet? The result says so and uses cautious defaults.</li>
          </ul>
        </Panel>
      </div>
    </div>
  );
}

function HistoryList({ sessions }) {
  if (!sessions.length) return <p className="empty">No saved checks yet. Run one, or run it a few times to build a baseline.</p>;
  const mx = Math.max(...sessions.map((x) => x.openness || 0), 0.001);
  return (
    <ol className="hist">
      {sessions.slice(-8).reverse().map((x, i) => (
        <li key={i}><span className="mono">{new Date(x.ts).toLocaleDateString(undefined, { month: "short", day: "numeric" })} {String(new Date(x.ts).getHours()).padStart(2, "0")}h</span>
          <span className="hbar"><i style={{ width: `${((x.openness || 0) / mx) * 100}%` }} /></span>
          <span className="mono">{num(x.rate_pm, 0)}/min</span></li>
      ))}
    </ol>
  );
}
