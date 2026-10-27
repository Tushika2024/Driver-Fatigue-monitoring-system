import { useEffect, useState } from "react";
import { api } from "../api";
import { useAsync } from "../hooks/useAsync";
import { LEVELS, MODE_COPY, clock, pct } from "../format";
import { ShiftChart } from "../components/Charts";
import { ErrorBox, Field, LevelChip, Loading, Panel, SyntheticBanner, Tag, Tile } from "../components/Ui";

export default function Report({ id: routeId }) {
  const sessions = useAsync(api.sessions);
  const scen = useAsync(api.scenarios);
  const profiles = useAsync(api.profiles);
  const [id, setId] = useState(routeId || null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState(null);
  const [demo, setDemo] = useState({ scenario: "gradual_fatigue", profile: "ambulance_patient" });

  useEffect(() => { if (routeId) setId(routeId); }, [routeId]);
  useEffect(() => { if (!id && sessions.data?.length) setId(sessions.data[0].id); }, [sessions.data, id]);
  const rep = useAsync(() => (id ? api.report(id) : Promise.resolve(null)), [id]);

  const runDemo = async () => {
    setBusy(true); setErr(null);
    try { const r = await api.demoShift({ ...demo, driver_id: "DEMO01", route_name: "Ring" }); await sessions.reload(); setId(r.id); location.hash = `#/report/${r.id}`; }
    catch (e) { setErr(e.message); } finally { setBusy(false); }
  };

  const m = rep.data?.metrics;
  return (
    <div className="report">
      <Panel title="Shift report" aside={
        <Field label="Session">
          <select value={id || ""} onChange={(e) => { setId(e.target.value); location.hash = `#/report/${e.target.value}`; }}>
            {!sessions.data?.length && <option value="">No sessions yet</option>}
            {(sessions.data || []).map((s) => <option key={s.id} value={s.id}>{s.id} {s.synthetic ? "· synthetic" : ""}</option>)}
          </select>
        </Field>}>
        <div className="demo-bar">
          <span className="muted">No data yet? Run a compressed simulated shift (takes a few seconds):</span>
          <select value={demo.scenario} onChange={(e) => setDemo({ ...demo, scenario: e.target.value })} aria-label="Scenario">
            {(scen.data?.live || []).map((x) => <option key={x.id} value={x.id}>{x.title}</option>)}
          </select>
          <select value={demo.profile} onChange={(e) => setDemo({ ...demo, profile: e.target.value })} aria-label="Vehicle profile">
            {(profiles.data || []).map((p) => <option key={p.name} value={p.name}>{p.name.replace(/_/g, " ")}</option>)}
          </select>
          <button className="btn primary" onClick={runDemo} disabled={busy}>{busy ? "Running…" : "Run demo shift"}</button>
        </div>
        {err && <ErrorBox error={err} />}
        {sessions.error && <ErrorBox error={sessions.error} onRetry={sessions.reload} />}
      </Panel>

      {rep.loading && id && <Loading what="Building report" />}
      {rep.error && <ErrorBox error={rep.error} onRetry={rep.reload} />}
      {rep.data?.synthetic && <SyntheticBanner>This log was recorded from a simulated driver. Every number here is illustrative.</SyntheticBanner>}

      {m && !m.empty && (
        <>
          <div className="tiles">
            <Tile label="Monitored" value={`${m.duration_min.toFixed(0)} min`} sub={rep.data.start_ts?.replace("T", " ")} />
            <Tile label="Eyes + head coverage" value={pct(m.full_mode_frac)} tone={m.low_coverage ? "warn" : "good"}
                  sub={m.low_coverage ? `below ${pct(rep.data.thresholds.min_full_mode_frac)}: treat results with caution` : "full model was running"} />
            <Tile label="Median warning before SEVERE" value={m.median_lead_to_severe_s == null ? "–" : `${Math.round(m.median_lead_to_severe_s)} s`}
                  sub={m.median_lead_to_severe_s == null ? "no SEVERE episode" : "first MILD → first SEVERE"} />
            <Tile label="Driver response" value={m.driver_response_rate == null ? "–" : pct(m.driver_response_rate)} sub="MODERATE+ episodes acknowledged" />
            <Tile label="Worst stretch" value={m.worst_stretch ? `${clock(m.worst_stretch.start_s)}–${clock(m.worst_stretch.end_s)}` : "–"}
                  sub={m.worst_stretch ? `mean score ${m.worst_stretch.mean_score.toFixed(2)}` : "not enough scored data"} />
          </div>

          <Panel title="Fatigue score across the shift">
            <ShiftChart series={rep.data.series} episodes={rep.data.episodes} />
          </Panel>

          <div className="two-col">
            <Panel title="Alert episodes">
              <div className="ep-counts">
                {[1, 2, 3, 4].map((k) => (
                  <div key={k} className="ep-c"><LevelChip lv={k} /><b className="mono">{m.alert_episodes[LEVELS[k].name] ?? 0}</b></div>
                ))}
              </div>
              {rep.data.episodes.length === 0 ? <p className="empty">No alert episodes in this shift.</p> : (
                <table className="tbl">
                  <thead><tr><th>From</th><th>To</th><th>Peak</th><th>Why</th></tr></thead>
                  <tbody>
                    {rep.data.episodes.map((e, i) => (
                      <tr key={i}><td className="mono">{clock(e.start)}</td><td className="mono">{clock(e.end)}</td>
                        <td><LevelChip lv={e.peak} small /></td><td>{e.reason || <span className="muted">–</span>}</td></tr>
                    ))}
                  </tbody>
                </table>
              )}
            </Panel>
            <Panel title="How well could it see?">
              <div className="modebar" role="img" aria-label="Share of the shift in each mode">
                {Object.entries(rep.data.mode_share).map(([k, v]) => <span key={k} className={"mb " + k} style={{ width: `${v * 100}%` }} title={`${k} ${pct(v)}`} />)}
              </div>
              <ul className="plain">
                {Object.entries(rep.data.mode_share).map(([k, v]) => (
                  <li key={k}><Tag tone={k === "full" ? "good" : k === "warmup" ? "neutral" : "warn"}>{(MODE_COPY[k]?.label) || k}</Tag> {pct(v)} <span className="muted">· {MODE_COPY[k]?.note}</span></li>
                ))}
              </ul>
              <p className="muted">This page is for the driver. The depot view only shows aggregates and never ranks individuals.</p>
            </Panel>
          </div>
        </>
      )}
      {m?.empty && <p className="empty">That log is empty.</p>}
    </div>
  );
}
