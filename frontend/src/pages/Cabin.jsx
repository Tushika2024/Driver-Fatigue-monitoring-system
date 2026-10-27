import { useEffect, useState } from "react";
import { api } from "../api";
import { useAsync } from "../hooks/useAsync";
import { LEVELS } from "../format";
import CabinMap from "../components/CabinMap";
import { ErrorBox, Field, LevelChip, Loading, Panel, Seg, Tag } from "../components/Ui";

const FLAGS = [
  ["others_present", "Someone else is in the cabin", "patient, passengers, children"],
  ["second_driver", "A second qualified person is there", "co-driver, paramedic"],
  ["overhearing_harmful", "A loud alarm would harm them", "patient in distress, sleeping children"],
  ["has_dispatch", "Dispatch can be notified", "fleet desk"],
  ["private_output", "A truly private speaker exists", "headrest speaker, earpiece"],
];
const ICON = { cabin: "🔊", private: "🎧", crew: "👥", dispatch: "📡" };

function cellText(actions) {
  if (!actions.length) return "—";
  return actions.map((a) => `${ICON[a.channel] || ""} ${a.channel}${a.volume && a.volume !== "n/a" ? ` · ${a.volume}` : ""}`);
}

export default function Cabin() {
  const profiles = useAsync(api.profiles);
  const [mode, setMode] = useState("preset");
  const [name, setName] = useState("ambulance_patient");
  const [flags, setFlags] = useState({ others_present: true, second_driver: false, overhearing_harmful: true, has_dispatch: false, private_output: false });
  const [lv, setLv] = useState(3);
  const [res, setRes] = useState(null);
  const [err, setErr] = useState(null);
  const [replay, setReplay] = useState(0);

  useEffect(() => {
    let dead = false;
    api.route(mode === "preset" ? { level: lv, profile: name } : { level: lv, flags })
      .then((r) => { if (!dead) { setRes(r); setErr(null); setReplay((x) => x + 1); } })
      .catch((e) => !dead && setErr(e.message));
    return () => { dead = true; };
  }, [mode, name, flags, lv]);

  if (profiles.loading) return <Loading what="Loading profiles" />;
  if (profiles.error) return <ErrorBox error={profiles.error} onRetry={profiles.reload} />;
  const announce = res?.actions.find((a) => a.announce);

  return (
    <div className="cabin-page">
      <Panel title="Cabin context explorer">
        <p className="lede">The same fatigue level should not sound the same everywhere. A solo truck driver needs a loud alarm; a driver with a patient in the back needs a quiet voice and a message to the crew. Three questions decide it. <b>A severe alert is never suppressed, only redirected.</b></p>
        <div className="cab-ctl">
          <Seg label="Mode" value={mode} onChange={setMode} options={[{ value: "preset", label: "Vehicle preset" }, { value: "custom", label: "Build your own" }]} />
          {mode === "preset"
            ? <select value={name} onChange={(e) => setName(e.target.value)} aria-label="Vehicle preset">{profiles.data.map((p) => <option key={p.name} value={p.name}>{p.name.replace(/_/g, " ")}</option>)}</select>
            : <div className="flags">{FLAGS.map(([k, t, h]) => (
                <label key={k} className="flag"><input type="checkbox" checked={flags[k]} onChange={(e) => setFlags({ ...flags, [k]: e.target.checked })} />
                  <span>{t}<small>{h}</small></span></label>))}</div>}
          <Field label="Alert level"><Seg label="Alert level" value={lv} onChange={setLv} options={[1, 2, 3, 4].map((i) => ({ value: i, label: LEVELS[i].name }))} /></Field>
        </div>
        {err && <ErrorBox error={err} />}
        {res && (
          <div className="cab-out">
            <CabinMap profile={res.profile} actions={res.actions} pulseKey={replay} />
            <div>
              <div className="cab-head"><LevelChip lv={lv} /> <button className="btn ghost sm" onClick={() => setReplay(replay + 1)}>▶ replay</button></div>
              {res.actions.length === 0 ? <p className="empty">Level {LEVELS[lv].name} adds no extra action here (CRITICAL needs a dispatch desk to escalate to).</p> : (
                <ul className="actions">{res.actions.map((a, i) => (
                  <li key={i}><span className="ic">{ICON[a.channel]}</span>
                    <div><b>{a.channel_label}</b> <Tag>{a.style}</Tag> <Tag>{a.volume === "n/a" ? "silent" : a.volume}</Tag>
                      {a.repeat_until_ack && <Tag tone="warn">repeats until acknowledged</Tag>}
                      <p>“{a.message}”</p>{a.announce && <p className="announce">{a.announce}</p>}</div></li>))}</ul>
              )}
              {announce && <p className="note">Nobody else can be told here, so the system falls back to cabin audio <i>and says so</i>, rather than staying silent.</p>}
            </div>
          </div>
        )}
      </Panel>

      <Panel title="Every preset at every level">
        <div className="matrix-wrap">
          <table className="matrix">
            <thead><tr><th>Vehicle</th>{[1, 2, 3, 4].map((i) => <th key={i}><LevelChip lv={i} small /></th>)}</tr></thead>
            <tbody>{profiles.data.map((p) => (
              <tr key={p.name} className={mode === "preset" && p.name === name ? "cur" : ""}>
                <th scope="row"><button className="linkish" onClick={() => { setMode("preset"); setName(p.name); }}>{p.name.replace(/_/g, " ")}</button></th>
                {Object.values(p.matrix).map((acts, i) => {
                  const t = cellText(acts);
                  return <td key={i}>{Array.isArray(t) ? t.map((x, j) => <div key={j}>{x}</div>) : t}</td>;
                })}
              </tr>))}</tbody>
          </table>
        </div>
      </Panel>
    </div>
  );
}
