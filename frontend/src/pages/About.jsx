import { api } from "../api";
import { useAsync } from "../hooks/useAsync";
import { Panel, Tag } from "../components/Ui";

export default function About() {
  const h = useAsync(api.health);
  const m = h.data?.model;
  return (
    <div className="about">
      <Panel title="What this is">
        <p className="lede">A fatigue monitor for vehicles that have no built-in driver monitoring: goods trucks, ambulances, buses, cabs. It runs on a laptop CPU with an ordinary camera, keeps no video, and works on a driver's <b>own</b> baseline rather than a population average.</p>
        <div className="steps">
          <div><b>Before</b><span>60-second check against your own last sessions. Advice, never a lock-out.</span></div>
          <div><b>During</b><span>Blink, head and posture signals → graded alerts routed by who is in the cabin.</span></div>
          <div><b>After</b><span>Your shift report. Depot sees aggregates only, small groups hidden.</span></div>
          <div><b>Offline</b><span>Leave-one-person-out evaluation on UTA-RLDD with earliness and fairness checks.</span></div>
        </div>
      </Panel>

      <div className="two-col">
        <Panel title="What has been tested" aside={<Tag tone="good">automated</Tag>}>
          <ul className="plain">
            <li>Geometry, blink timing, adaptive reference, streaming causality, alert persistence and hysteresis.</li>
            <li>Cabin routing for every preset: severe is always redirected, never dropped.</li>
            <li>Pre-trip verdicts, inconclusive handling, report suppression and synthetic stamping.</li>
            <li>This API: REST contracts, the live and pre-trip sockets with simulated drivers, and the camera path with a <i>fake</i> landmark extractor.</li>
          </ul>
        </Panel>
        <Panel title="What has NOT been tested" aside={<Tag tone="warn">be honest in the viva</Tag>}>
          <ul className="plain">
            <li><b>No real UTA-RLDD videos</b> have been run. Folder layout and frame rates are assumptions.</li>
            <li><b>No real webcam</b> and no real MediaPipe output (especially with glasses) in this build.</li>
            <li>Every accuracy figure in this app is from <b>simulated</b> drivers. It shows the code runs, not how good it is.</li>
            <li>RLDD drowsiness is acted, and ~12 people means wide uncertainty.</li>
            <li>Compensatory-behaviour detectors have no ground-truth labels. Crew/dispatch notification is a stub.</li>
          </ul>
        </Panel>
      </div>

      <Panel title="Ethics and scope">
        <ul className="plain">
          <li>No face recognition, no biometric template: identity is the profile the driver picks.</li>
          <li>Scores are a driving-safety aid, not a medical or disciplinary instrument; the depot view cannot rank individuals.</li>
          <li>Only a score log is written. Camera frames are analysed and discarded.</li>
        </ul>
      </Panel>

      <Panel title="System status">
        {h.error ? <p className="empty">Backend unreachable: {h.error}</p> : m ? (
          <dl className="kv">
            <dt>Model</dt><dd>{m.state} {m.synthetic ? <Tag tone="warn">trained on synthetic drivers</Tag> : <Tag tone="good">loaded from file</Tag>}</dd>
            <dt>Source</dt><dd>{m.source || "–"}</dd>
            <dt>Config hash</dt><dd className="mono">{h.data.config_hash}</dd>
            {m.error && (<><dt>Error</dt><dd>{m.error}</dd></>)}
          </dl>
        ) : <p className="muted">Checking…</p>}
      </Panel>
    </div>
  );
}
