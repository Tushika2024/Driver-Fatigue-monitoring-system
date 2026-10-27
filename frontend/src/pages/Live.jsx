import { useEffect, useMemo, useRef, useState } from "react";
import { api } from "../api";
import { useAsync } from "../hooks/useAsync";
import { useLive } from "../hooks/useLive";
import { playAction } from "../sound";
import { LEVELS, MODE_COPY, clock, level, pct } from "../format";
import Gauge from "../components/Gauge";
import Eye from "../components/Eye";
import Trace from "../components/Trace";
import Ribbon from "../components/Ribbon";
import CabinMap from "../components/CabinMap";
import { ErrorBox, Field, LevelChip, Panel, Seg, SyntheticBanner, Tag } from "../components/Ui";

const store = {
  get(k, d) { try { const v = localStorage.getItem("nw." + k); return v == null ? d : JSON.parse(v); } catch { return d; } },
  set(k, v) { try { localStorage.setItem("nw." + k, JSON.stringify(v)); } catch { /* private mode */ } },
};

function useCamera(active, onFrame) {
  const [err, setErr] = useState(null);
  const videoRef = useRef(null);
  useEffect(() => {
    if (!active) return undefined;
    let stream, timer, dead = false;
    const canvas = document.createElement("canvas");
    (async () => {
      try {
        stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480, frameRate: 30 }, audio: false });
        if (dead) { stream.getTracks().forEach((t) => t.stop()); return; }
        const v = videoRef.current;
        v.srcObject = stream; await v.play();
        canvas.width = 480; canvas.height = 360;
        const g = canvas.getContext("2d");
        timer = setInterval(() => {
          if (v.readyState < 2) return;
          g.drawImage(v, 0, 0, canvas.width, canvas.height);
          canvas.toBlob((b) => b && onFrame(b), "image/jpeg", 0.7);
        }, 34);
      } catch (e) {
        setErr(e.name === "NotAllowedError" ? "Camera permission was denied." : `Camera unavailable: ${e.message}`);
      }
    })();
    return () => { dead = true; clearInterval(timer); stream?.getTracks().forEach((t) => t.stop()); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [active]);
  return { videoRef, err };
}

export default function Live() {
  const { state: s, start, stop, ack, setSpeed, sendFrame } = useLive();
  const profiles = useAsync(api.profiles);
  const scen = useAsync(api.scenarios);
  const cfg = useAsync(api.config);

  const [source, setSource] = useState(store.get("source", "simulated"));
  const [scenario, setScenario] = useState(store.get("scenario", "gradual_fatigue"));
  const [profile, setProfile] = useState(store.get("profile", "ambulance_patient"));
  const [speed, setSpd] = useState(store.get("speed", 15));
  const [driver, setDriver] = useState(store.get("driver", "D01"));
  const [sound, setSound] = useState(false);
  useEffect(() => { store.set("source", source); store.set("scenario", scenario); store.set("profile", profile); store.set("speed", speed); store.set("driver", driver); },
    [source, scenario, profile, speed, driver]);

  const running = s.status === "running" || s.status === "connecting";
  const cam = useCamera(running && s.status === "running" && source === "camera", sendFrame);

  // sound follows the server's routing decision
  useEffect(() => { if (sound && s.lastActionAt) s.lastActions.forEach(playAction); }, [s.lastActionAt]); // eslint-disable-line
  // Space acknowledges
  useEffect(() => {
    const h = (e) => {
      if (e.code === "Space" && running && !/INPUT|SELECT|TEXTAREA|BUTTON/.test(document.activeElement?.tagName || "")) { e.preventDefault(); ack(); }
    };
    window.addEventListener("keydown", h);
    return () => window.removeEventListener("keydown", h);
  }, [running, ack]);

  const selProfile = profiles.data?.find((p) => p.name === profile);
  const cabinProfile = s.info?.profile || selProfile;
  const L = level(s.level);
  const mode = MODE_COPY[s.mode] || MODE_COPY.warmup;
  const lost = s.mode === "blind";
  const scnInfo = scen.data?.live.find((x) => x.id === scenario);
  const needsAck = running && s.level >= 3;
  const warm = cfg.data?.reference_min_s ?? 20;

  const go = () => start({
    source, scenario, profile, speed, driver_id: driver || "D01", vehicle_id: "VEH-1", route: "Live", 
  });
  const changeSpeed = (v) => { setSpd(v); setSpeed(v); };

  const sinceAction = useMemo(() => s.lastActionAt, [s.lastActionAt]);
  const lastEvents = [...s.events].reverse().slice(0, 40);

  return (
    <div className="live" data-lv={running ? s.level : 0}>
      <div className="vignette" aria-hidden />
      {source === "simulated" && <SyntheticBanner>The driver below is simulated, but the pipeline that reads it (blink timing, adaptive reference, fusion model, alert policy, cabin routing) is the real one.</SyntheticBanner>}

      <Panel className="controls">
        <div className="ctl-row">
          <Field label="Source">
            <Seg label="Source" value={source} onChange={setSource} disabled={running}
                 options={[{ value: "simulated", label: "Simulated driver" }, { value: "camera", label: "My camera" }]} />
          </Field>
          {source === "simulated" && (
            <Field label="Scenario" hint={scnInfo?.blurb}>
              <select value={scenario} onChange={(e) => setScenario(e.target.value)} disabled={running}>
                {(scen.data?.live || []).map((x) => <option key={x.id} value={x.id}>{x.title} · {Math.round(x.duration_s / 60)} min</option>)}
              </select>
            </Field>
          )}
          <Field label="Vehicle profile" hint="Decides who hears what">
            <select value={profile} onChange={(e) => setProfile(e.target.value)} disabled={running}>
              {(profiles.data || []).map((p) => <option key={p.name} value={p.name}>{p.name.replace(/_/g, " ")}</option>)}
            </select>
          </Field>
          <Field label="Driver ID"><input value={driver} maxLength={32} onChange={(e) => setDriver(e.target.value.replace(/[^A-Za-z0-9_.-]/g, ""))} disabled={running} /></Field>
          {source === "simulated" && (
            <Field label="Playback speed">
              <Seg label="Speed" value={speed} onChange={changeSpeed}
                   options={[1, 5, 15, 30, 60].map((v) => ({ value: v, label: `${v}×` }))} />
            </Field>
          )}
          <div className="ctl-actions">
            <label className="switch"><input type="checkbox" checked={sound} onChange={(e) => setSound(e.target.checked)} /><span>Sound</span></label>
            {!running
              ? <button className="btn primary" onClick={go} disabled={profiles.loading}>Start monitoring</button>
              : <button className="btn danger" onClick={stop}>Stop</button>}
          </div>
        </div>
        {(profiles.error || scen.error) && <ErrorBox error={profiles.error || scen.error} onRetry={() => { profiles.reload(); scen.reload(); }} />}
        {s.status === "error" && <ErrorBox error={s.error} />}
        {cam.err && <ErrorBox error={cam.err} />}
        {source === "camera" && !running && (
          <p className="note">Camera mode streams JPEG frames to your own backend for analysis; nothing is stored except the score log. It needs a face landmark model on the server and ~25 fps. <b>This path has not been tested with a real webcam.</b></p>
        )}
      </Panel>

      <div className="cockpit">
        <Panel className="dial-panel" title="Fatigue" aside={<Tag tone={s.mode === "full" ? "good" : s.mode === "warmup" ? "neutral" : "warn"}>{mode.label}</Tag>}>
          <Gauge score={s.mode === "warmup" ? null : s.score} lv={s.level} mode={s.mode}
                 label={s.mode === "warmup" ? `calibrating… ${Math.min(99, Math.round((s.t / warm) * 100))}%` : undefined} />
          <div className="two-meters">
            <Meter label="Early signs" v={s.early} hint="blink slowdown, rubbing, head shake" />
            <Meter label="Late signs" v={s.late} hint="eye closure, long closures, head drop" />
          </div>
          <div className="reason" aria-live="polite">
            {s.level > 0 ? <><b style={{ color: L.color }}>{L.hint}:</b> {s.reason}</> : running ? (s.mode === "degraded" || s.mode === "blind" ? null : <span className="muted">{mode.note}</span>) : <span className="muted">Start a session to begin.</span>}
          </div>
          {(s.mode === "degraded" || s.mode === "blind") && <div className={"modebox " + s.mode}><b>{mode.label}.</b> {mode.note}</div>}
        </Panel>

        <Panel className="eye-panel" title="What the system sees"
               aside={running && <span className="clock mono">{clock(s.t)}{scnInfo && source === "simulated" ? ` / ${clock(scnInfo.duration_s)}` : ""}</span>}>
          <div className="eye-wrap">
            <Eye openness={s.eye} color={L.color} lost={lost || (running && s.t > 1 && (s.eye == null || !s.face))}
                 lostText={!s.face || lost ? "no face in view" : "eyes unreadable"} />
            {source === "camera" && <video ref={cam.videoRef} className="selfview" muted playsInline aria-label="Camera preview" />}
          </div>
          <div className="eye-cap">
            <span>eye openness · <i className="mono">{s.eye == null ? "–" : pct(s.eye)}</i> of this driver's own open level</span>
          </div>
          <Trace data={s.trace} closedAt={cfg.data?.closed_frac ?? 0.5} height={130} />
          <div className="trace-legend"><span><i className="sw open" />driver's open level</span><span><i className="sw thr" />counted as closed</span><span><i className="sw gap" />eyes unreadable</span></div>
        </Panel>

        <Panel className="cabin-panel" title="What the cabin hears" aside={cabinProfile && <Tag>{cabinProfile.name.replace(/_/g, " ")}</Tag>}>
          <CabinMap profile={cabinProfile} actions={s.lastActions} pulseKey={sinceAction} />
          <div className="cabin-now" aria-live="polite">
            {s.lastActions.length === 0 && <span className="muted">Quiet. Alerts are routed by who is in the vehicle.</span>}
            {s.lastActions.map((a, i) => (
              <div key={i} className="act"><LevelChip lv={a.level} small /> <b>{a.channel_label}</b> · {a.style}, {a.volume === "n/a" ? "silent" : a.volume}
                {a.repeat_until_ack && <Tag tone="warn">repeats until acknowledged</Tag>}
                {a.announce && <div className="announce">{a.announce}</div>}</div>
            ))}
          </div>
          <button className={"btn ack" + (needsAck ? " urgent" : "")} onClick={ack} disabled={!running}>
            I'm alert · acknowledge <kbd>Space</kbd>
          </button>
        </Panel>
      </div>

      <Panel title="Shift timeline" aside={<span className="legend">{LEVELS.slice(1).map((l) => <span key={l.id} className="lg"><i style={{ background: l.color }} />{l.word}</span>)}</span>}>
        <Ribbon history={s.history} duration={s.info?.duration_s} acks={s.acks} nowT={running ? s.t : null} />
        {s.status === "ended" && (
          <div className="ended">
            Session {s.ended ? (s.ended.stopped_by_user ? "stopped" : "complete") : "closed"} · {s.history.length} windows scored
            {s.ended?.perf?.fps ? ` · ${Math.round(s.ended.perf.fps)} frames/s, p95 ${s.ended.perf.latency_ms_p95?.toFixed(1)} ms` : ""}.{" "}
            {(s.ended?.session_id || s.info?.session_id) && <a href={`#/report/${s.ended?.session_id || s.info.session_id}`}>Open shift report →</a>}
          </div>
        )}
      </Panel>

      <Panel title="Alert log" aside={<span className="muted">{s.events.length} actions</span>}>
        {lastEvents.length === 0 ? <p className="muted">Nothing yet. The log lists every routed action: who was alerted, how, and why.</p> : (
          <ol className="log">
            {lastEvents.map((e, i) => (
              <li key={i}><span className="mono t">{clock(e.t)}</span><LevelChip lv={e.level} small />
                <span><b>{e.channel_label}</b> · {e.style} · {e.volume === "n/a" ? "silent" : e.volume}</span>
                <span className="msg">{e.message}</span></li>
            ))}
          </ol>
        )}
      </Panel>
    </div>
  );
}

function Meter({ label, v, hint }) {
  return (
    <div className="meter" title={hint}>
      <div className="meter-l"><span>{label}</span><span className="mono">{v == null ? "–" : v.toFixed(2)}</span></div>
      <div className="meter-t"><div className="meter-f" style={{ width: `${(v ?? 0) * 100}%` }} /></div>
    </div>
  );
}
