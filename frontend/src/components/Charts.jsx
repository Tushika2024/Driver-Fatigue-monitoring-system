import { Fragment, useState } from "react";
import { LEVELS, clock, num, pct } from "../format";

/** Horizontal bars with optional ± whisker. rows: [{label, value, spread?, note?, tone?}] */
export function BarList({ rows, max = 1, fmt = (v) => pct(v), highlight }) {
  return (
    <div className="barlist">
      {rows.map((r, i) => (
        <div key={i} className={"bl-row" + (highlight === i ? " hi" : "") + (r.tone ? " " + r.tone : "")}>
          <div className="bl-lab" title={r.label}>{r.label}</div>
          <div className="bl-track" role="img" aria-label={`${r.label}: ${fmt(r.value)}`}>
            <div className="bl-bar" style={{ width: `${Math.max(0, Math.min(1, r.value / max)) * 100}%` }} />
            {r.spread != null && isFinite(r.spread) && (
              <div className="bl-whisk" style={{
                left: `${Math.max(0, (r.value - r.spread) / max) * 100}%`,
                width: `${Math.min(1, (r.value + r.spread) / max) * 100 - Math.max(0, (r.value - r.spread) / max) * 100}%` }} />
            )}
          </div>
          <div className="bl-val">{fmt(r.value)}{r.spread != null && isFinite(r.spread) && <small> ±{fmt(r.spread)}</small>}</div>
        </div>
      ))}
    </div>
  );
}

const STOPS = [[22, 34, 58], [38, 96, 168], [255, 209, 102], [255, 93, 108]];
const lum = ([r, g, b]) => 0.2126 * r + 0.7152 * g + 0.0722 * b;
function rgbAt(t) {
  const x = Math.max(0, Math.min(1, t)) * (STOPS.length - 1), i = Math.min(STOPS.length - 2, Math.floor(x)), f = x - i;
  return STOPS[i].map((c, k) => Math.round(c + (STOPS[i + 1][k] - c) * f));
}
const ramp = (t) => { const c = rgbAt(t); return `rgb(${c[0]} ${c[1]} ${c[2]})`; };
const ink = (t) => (lum(rgbAt(t)) > 150 ? "#07101f" : "#fff");

/** Slot x route heatmap. cells: [{slot_label, route, pct_time_alert, suppressed, drivers, hours, events_per_100h}] */
export function Heatmap({ slots, routes, cells, k }) {
  const [sel, setSel] = useState(null);
  const by = Object.fromEntries(cells.map((c) => [c.slot_label + "|" + c.route, c]));
  const vals = cells.filter((c) => !c.suppressed && c.pct_time_alert != null).map((c) => c.pct_time_alert);
  const mx = Math.max(10, ...vals);
  const shade = (v) => ramp(v / mx);
  const cur = sel && by[sel];
  return (
    <div className="heat">
      <div className="heat-grid" style={{ gridTemplateColumns: `90px repeat(${routes.length}, minmax(80px, 1fr))` }}>
        <div />
        {routes.map((r) => <div key={r} className="heat-col">{r}</div>)}
        {slots.map((s) => (
          <Fragment key={s}>
            <div className="heat-row">{s}</div>
            {routes.map((r) => {
              const c = by[s + "|" + r];
              const id = s + "|" + r;
              if (!c) return <div key={id} className="heat-cell empty">·</div>;
              return (
                <button key={id} className={"heat-cell" + (c.suppressed ? " hidden" : "") + (sel === id ? " sel" : "")}
                        style={c.suppressed ? undefined : { background: shade(c.pct_time_alert), color: ink(c.pct_time_alert / mx) }}
                        onClick={() => setSel(sel === id ? null : id)}
                        aria-label={c.suppressed ? `${s}, ${r}: hidden, fewer than ${k} drivers` : `${s}, ${r}: ${num(c.pct_time_alert, 1)} percent of time at moderate or higher`}>
                  {c.suppressed ? <span>hidden<small>&lt; {k} drivers</small></span> : <span>{num(c.pct_time_alert, 0)}%</span>}
                </button>
              );
            })}
          </Fragment>
        ))}
      </div>
      <div className="heat-detail" aria-live="polite">
        {cur ? (
          cur.suppressed
            ? <>Only <b>{cur.drivers}</b> driver{cur.drivers === 1 ? "" : "s"} contributed to <b>{cur.slot_label} · {cur.route}</b>. Below the privacy floor (k = {k}), so the numbers are withheld.</>
            : <><b>{cur.slot_label} · {cur.route}</b>: {num(cur.pct_time_alert, 1)}% of monitored time at MODERATE or higher · {num(cur.events_per_100h, 1)} episodes per 100 h · {num(cur.hours, 1)} h from {cur.drivers} drivers.</>
        ) : "Select a cell. No individual driver is ever shown."}
      </div>
    </div>
  );
}

/** Score-over-time chart for a finished shift with level bands and episode markers. */
export function ShiftChart({ series, episodes = [], height = 220, onHover }) {
  const W = 900, H = height, padL = 34, padB = 22, padT = 8;
  const [hv, setHv] = useState(null);
  if (!series?.length) return null;
  const tMax = series[series.length - 1].t || 1;
  const x = (t) => padL + (t / tMax) * (W - padL - 8);
  const y = (v) => padT + (1 - v) * (H - padT - padB);
  let d = "", pen = false;
  series.forEach((p) => {
    if (p.score == null) { pen = false; return; }
    d += `${pen ? "L" : "M"}${x(p.t).toFixed(1)} ${y(p.score).toFixed(1)} `; pen = true;
  });
  const bands = [];
  series.forEach((p, i) => {
    const t1 = series[i + 1]?.t ?? p.t;
    const last = bands[bands.length - 1];
    if (last && last.lv === p.level) last.t1 = t1; else bands.push({ t0: p.t, t1, lv: p.level });
  });
  const move = (e) => {
    const r = e.currentTarget.getBoundingClientRect();
    const t = ((e.clientX - r.left) / r.width * W - padL) / (W - padL - 8) * tMax;
    let best = null, bd = 1e9;
    series.forEach((p) => { const dd = Math.abs(p.t - t); if (dd < bd) { bd = dd; best = p; } });
    setHv(best); onHover?.(best);
  };
  const ticks = [];
  const step = tMax > 3600 ? 900 : tMax > 1200 ? 300 : 120;
  for (let m = 0; m <= tMax; m += step) ticks.push(m);
  return (
    <div className="shift-chart">
      <svg viewBox={`0 0 ${W} ${H}`} onMouseMove={move} onMouseLeave={() => setHv(null)} role="img"
           aria-label="Fatigue score across the shift">
        {[0, 0.25, 0.5, 0.75, 1].map((g) => (
          <g key={g}><line x1={padL} x2={W - 8} y1={y(g)} y2={y(g)} className="sc-grid" />
            <text x={padL - 6} y={y(g) + 4} textAnchor="end" className="sc-lab">{g}</text></g>
        ))}
        {bands.map((b, i) => b.lv > 0 && (
          <rect key={i} x={x(b.t0)} y={padT} width={Math.max(x(b.t1) - x(b.t0), 1)} height={H - padT - padB}
                fill={LEVELS[b.lv].color} opacity={0.12 + b.lv * 0.07} />
        ))}
        <path d={d} className="sc-line" />
        {episodes.filter((e) => e.peak >= 2).map((e, i) => (
          <g key={i}><line x1={x(e.start)} x2={x(e.start)} y1={padT} y2={H - padB} stroke={LEVELS[e.peak].color} className="sc-ep" />
            <text x={x(e.start) + 4} y={padT + 12} className="sc-eplab" fill={LEVELS[e.peak].color}>{LEVELS[e.peak].glyph}</text></g>
        ))}
        {ticks.map((m) => <text key={m} x={x(m)} y={H - 6} textAnchor="middle" className="sc-lab">{clock(m)}</text>)}
        {hv && <line x1={x(hv.t)} x2={x(hv.t)} y1={padT} y2={H - padB} className="sc-cursor" />}
      </svg>
      <div className="sc-read" aria-live="polite">
        {hv ? <>at <b>{clock(hv.t)}</b> · score <b>{hv.score == null ? "–" : hv.score.toFixed(2)}</b> · {LEVELS[hv.level].name} · {hv.mode}</> : "Hover the chart to read values."}
      </div>
    </div>
  );
}

/** Signed z bars for the pre-trip measures (0 is "your usual"). */
export function ZBars({ z, labels, worse = {} }) {
  const keys = Object.keys(z || {});
  if (!keys.length) return null;
  const R = 3;
  const isWorse = (v, dir) => (dir === 0 ? Math.abs(v) > 1 : dir < 0 ? v < 0 : v > 0);
  return (
    <div className="zbars">
      {keys.map((k) => {
        const v = Math.max(-R, Math.min(R, z[k]));
        const w = (Math.abs(v) / R) * 50;
        return (
          <div className="zb-row" key={k}>
            <div className="zb-lab">{labels?.[k] || k}</div>
            <div className="zb-track" role="img" aria-label={`${labels?.[k] || k}: ${z[k].toFixed(1)} robust standard deviations from your usual`}>
              <span className="zb-mid" />
              <span className={"zb-bar " + (isWorse(z[k], worse[k]) ? "pos" : "neg")} style={{ left: v >= 0 ? "50%" : `${50 - w}%`, width: `${w}%` }} />
            </div>
            <div className="zb-val">{z[k] > 0 ? "+" : ""}{z[k].toFixed(1)}</div>
          </div>
        );
      })}
      <div className="zb-axis"><span>← lower than your usual</span><span>usual</span><span>higher →</span></div>
      <div className="zb-key"><i className="k-warn" /> toward tired <i className="k-ok" /> toward rested / neutral</div>
    </div>
  );
}
