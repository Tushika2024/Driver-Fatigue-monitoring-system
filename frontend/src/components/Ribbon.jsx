import { useMemo } from "react";
import { LEVELS, clock } from "../format";

/** Whole-shift timeline: coloured band = alert level, line = fused score, ticks = acknowledgements. */
export default function Ribbon({ history, duration, acks = [], height = 84, nowT }) {
  const W = 1000, H = height;
  const maxT = duration || Math.max(600, history.length ? history[history.length - 1][0] : 0);
  const x = (t) => (t / maxT) * W;
  const { bands, line, dropouts } = useMemo(() => {
    const bands = [], dropouts = [];
    let cur = null;
    history.forEach(([t, , lv, mode], i) => {
      const t1 = i + 1 < history.length ? history[i + 1][0] : t + 1.5;
      if (cur && cur.lv === lv) cur.t1 = t1; else { cur = { t0: t, t1, lv }; bands.push(cur); }
      if (mode === "degraded" || mode === "blind") dropouts.push([t, t1, mode]);
    });
    let d = "", pen = false;
    history.forEach(([t, sc]) => {
      if (sc == null) { pen = false; return; }
      d += `${pen ? "L" : "M"}${x(t).toFixed(1)} ${(H - 18 - sc * (H - 30)).toFixed(1)} `;
      pen = true;
    });
    return { bands, line: d, dropouts };
  }, [history, maxT, H]);
  const marks = [];
  for (let m = 0; m <= maxT; m += maxT > 1200 ? 300 : 120) marks.push(m);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="ribbon" role="img" aria-label="Alert level over the shift">
      <rect x="0" y="0" width={W} height={H - 18} className="rb-bg" />
      {bands.map((b, i) => b.lv > 0 && (
        <rect key={i} x={x(b.t0)} y="0" width={Math.max(x(b.t1) - x(b.t0), 1)} height={H - 18}
              fill={LEVELS[b.lv].color} opacity={0.32 + b.lv * 0.13} />
      ))}
      {dropouts.map(([a, b, m], i) => (
        <rect key={i} x={x(a)} y={H - 24} width={Math.max(x(b) - x(a), 1)} height="6" className={"rb-drop " + m} />
      ))}
      <path d={line} className="rb-line" />
      {acks.map((t, i) => <line key={i} x1={x(t)} x2={x(t)} y1="0" y2={H - 18} className="rb-ack" />)}
      {nowT != null && <line x1={x(nowT)} x2={x(nowT)} y1="0" y2={H - 18} className="rb-now" />}
      {marks.map((m) => (
        <g key={m}><line x1={x(m)} x2={x(m)} y1={H - 18} y2={H - 14} className="rb-tick" />
          <text x={x(m)} y={H - 3} className="rb-lab" textAnchor={m === 0 ? "start" : "middle"}>{clock(m)}</text></g>
      ))}
    </svg>
  );
}
