import { clamp, level } from "../format";

// 240° instrument-cluster dial. The needle is the fused fatigue probability (0..1).
const R = 90, CX = 120, CY = 116, A0 = 240, SWEEP = 240;
const pol = (a, r = R) => {
  const t = ((a - 90) * Math.PI) / 180;
  return [CX + r * Math.cos(t), CY + r * Math.sin(t)];
};
const arc = (a0, a1, r = R) => {
  const [x0, y0] = pol(a0, r), [x1, y1] = pol(a1, r);
  return `M${x0.toFixed(2)} ${y0.toFixed(2)} A${r} ${r} 0 ${a1 - a0 > 180 ? 1 : 0} 1 ${x1.toFixed(2)} ${y1.toFixed(2)}`;
};

export default function Gauge({ score, lv = 0, mode = "warmup", label }) {
  const L = level(lv);
  const v = score == null ? 0 : clamp(score, 0, 1);
  const ang = A0 + SWEEP * v;
  const ticks = Array.from({ length: 41 }, (_, i) => i);
  return (
    <svg viewBox="0 0 240 188" className="gauge" role="img"
         aria-label={`Fatigue score ${score == null ? "not available" : Math.round(v * 100) + " percent"}, level ${L.word}`}>
      <defs>
        <linearGradient id="gArc" x1="0" x2="1">
          <stop offset="0" stopColor="var(--lv0)" /><stop offset=".45" stopColor="var(--lv1)" />
          <stop offset=".72" stopColor="var(--lv2)" /><stop offset="1" stopColor="var(--lv3)" />
        </linearGradient>
        <filter id="gGlow" x="-30%" y="-30%" width="160%" height="160%"><feGaussianBlur stdDeviation="4" /></filter>
      </defs>
      <path d={arc(A0, A0 + SWEEP)} className="g-track" />
      {score != null && <path d={arc(A0, ang)} stroke="url(#gArc)" className="g-fill" style={{ transition: "d .5s" }} />}
      {score != null && <path d={arc(A0, ang)} stroke={L.color} className="g-glow" filter="url(#gGlow)" />}
      {ticks.map((i) => {
        const a = A0 + (SWEEP * i) / 40, major = i % 10 === 0;
        const [x0, y0] = pol(a, R - 12), [x1, y1] = pol(a, R - (major ? 24 : 18));
        return <line key={i} x1={x0} y1={y0} x2={x1} y2={y1} className={major ? "g-tick major" : "g-tick"} />;
      })}
      {[0, 0.5, 1].map((p) => {
        const [x, y] = pol(A0 + SWEEP * p, R + 15);
        return <text key={p} x={x} y={y + 4} className="g-num" textAnchor="middle">{p === 0 ? "0" : p === 1 ? "1" : ".5"}</text>;
      })}
      <g className="needle" style={{ transform: `rotate(${ang}deg)`, transformOrigin: `${CX}px ${CY}px` }}>
        <path d={`M${CX} ${CY - R + 10} L${CX - 3.5} ${CY} L${CX + 3.5} ${CY} Z`} fill={score == null ? "var(--muted)" : L.color} />
      </g>
      <circle cx={CX} cy={CY} r="9" className="g-hub" />
      <text x={CX} y={CY + 40} textAnchor="middle" className="g-level" fill={L.color}>{L.glyph} {L.name}</text>
      <text x={CX} y={CY + 58} textAnchor="middle" className="g-sub">{label || (score == null ? "no score yet" : `fatigue score ${v.toFixed(2)}`)}</text>
    </svg>
  );
}
