/** Scrolling eye-openness trace (last ~36 s). Dips below the dashed line are closures. */
export default function Trace({ data, closedAt = 0.5, height = 90 }) {
  const W = 600, H = height, n = data.length;
  if (n < 2) return <div className="trace-empty" style={{ height }}>waiting for signal…</div>;
  const y = (v) => H - 6 - Math.min(Math.max(v, 0), 1.3) / 1.3 * (H - 12);
  let d = "", pen = false;
  data.forEach((p, i) => {
    const x = (i / (360 - 1)) * W + (W - (n / 360) * W);
    if (p[1] == null) { pen = false; return; }
    d += `${pen ? "L" : "M"}${x.toFixed(1)} ${y(p[1]).toFixed(1)} `;
    pen = true;
  });
  const gaps = [];
  let s = null;
  data.forEach((p, i) => {
    const x = (i / 359) * W + (W - (n / 360) * W);
    if (p[1] == null && s == null) s = x;
    if ((p[1] != null || i === n - 1) && s != null) { gaps.push([s, x]); s = null; }
  });
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="trace" preserveAspectRatio="none" role="img"
         aria-label="Eye openness over the last 36 seconds">
      {gaps.map(([a, b], i) => <rect key={i} x={a} y="0" width={Math.max(b - a, 1)} height={H} className="trace-gap" />)}
      <line x1="0" x2={W} y1={y(closedAt)} y2={y(closedAt)} className="trace-thr" />
      <line x1="0" x2={W} y1={y(1)} y2={y(1)} className="trace-open" />
      <path d={d} className="trace-line" />
    </svg>
  );
}
