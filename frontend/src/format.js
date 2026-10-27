export const LEVELS = [
  { id: 0, name: "CLEAR",    word: "Clear",    color: "var(--lv0)", glyph: "●", hint: "No fatigue signs" },
  { id: 1, name: "MILD",     word: "Mild",     color: "var(--lv1)", glyph: "▲", hint: "Early signs only" },
  { id: 2, name: "MODERATE", word: "Moderate", color: "var(--lv2)", glyph: "◆", hint: "Early and late signs" },
  { id: 3, name: "SEVERE",   word: "Severe",   color: "var(--lv3)", glyph: "■", hint: "Pull over safely" },
  { id: 4, name: "CRITICAL", word: "Critical", color: "var(--lv4)", glyph: "✖", hint: "Unacknowledged severe" },
];
export const level = (i) => LEVELS[Math.max(0, Math.min(4, i | 0))];

export function clock(s) {
  if (s == null || !isFinite(s)) return "–:––";
  s = Math.max(0, Math.round(s));
  const m = Math.floor(s / 60), r = s % 60;
  return `${m}:${String(r).padStart(2, "0")}`;
}
export const pct = (v, d = 0) => (v == null || !isFinite(v) ? "–" : `${(v * 100).toFixed(d)}%`);
export const num = (v, d = 2) => (v == null || !isFinite(v) ? "–" : Number(v).toFixed(d));
export const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
export const lerp = (a, b, t) => a + (b - a) * t;

// "slot_label" strings like "02-06h" sort fine as text; keep a helper for readable names.
export const MODE_COPY = {
  warmup:   { label: "Calibrating", note: "Learning this driver's own normal before scoring. No alerts fire until it is ready." },
  full:     { label: "Eyes + head", note: "Full model: eyes, blinks, head and posture." },
  degraded: { label: "Degraded",    note: "Eyes unreadable (glare, glasses, shadow). Using head and posture only, with wider thresholds." },
  blind:    { label: "Can't see driver", note: "No face in view. The system says so instead of guessing." },
};
