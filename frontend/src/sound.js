// Optional audible feedback in the browser. Off by default (browsers also require a click first).
// Mirrors the backend's routing: the *style* and *volume* come from the action the server chose.
let ctx = null;
const ac = () => (ctx ||= new (window.AudioContext || window.webkitAudioContext)());

function beep(freq, dur, vol, type = "sine", when = 0) {
  const c = ac(), o = c.createOscillator(), g = c.createGain();
  o.type = type; o.frequency.value = freq;
  const t0 = c.currentTime + when;
  g.gain.setValueAtTime(0.0001, t0);
  g.gain.exponentialRampToValueAtTime(vol, t0 + 0.02);
  g.gain.exponentialRampToValueAtTime(0.0001, t0 + dur);
  o.connect(g).connect(c.destination); o.start(t0); o.stop(t0 + dur + 0.05);
}
const VOL = { soft: 0.05, low: 0.1, normal: 0.25, loud: 0.5 };

export function playAction(a) {
  try {
    const v = VOL[a.volume] ?? 0.15;
    if (a.channel === "crew" || a.channel === "dispatch") { beep(660, 0.12, 0.08); beep(880, 0.12, 0.08, "sine", 0.16); return; }
    if (a.style === "chime") { beep(880, 0.25, v); return; }
    if (a.style === "alarm") { for (let i = 0; i < 4; i++) beep(i % 2 ? 740 : 988, 0.16, v, "square", i * 0.2); return; }
    if (a.style === "voice" && "speechSynthesis" in window) {
      const u = new SpeechSynthesisUtterance(a.message);
      u.volume = Math.min(1, v * 2.5); u.rate = 1.02;
      window.speechSynthesis.cancel(); window.speechSynthesis.speak(u);
    }
  } catch { /* audio is a convenience; never break the UI */ }
}
