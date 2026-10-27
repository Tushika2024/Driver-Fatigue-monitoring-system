import { clamp } from "../format";

/** The eye the system is actually reading: lid height follows measured eye-openness (relative to this
 *  driver's own open level), so what you see is the signal, not an animation. */
export default function Eye({ openness, color = "var(--lv0)", lost = false, lostText = "no face", size = 220 }) {
  const o = openness == null ? 0.0 : clamp(openness, 0, 1.25);
  const up = 50 - 40 * Math.min(o, 1.15);          // upper lid y
  const lo = 50 + 26 * Math.min(o, 1);              // lower lid y
  const eyePath = `M14 50 Q100 ${up - 14 * Math.min(o, 1) - 2} 186 50 Q100 ${lo + 12 * Math.min(o, 1) + 2} 14 50 Z`;
  const closed = o < 0.18;
  return (
    <svg viewBox="0 0 200 100" width={size} className={"eye" + (lost ? " lost" : "")} role="img"
         aria-label={lost ? lostText : closed ? "Eyes closed" : `Eye ${Math.round(o * 100)} percent open`}>
      <defs>
        <clipPath id="eyeClip"><path d={eyePath} /></clipPath>
        <radialGradient id="iris" cx=".5" cy=".5" r=".5">
          <stop offset="0" stopColor="#05070d" /><stop offset=".35" stopColor="#05070d" />
          <stop offset=".38" stopColor={color} stopOpacity=".95" /><stop offset="1" stopColor={color} stopOpacity=".25" />
        </radialGradient>
      </defs>
      {!lost && <>
        <path d={eyePath} className="eye-white" />
        <g clipPath="url(#eyeClip)">
          <circle cx="100" cy="50" r="30" fill="url(#iris)" />
          <circle cx="110" cy="40" r="5" fill="#fff" opacity=".7" />
        </g>
        <path d={eyePath} className="eye-line" stroke={color} />
        {closed && <path d="M14 50 Q100 62 186 50" className="eye-line thick" stroke={color} />}
      </>}
      {lost && <text x="100" y="56" textAnchor="middle" className="eye-lost">{lostText}</text>}
    </svg>
  );
}
