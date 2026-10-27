import { level } from "../format";

/** Top-down cabin. Rings show where an alert is *heard*: the whole cabin (loud), only the driver (private),
 *  or a notification going to a second person / dispatch. This is the project's central design idea. */
export default function CabinMap({ profile, actions = [], pulseKey = 0, compact = false }) {
  const p = profile || {};
  const has = (ch) => actions.filter((a) => a.channel === ch);
  const cabin = has("cabin"), priv = has("private"), crew = has("crew"), disp = has("dispatch");
  const top = actions.reduce((m, a) => Math.max(m, a.level), 0);
  const col = level(top || 1).color;
  const loud = cabin.some((a) => a.volume === "loud");
  return (
    <svg viewBox="0 0 300 250" className={"cabin" + (compact ? " compact" : "")} role="img"
         aria-label="Cabin diagram showing where the alert is heard">
      <defs>
        <filter id="cGlow"><feGaussianBlur stdDeviation="3" /></filter>
      </defs>
      {/* dispatch */}
      {p.has_dispatch && (
        <g className="c-cloud" transform="translate(238 18)">
          <path d="M12 30 q-10 0 -10 -9 q0 -8 9 -9 q3 -10 14 -9 q10 0 12 9 q10 0 10 9 q0 9 -10 9 z" className="c-cloud-shape" />
          <text x="25" y="52" textAnchor="middle" className="c-lab">dispatch</text>
        </g>
      )}
      {/* vehicle */}
      <rect x="40" y="40" width="170" height="190" rx="34" className="c-body" />
      <rect x="52" y="48" width="146" height="26" rx="12" className="c-glass" />
      {/* seats */}
      <g className="c-seat driver"><rect x="62" y="100" width="44" height="50" rx="12" />
        <circle cx="84" cy="92" r="12" className="c-wheel" /><text x="84" y="170" textAnchor="middle" className="c-lab">driver</text></g>
      {p.others_present && !p.second_driver && (
        <g className="c-seat"><rect x="144" y="100" width="44" height="50" rx="12" />
          <text x="166" y="170" textAnchor="middle" className="c-lab">{p.overhearing_harmful ? "patient / children" : "passengers"}</text></g>
      )}
      {p.second_driver && (
        <g className="c-seat second"><rect x="144" y="100" width="44" height="50" rx="12" />
          <text x="166" y="170" textAnchor="middle" className="c-lab">2nd driver</text></g>
      )}
      {p.others_present && p.second_driver && (
        <g className="c-seat"><rect x="100" y="188" width="90" height="30" rx="10" />
          <text x="145" y="208" textAnchor="middle" className="c-lab">{p.overhearing_harmful ? "patient / children" : "passengers"}</text></g>
      )}
      {p.overhearing_harmful && <g className="c-quiet"><text x="125" y="126" textAnchor="middle">🤫</text></g>}

      {/* sound */}
      <g key={pulseKey}>
        {cabin.length > 0 && [0, 1, 2].map((i) => (
          <circle key={i} cx="125" cy="130" r="20" className="ring cabin-ring" stroke={col}
                  style={{ animationDelay: `${i * 0.35}s`, "--to": loud ? 3.6 : 2.4 }} />
        ))}
        {priv.length > 0 && [0, 1].map((i) => (
          <circle key={i} cx="84" cy="116" r="10" className="ring priv-ring" stroke={col} style={{ animationDelay: `${i * 0.4}s` }} />
        ))}
        {crew.length > 0 && <>
          <circle cx="166" cy="125" r="10" className="ring ping" stroke={col} />
          <path d="M96 125 L150 125" className="c-link" stroke={col} />
        </>}
        {disp.length > 0 && <>
          <path d="M125 40 Q190 4 245 34" className="c-link" stroke={col} />
          <circle cx="258" cy="34" r="14" className="ring ping" stroke={col} />
        </>}
      </g>
    </svg>
  );
}
