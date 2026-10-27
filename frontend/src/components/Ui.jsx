import { level } from "../format";

export const Panel = ({ title, aside, children, className = "", ...rest }) => (
  <section className={"panel " + className} {...rest}>
    {(title || aside) && <header className="panel-h"><h3>{title}</h3><div>{aside}</div></header>}
    {children}
  </section>
);

export const LevelChip = ({ lv, small }) => {
  const L = level(lv);
  return <span className={"chip lvl" + (small ? " sm" : "")} style={{ "--c": L.color }}><i aria-hidden>{L.glyph}</i>{L.name}</span>;
};

export const Tag = ({ children, tone = "neutral", title }) => <span className={"tag " + tone} title={title}>{children}</span>;

export const Tile = ({ label, value, sub, tone }) => (
  <div className={"tile" + (tone ? " " + tone : "")}>
    <div className="tile-l">{label}</div><div className="tile-v">{value}</div>{sub && <div className="tile-s">{sub}</div>}
  </div>
);

export const SyntheticBanner = ({ children }) => (
  <div className="banner" role="note">
    <b>SYNTHETIC DATA.</b> {children || "Everything on this page comes from the simulator. It shows the system works end to end; it says nothing about real drivers."}
  </div>
);

export function Seg({ value, onChange, options, label, disabled }) {
  return (
    <div className="seg" role="radiogroup" aria-label={label}>
      {options.map((o) => (
        <button key={o.value} role="radio" aria-checked={value === o.value} disabled={disabled}
                className={value === o.value ? "on" : ""} onClick={() => onChange(o.value)}>{o.label}</button>
      ))}
    </div>
  );
}

export const Field = ({ label, children, hint }) => (
  <label className="field"><span>{label}</span>{children}{hint && <small>{hint}</small>}</label>
);

export const Loading = ({ what = "Loading" }) => <div className="loading"><span className="spinner" />{what}…</div>;

export const ErrorBox = ({ error, onRetry }) => (
  <div className="errorbox" role="alert"><b>Something went wrong.</b> {error}
    {onRetry && <button className="btn ghost" onClick={onRetry}>Retry</button>}</div>
);
