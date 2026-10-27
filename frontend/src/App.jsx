import { Component, useEffect, useState } from "react";
import { api } from "./api";
import Live from "./pages/Live";
import Pretrip from "./pages/Pretrip";
import Report from "./pages/Report";
import Depot from "./pages/Depot";
import Evaluation from "./pages/Evaluation";
import Cabin from "./pages/Cabin";
import About from "./pages/About";

const ROUTES = [
  { path: "live", label: "Drive", icon: "◉", el: () => <Live /> },
  { path: "pretrip", label: "Pre-trip", icon: "◌", el: () => <Pretrip /> },
  { path: "report", label: "Shift report", icon: "≋", el: (a) => <Report id={a} /> },
  { path: "depot", label: "Depot", icon: "▦", el: () => <Depot /> },
  { path: "cabin", label: "Cabin", icon: "◫", el: () => <Cabin /> },
  { path: "evaluation", label: "Evidence", icon: "∿", el: () => <Evaluation /> },
  { path: "about", label: "About", icon: "i", el: () => <About /> },
];

function useHash() {
  const [h, setH] = useState(location.hash || "#/live");
  useEffect(() => {
    const f = () => setH(location.hash || "#/live");
    window.addEventListener("hashchange", f);
    return () => window.removeEventListener("hashchange", f);
  }, []);
  const [, path = "live", arg] = h.replace(/^#/, "").split("/");
  return { path, arg };
}

class Boundary extends Component {
  state = { err: null };
  static getDerivedStateFromError(err) { return { err }; }
  componentDidUpdate(p) { if (p.k !== this.props.k && this.state.err) this.setState({ err: null }); }
  render() {
    if (this.state.err) return <div className="errorbox" role="alert"><b>This page crashed.</b> {String(this.state.err.message || this.state.err)}</div>;
    return this.props.children;
  }
}

function Status() {
  const [h, setH] = useState(null);
  const [down, setDown] = useState(false);
  useEffect(() => {
    let dead = false, timer;
    const tick = async () => {
      try { const r = await api.health(); if (!dead) { setH(r); setDown(false); } } catch { if (!dead) setDown(true); }
      if (!dead) timer = setTimeout(tick, 4000);
    };
    tick();
    return () => { dead = true; clearTimeout(timer); };
  }, []);
  if (down) return <span className="status bad" role="status">backend offline</span>;
  if (!h) return <span className="status" role="status">connecting…</span>;
  const m = h.model;
  if (m.state === "training") return <span className="status warn" role="status">training demo model… (first start)</span>;
  if (m.state === "loading") return <span className="status warn" role="status">loading model…</span>;
  if (m.state === "error") return <span className="status bad" role="status" title={m.error}>model error</span>;
  return <span className={"status " + (m.synthetic ? "warn" : "ok")} role="status">{m.synthetic ? "demo model · synthetic" : "model ready"}</span>;
}

export default function App() {
  const { path, arg } = useHash();
  const route = ROUTES.find((r) => r.path === path) || ROUTES[0];
  return (
    <div className="app">
      <header className="top">
        <a className="brand" href="#/live" aria-label="Night Watch home">
          <svg viewBox="0 0 32 32" width="28" height="28" aria-hidden><path d="M3 16c4-8 22-8 26 0-4 8-22 8-26 0z" fill="none" stroke="currentColor" strokeWidth="2" /><circle cx="16" cy="16" r="4.5" fill="currentColor" /></svg>
          <span>Night<b>Watch</b></span>
        </a>
        <nav aria-label="Sections">
          {ROUTES.map((r) => (
            <a key={r.path} href={`#/${r.path}`} className={r.path === route.path ? "on" : ""} aria-current={r.path === route.path ? "page" : undefined}>
              <i aria-hidden>{r.icon}</i><span>{r.label}</span>
            </a>
          ))}
        </nav>
        <Status />
      </header>
      <main>
        <Boundary k={route.path}>{route.el(arg)}</Boundary>
      </main>
      <footer>Driving-safety aid, not a medical or disciplinary tool · no video is stored · demo data is always labelled <b>synthetic</b></footer>
    </div>
  );
}
