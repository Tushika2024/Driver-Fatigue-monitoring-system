"""Shift report from session_log.csv.

Driver page: trend, worst stretch, alert counts, lead time. Depot page: aggregate only
(by time slot and route), never a per-driver leaderboard, small cells suppressed (k-anonymity).
Synthetic logs are stamped on every page.
"""
from __future__ import annotations

import base64
import html
import io
import json
from datetime import datetime, timedelta
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import lfilter

from .alerts import AlertPolicy
from .config import AlertsCfg, Config
from .plots import BLUE, GRID, INK, MUTED, ORANGE
from .schema import LEVEL_NAMES

LOG_COLUMNS = ["t", "ts", "driver_id", "vehicle_id", "route", "score", "early_score", "late_score",
               "level", "mode", "responded", "reason"]


def write_meta(log_path: str, **meta) -> None:
    with open(log_path + ".meta.json", "w") as fh:
        json.dump(meta, fh, indent=2)


def read_log(path: str):
    df = pd.read_csv(path)
    try:
        with open(path + ".meta.json") as fh:
            meta = json.load(fh)
    except FileNotFoundError:
        meta = {}
    return df, meta


def _onsets(level: np.ndarray, at_least: int) -> np.ndarray:
    on = level >= at_least
    return np.where(on & ~np.r_[False, on[:-1]])[0]


def episode_starts(t: np.ndarray, level: np.ndarray, at_least: int, merge_s: float) -> np.ndarray:
    """Indices where an episode (level >= at_least) begins. Alerts separated by less than merge_s
    seconds belong to the same episode, so flicker around a threshold is not counted repeatedly."""
    out, last_on = [], -1e18
    for i in _onsets(level, at_least):
        if t[i] - last_on >= merge_s:
            out.append(i)
        # extend the 'last on' marker to the end of this stretch
        j = i
        while j < len(level) and level[j] >= at_least:
            j += 1
        last_on = t[min(j, len(t)) - 1]
    return np.array(out, dtype=int)


def driver_metrics(df: pd.DataFrame, cfg: Config) -> Dict:
    if df.empty:
        return {"empty": True}
    t, lvl = df["t"].to_numpy(float), df["level"].to_numpy(int)
    full_frac = float((df["mode"] == "full").mean())
    # one episode = a continuous stretch with level > 0, counted at its peak level
    peaks, i = [], 0
    while i < len(lvl):
        if lvl[i] > 0:
            j = i
            while j < len(lvl) and lvl[j] > 0:
                j += 1
            peaks.append((t[i], int(lvl[i:j].max()), t[i], t[min(j, len(t) - 1)]))
            i = j
        else:
            i += 1
    counts = {LEVEL_NAMES[k]: sum(1 for p in peaks if p[1] == k) for k in (1, 2, 3, 4)}
    # lead time: first MILD to first SEVERE inside an episode
    leads = []
    i = 0
    while i < len(lvl):
        if lvl[i] > 0:
            j = i
            while j < len(lvl) and lvl[j] > 0:
                j += 1
            sev = np.where(lvl[i:j] >= 3)[0]
            if len(sev):
                leads.append(float(t[i + sev[0]] - t[i]))
            i = j
        else:
            i += 1
    # worst stretch: highest 10-minute rolling mean score
    step = float(np.median(np.diff(t))) if len(t) > 1 else 3.0
    w = max(1, int(round(600.0 / step)))
    s = df["score"].astype(float).interpolate().bfill().to_numpy()
    roll = pd.Series(s).rolling(w, min_periods=max(1, w // 3)).mean().to_numpy()
    if np.isfinite(roll).any():
        k = int(np.nanargmax(roll))
        worst = {"start_s": float(t[max(0, k - w + 1)]), "end_s": float(t[k]), "mean_score": float(roll[k])}
    else:
        worst = None          # nothing scored yet (e.g. the whole log is warm-up)
    acks = int((df["responded"] == 1).sum()) if "responded" in df else 0
    n_mod = sum(1 for p in peaks if p[1] >= 2)
    return {"empty": False, "duration_min": float((t[-1] - t[0]) / 60.0), "full_mode_frac": full_frac,
            "low_coverage": full_frac < cfg.report.min_full_mode_frac, "alert_episodes": counts,
            "median_lead_to_severe_s": float(np.median(leads)) if leads else None,
            "worst_stretch": worst,
            "driver_response_rate": (min(1.0, acks / n_mod) if n_mod else None)}


def _fig_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor="#fcfcfb")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def trend_figure(df: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(8, 3.2), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    t = df["t"].to_numpy(float) / 60.0
    s = df["score"].astype(float).interpolate().bfill().fillna(0.0)
    sm = s.rolling(max(1, int(round(300 / max(np.median(np.diff(df["t"])), 1)))), min_periods=1).mean()
    ax.plot(t, s, color=GRID, lw=0.8)
    ax.plot(t, sm, color=BLUE, lw=2)
    lv = df["level"].to_numpy(int)
    for k, c in ((2, ORANGE), (3, "#e34948")):
        m = lv >= k
        if m.any():
            ax.scatter(t[m], np.full(m.sum(), 1.03), s=6, color=c, marker="s")
    ax.set_ylim(0, 1.08); ax.set_xlabel("minutes into shift", color=MUTED, fontsize=9)
    ax.set_ylabel("fatigue score", color=MUTED, fontsize=9)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.text(t[-1], sm.iloc[-1], "  5-min average", color=BLUE, fontsize=8, va="center")
    return _fig_b64(fig)


_CSS = """body{font:15px/1.5 system-ui,sans-serif;max-width:860px;margin:2rem auto;padding:0 16px;color:#0b0b0b;background:#fcfcfb}
h1{font-size:1.4rem}h2{font-size:1.1rem;margin-top:1.8rem}.banner{background:#fff3cd;border:1px solid #e6c200;padding:8px 12px;border-radius:6px;margin:12px 0}
.warn{background:#fde8e8;border:1px solid #e34948;padding:8px 12px;border-radius:6px;margin:12px 0}
.cards{display:flex;flex-wrap:wrap;gap:10px}.card{border:1px solid #e4e3df;border-radius:8px;padding:10px 14px;min-width:140px}
.card b{display:block;font-size:1.3rem}table{border-collapse:collapse;width:100%}td,th{border-bottom:1px solid #e4e3df;padding:6px 8px;text-align:left;font-size:.9rem}
img{max-width:100%}.muted{color:#52514e;font-size:.85rem}
@media (prefers-color-scheme:dark){body{background:#1a1a19;color:#f0efec}.card,td,th{border-color:#3a3a38}.muted{color:#c3c2b7}}"""


def render_driver_html(df: pd.DataFrame, meta: Dict, cfg: Config) -> str:
    m = driver_metrics(df, cfg)
    h = ["<!doctype html><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>",
         f"<title>Shift report</title><style>{_CSS}</style>", "<h1>Shift report</h1>"]
    if meta.get("synthetic"):
        h.append("<div class=banner><b>SYNTHETIC DATA.</b> Generated by a simulator to show the page. "
                 "Not a real driver.</div>")
    if m.get("empty"):
        h.append("<p>No data in this log.</p>")
        return "".join(h)
    h.append(f"<p class=muted>Driver {html.escape(str(meta.get('driver_id', df['driver_id'].iloc[0])))} · "
             f"route {html.escape(str(meta.get('route', df['route'].iloc[0])))} · "
             f"{m['duration_min']:.0f} min</p>")
    if m["low_coverage"]:
        h.append(f"<div class=warn><b>Low coverage.</b> Only {m['full_mode_frac']:.0%} of the shift was monitored "
                 f"with full eye tracking (needs {cfg.report.min_full_mode_frac:.0%}). Treat this report with caution.</div>")
    ws = m["worst_stretch"]
    worst_txt = f"{ws['start_s'] / 60:.0f}–{ws['end_s'] / 60:.0f} min" if ws else "n/a"
    lead = f"{m['median_lead_to_severe_s']:.0f} s" if m["median_lead_to_severe_s"] is not None else "n/a"
    resp = f"{m['driver_response_rate']:.0%}" if m["driver_response_rate"] is not None else "n/a"
    h.append("<div class=cards>"
             f"<div class=card>Full-mode coverage<b>{m['full_mode_frac']:.0%}</b></div>"
             f"<div class=card>Worst 10 min<b>{worst_txt}</b></div>"
             f"<div class=card>Warning before SEVERE<b>{lead}</b></div>"
             f"<div class=card>Driver responded<b>{resp}</b></div></div>")
    h.append("<h2>Fatigue over the shift</h2>"
             f"<img alt='fatigue score over time' src='data:image/png;base64,{trend_figure(df)}'>")
    h.append("<h2>Alerts</h2><table><tr><th>Level</th><th>Episodes</th></tr>")
    for k, v in m["alert_episodes"].items():
        h.append(f"<tr><td>{k}</td><td>{v}</td></tr>")
    h.append("</table><p class=muted>Episode = one stretch of continuous alerting, counted at its highest level.</p>")
    return "".join(h)


# ----------------------------------------------------------------------------- depot
def slot_label(slot: int, hours: int) -> str:
    return f"{slot * hours:02d}-{(slot + 1) * hours:02d}h"


def depot_table(logs: List[pd.DataFrame], cfg: Config) -> pd.DataFrame:
    """Alert episodes (MODERATE+) per 100 monitored hours, by time slot x route.
    Cells with fewer than k distinct drivers are suppressed. No driver id leaves this function."""
    k, sh = cfg.report.k_anonymity, cfg.report.slot_hours
    rows = []
    for df in logs:
        if df.empty:
            continue
        ts = pd.to_datetime(df["ts"])
        step_h = float(np.median(np.diff(df["t"]))) / 3600.0 if len(df) > 1 else 3.0 / 3600
        slot = (ts.dt.hour // sh).to_numpy()
        lvl = df["level"].to_numpy(int)
        on = episode_starts(df["t"].to_numpy(float), lvl, 2, cfg.report.episode_merge_s)
        for s in np.unique(slot):
            m = slot == s
            rows.append({"driver": df["driver_id"].iloc[0], "route": df["route"].iloc[0], "slot": int(s),
                         "hours": float(m.sum() * step_h), "events": int(sum(1 for i in on if m[i])),
                         "alert_hours": float(((lvl >= 2) & m).sum() * step_h)})
    if not rows:
        return pd.DataFrame(columns=["slot", "slot_label", "route", "events_per_100h", "pct_time_alert",
                                     "hours", "drivers", "suppressed"])
    r = pd.DataFrame(rows)
    g = r.groupby(["slot", "route"]).agg(events=("events", "sum"), hours=("hours", "sum"),
                                          alert_hours=("alert_hours", "sum"),
                                          drivers=("driver", "nunique")).reset_index()
    ok = g["drivers"] >= k
    g["events_per_100h"] = np.where(ok, 100.0 * g["events"] / g["hours"].replace(0, np.nan), np.nan)
    g["pct_time_alert"] = np.where(ok, 100.0 * g["alert_hours"] / g["hours"].replace(0, np.nan), np.nan)
    g["suppressed"] = g["drivers"] < k
    g["slot_label"] = g["slot"].map(lambda s: slot_label(int(s), sh))
    return g[["slot", "slot_label", "route", "events_per_100h", "pct_time_alert", "hours", "drivers", "suppressed"]]


def render_depot_html(tbl: pd.DataFrame, meta: Dict, cfg: Config) -> str:
    h = ["<!doctype html><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>",
         f"<title>Depot report</title><style>{_CSS}</style>", "<h1>Depot report (aggregate)</h1>"]
    if meta.get("synthetic"):
        h.append("<div class=banner><b>ILLUSTRATIVE, SYNTHETIC DATA.</b> The pattern shown was built into the "
                 "simulator to demonstrate the page. It is <b>not</b> a finding and is not validated.</div>")
    h.append(f"<p class=muted>Share of monitored time at MODERATE or higher, and alert episodes per 100 monitored hours, "
             f"by time of day and route. Cells with fewer than {cfg.report.k_anonymity} drivers are hidden. "
             f"No individual driver is shown.</p>")
    if tbl.empty:
        return "".join(h + ["<p>No data.</p>"])
    piv = tbl.pivot(index="slot_label", columns="route", values="pct_time_alert").sort_index()
    fig, ax = plt.subplots(figsize=(6.5, 0.7 * len(piv) + 1.4), facecolor="#fcfcfb")
    vals = piv.to_numpy(float)
    im = ax.imshow(np.where(np.isfinite(vals), vals, np.nan), cmap="Blues", aspect="auto")
    ax.set_xticks(range(piv.shape[1])); ax.set_xticklabels(piv.columns, fontsize=8)
    ax.set_yticks(range(piv.shape[0])); ax.set_yticklabels(piv.index, fontsize=8)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = vals[i, j]
            ax.text(j, i, "hidden" if not np.isfinite(v) else f"{v:.0f}%", ha="center", va="center",
                    fontsize=8, color=INK if not np.isfinite(v) or v < np.nanmax(vals) * 0.6 else "white")
    fig.colorbar(im, ax=ax, label="% of time at MODERATE+")
    h.append(f"<img alt='alert episodes by time slot and route' src='data:image/png;base64,{_fig_b64(fig)}'>")
    h.append("<h2>Table</h2><table><tr><th>Time slot</th><th>Route</th><th>% time MODERATE+</th><th>Episodes / 100 h</th><th>Hours</th><th>Drivers</th></tr>")
    for r in tbl.sort_values(["slot", "route"]).itertuples():
        v = "hidden (small group)" if r.suppressed else f"{r.pct_time_alert:.1f}%"
        e = "" if r.suppressed else f"{r.events_per_100h:.1f}"
        h.append(f"<tr><td>{r.slot_label}</td><td>{html.escape(str(r.route))}</td><td>{v}</td><td>{e}</td>"
                 f"<td>{r.hours:.1f}</td><td>{r.drivers}</td></tr>")
    h.append("</table>")
    return "".join(h)


def synthetic_depot_logs(n_drivers: int = 12, shifts_per_driver: int = 6, seed: int = 0,
                         routes=("North", "Harbour", "Ring")) -> List[pd.DataFrame]:
    """ILLUSTRATIVE logs: fatigue is built to rise in the 02-06h slot and on 'Ring'. Not a finding."""
    rng = np.random.default_rng(seed)
    logs = []
    base = datetime(2026, 1, 5, 0, 0)
    for d in range(n_drivers):
        for s in range(shifts_per_driver):
            route = routes[int(rng.integers(len(routes)))]
            start = base + timedelta(days=int(rng.integers(0, 28)), hours=int(rng.choice([6, 10, 14, 18, 22, 1])))
            n = int(4 * 3600 / 3)
            t = np.arange(n) * 3.0
            hours = ((start.hour + t / 3600) % 24)
            risk = 0.15 + 0.35 * ((hours >= 2) & (hours < 6)) + 0.15 * (route == "Ring") + 0.04 * (t / 3600)
            drift = lfilter([1.0], [1.0, -0.998], rng.normal(0, 0.003, n))      # slow ~25-minute wander (std ~0.05)
            score = np.clip(risk + drift + rng.normal(0, 0.015, n), 0, 1)
            raw = np.where(score > 0.8, 3, np.where(score > 0.6, 2, np.where(score > 0.45, 1, 0)))
            pol = AlertPolicy(AlertsCfg())                  # same persistence + hysteresis as the live system
            lvl = np.array([pol.update(float(ti), int(r)).level for ti, r in zip(t, raw)])
            logs.append(pd.DataFrame({
                "t": t, "ts": [(start + timedelta(seconds=float(x))).isoformat() for x in t],
                "driver_id": f"D{d + 1:02d}", "vehicle_id": f"V{d % 5 + 1}", "route": route, "score": score,
                "early_score": score, "late_score": score, "level": lvl, "mode": "full",
                "responded": np.nan, "reason": ""}))
    return logs
