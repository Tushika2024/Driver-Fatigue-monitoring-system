"""End-to-end demonstration on SYNTHETIC data: no dataset and no webcam needed.

train model -> pre-trip checks (FIT then deliberately tired) -> live monitored shift with a
cabin profile -> driver report + (illustrative) depot report. Everything is stamped synthetic.
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta

import numpy as np

from .cabin import get_profile
from .config import load_config
from .live import LiveSession
from .pretrip import HistoryStore, PretripSession
from .report import (depot_table, read_log, render_depot_html, render_driver_html, synthetic_depot_logs)
from .sinks import RecordingSink, SinkRouter, ConsoleSink
from .sources import FeatureSource
from .synthetic import generate_clip, make_dataset, make_subjects
from .training import fit_final


def run_demo(outdir: str = "demo", with_eval: bool = False, jobs: int = 1,
             profile: str = "ambulance_patient", log=print) -> dict:
    os.makedirs(outdir, exist_ok=True)
    cfg = load_config()
    log("[synthetic] generating 8 training drivers ...")
    frames, meta = make_dataset(8, duration_s=240, seed=21)
    frames.to_csv(os.path.join(outdir, "synthetic_features.csv"), index=False)
    log("[train] fitting the model on all training drivers ...")
    model = fit_final(frames, cfg, n_jobs=jobs)
    model.save(os.path.join(outdir, "model.joblib"))

    # a NEW driver the model has never seen
    sp = make_subjects(1, seed=777)[0]
    sp.subject = "NEW01"

    log("[pre-trip] building history (4 normal sessions), then a normal and a tired check ...")
    store = HistoryStore(os.path.join(outdir, "history"))
    base_day = datetime(2026, 3, 2, 8, 0)
    results = {}
    for i in range(4):
        s = PretripSession(cfg, sp.subject, store, base_day + timedelta(days=i))
        for r in generate_clip(sp, "alert", 100, cfg.extraction.fps, seed=100 + i).to_dict("records"):
            s.push(r)
            if s.done:
                break
        results[f"history_{i}"] = s.result().verdict
    for name, kind in (("normal day", "alert"), ("tired day", "drowsy")):
        s = PretripSession(cfg, sp.subject, store, base_day + timedelta(days=9))
        for r in generate_clip(sp, kind, 100, cfg.extraction.fps, seed=300 + (kind == "drowsy")).to_dict("records"):
            s.push(r)
            if s.done:
                break
        r = s.result()
        log(f"   {name}: {r.verdict}  {r.reason}")
        results[name] = r.verdict

    log(f"[live] a compressed 20-minute shift, vehicle profile '{profile}' ...")
    t = np.arange(0, 1200, 1 / cfg.extraction.fps)
    curve = np.clip(0.05 + 0.85 * np.clip((t - 300) / 600.0, 0, 1), 0, 1)
    clip = generate_clip(sp, "prog", 1200, cfg.extraction.fps, seed=5, d_override=curve)
    rec = RecordingSink()
    sinks = SinkRouter({c: [rec, ConsoleSink()] for c in ("cabin", "private", "crew", "dispatch")}, rec)
    log_path = os.path.join(outdir, "session_log.csv")
    sess = LiveSession(model, get_profile(profile), sinks, "NEW01", "TRUCK-7", "Ring", log_path,
                       start=datetime(2026, 3, 11, 2, 0), synthetic=True)
    ups = sess.run(FeatureSource(clip.to_dict("records")), ack_when=lambda u: u.t % 60 < 3)
    perf = sess.perf.summary()
    log(f"   {len(ups)} windows, {len(rec.actions)} alert actions; "
        f"processing {perf['fps']:.0f} frames/s, p95 latency {perf['latency_ms_p95']:.2f} ms")

    log("[report] driver page and depot page ...")
    df, m = read_log(log_path)
    open(os.path.join(outdir, "driver_report.html"), "w").write(render_driver_html(df, m, cfg))
    logs = synthetic_depot_logs(12, 6, seed=3)
    tbl = depot_table(logs, cfg)
    open(os.path.join(outdir, "depot_report.html"), "w").write(render_depot_html(tbl, {"synthetic": True}, cfg))

    if with_eval:
        from .experiments import run_all
        run_all(frames, meta, cfg, os.path.join(outdir, "results"), synthetic=True, n_jobs=jobs, log=log)
    log(f"done -> {outdir}/")
    return {"pretrip": results, "perf": perf, "n_actions": len(rec.actions)}
