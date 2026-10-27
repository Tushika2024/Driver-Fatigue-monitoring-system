"""Command line: fatigue <command>. Run `fatigue --help`."""
from __future__ import annotations

import argparse
import json
import os

import pandas as pd

from .config import load_config


def _cfg(a):
    return load_config(getattr(a, "config", None))


def cmd_extract(a):
    from .extract import extract_all
    extract_all(a.videos, a.out, _cfg(a), [s.strip() for s in a.labels.split(",")], a.jobs,
                max_seconds=a.max_seconds)


def cmd_synth(a):
    from .synthetic import make_dataset
    fr, meta = make_dataset(a.subjects, a.duration, seed=a.seed)
    os.makedirs(a.outdir, exist_ok=True)
    fr.to_csv(os.path.join(a.outdir, "features.csv"), index=False)
    meta.to_csv(os.path.join(a.outdir, "subjects.csv"), index=False)
    print(f"wrote {a.outdir}/features.csv ({len(fr)} rows) and subjects.csv   [SYNTHETIC]")


def cmd_subjects_template(a):
    from .windows import read_frames
    s = sorted(read_frames(a.features)["subject"].unique())
    pd.DataFrame({"subject": s, "glasses": "", "facial_hair": ""}).to_csv(a.out, index=False)
    print(f"wrote {a.out}: fill glasses / facial_hair with 0 or 1 for each subject (objective, from the videos)")


def cmd_evaluate(a):
    from .experiments import run_all
    from .windows import read_frames
    cfg = _cfg(a)
    fr = read_frames(a.features)
    meta = pd.read_csv(a.subjects, dtype={"subject": str}) if a.subjects else None
    if meta is not None:
        meta = meta.dropna(subset=["glasses", "facial_hair"], how="all")
    run_all(fr, meta, cfg, a.outdir, synthetic=a.synthetic, n_jobs=a.jobs)


def cmd_train(a):
    from .training import fit_final
    from .windows import read_frames
    cfg = _cfg(a)
    m = fit_final(read_frames(a.features), cfg, a.jobs)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    m.save(a.out)
    print(f"saved model -> {a.out}  (config hash {cfg.hash()})")


def cmd_profiles(a):
    from .cabin import describe, load_profiles
    for p in load_profiles().values():
        print(describe(p)); print()


def _source(spec: str, cfg, seed: int = 11):
    from .sources import FeatureSource, VideoSource, WebcamSource
    from .synthetic import generate_clip, make_subjects
    if spec.startswith("webcam"):
        idx = int(spec.split(":")[1]) if ":" in spec else 0
        return WebcamSource(idx, cfg.extraction.fps), True
    if spec.startswith("video:"):
        return VideoSource(spec[6:], cfg.extraction.fps), True
    if spec.startswith("synthetic"):
        kind = spec.split(":")[1] if ":" in spec else "prog"
        sp = make_subjects(1, seed)[0]
        clip = generate_clip(sp, kind, 240, cfg.extraction.fps, seed)
        return FeatureSource(clip.to_dict("records")), False
    raise SystemExit(f"unknown source '{spec}' (webcam[:N] | video:PATH | synthetic[:alert|drowsy|prog])")


def cmd_live(a):
    from .cabin import get_profile
    from .extractor import LandmarkExtractor
    from .live import LiveSession
    from .models import FusionModel
    from .sinks import SinkRouter
    model = FusionModel.load(a.model)
    cfg = model.cfg
    src, needs_extractor = _source(a.source, cfg)
    ex = LandmarkExtractor(cfg) if needs_extractor else None
    sess = LiveSession(model, get_profile(a.profile), SinkRouter.default(audio=a.audio), a.driver, a.vehicle,
                       a.route, a.log, ex, synthetic=a.source.startswith("synthetic"))
    ups = sess.run(src, a.max_seconds, show=a.show,
                   ack_when=(lambda u: True) if a.auto_ack else None)
    print(json.dumps(sess.perf.summary(), indent=2))
    print(f"{len(ups)} windows scored; log -> {a.log}" if a.log else f"{len(ups)} windows scored")


def cmd_pretrip(a):
    from .pretrip import HistoryStore, PretripSession
    cfg = _cfg(a)
    src, needs_ex = _source(a.source, cfg, seed=a.seed)
    ex = None
    if needs_ex:
        from .extractor import LandmarkExtractor
        ex = LandmarkExtractor(cfg)
    sess = PretripSession(cfg, a.driver, HistoryStore(a.history_dir))
    print(f"Pre-trip check for {a.driver}: look at the camera, relax, blink normally ...")
    for fr in src:
        f = fr.features if fr.features is not None else ex.process(fr.image, fr.t)
        sess.push(f)
        if sess.done:
            break
    r = sess.result()
    print(f"\nVERDICT: {r.verdict}" + (f"   index {r.index:.2f}" if r.index is not None else ""))
    print(r.reason)
    for w in r.warnings:
        print("WARNING:", w)
    print("(advisory only: this never blocks the shift)")


def cmd_report(a):
    from .report import read_log, render_depot_html, render_driver_html, depot_table
    cfg = _cfg(a)
    os.makedirs(a.outdir, exist_ok=True)
    df, meta = read_log(a.log)
    p = os.path.join(a.outdir, "driver_report.html")
    open(p, "w").write(render_driver_html(df, meta, cfg))
    print("wrote", p)
    if a.depot_glob:
        import glob
        files = sorted(glob.glob(a.depot_glob))
        logs = [read_log(f)[0] for f in files]
        synth = any(read_log(f)[1].get("synthetic") for f in files)
        q = os.path.join(a.outdir, "depot_report.html")
        open(q, "w").write(render_depot_html(depot_table(logs, cfg), {"synthetic": synth}, cfg))
        print(f"wrote {q} from {len(files)} logs (aggregate only)")


def cmd_camera_check(a):
    from .camera_check import run
    run(_cfg(a), a.index)


def cmd_demo(a):
    from .demo import run_demo
    run_demo(a.outdir, with_eval=a.with_eval, jobs=a.jobs, profile=a.profile)


def main(argv=None):
    p = argparse.ArgumentParser(prog="fatigue", description=__doc__)
    p.add_argument("--config", help="YAML overriding configs (see src/fatigue/data/default.yaml)")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("extract", help="videos -> features.csv (MediaPipe, CPU)")
    s.add_argument("--videos", required=True); s.add_argument("--out", default="features.csv")
    s.add_argument("--labels", default="0,10", help="comma list of labels to extract (label 5 is dropped)")
    s.add_argument("--jobs", type=int, default=1)
    s.add_argument("--max-seconds", type=float, default=0.0,
                   help="only the first N seconds of each video (QUICK TEST; use a separate --out, never for the real run)")
    s.set_defaults(f=cmd_extract)

    s = sub.add_parser("synth", help="write a SYNTHETIC features.csv + subjects.csv")
    s.add_argument("--outdir", default="data/synth"); s.add_argument("--subjects", type=int, default=10)
    s.add_argument("--duration", type=float, default=240); s.add_argument("--seed", type=int, default=0)
    s.set_defaults(f=cmd_synth)

    s = sub.add_parser("subjects-template", help="make subjects.csv to fill in glasses / facial_hair")
    s.add_argument("--features", required=True); s.add_argument("--out", default="subjects.csv")
    s.set_defaults(f=cmd_subjects_template)

    s = sub.add_parser("evaluate", help="leave-one-person-out experiments, ablations, fairness, figures")
    s.add_argument("--features", required=True); s.add_argument("--subjects")
    s.add_argument("--outdir", default="results"); s.add_argument("--jobs", type=int, default=1)
    s.add_argument("--synthetic", action="store_true", help="stamp outputs as synthetic")
    s.set_defaults(f=cmd_evaluate)

    s = sub.add_parser("train", help="fit the deployable model on all subjects")
    s.add_argument("--features", required=True); s.add_argument("--out", default="models/bundle.joblib")
    s.add_argument("--jobs", type=int, default=1); s.set_defaults(f=cmd_train)

    s = sub.add_parser("profiles", help="show the cabin-context escalation tables")
    s.set_defaults(f=cmd_profiles)

    s = sub.add_parser("live", help="run the monitor on webcam / video / synthetic")
    s.add_argument("--model", required=True)
    s.add_argument("--source", default="webcam", help="webcam[:N] | video:PATH | synthetic[:kind]")
    s.add_argument("--profile", default=None, help="vehicle profile (see `fatigue profiles`); default: unset_private")
    s.add_argument("--driver", default="driver"); s.add_argument("--vehicle", default="vehicle")
    s.add_argument("--route", default="route"); s.add_argument("--log", default="runs/session_log.csv")
    s.add_argument("--max-seconds", type=float); s.add_argument("--show", action="store_true")
    s.add_argument("--audio", action="store_true", help="speak alerts (needs pyttsx3)")
    s.add_argument("--auto-ack", action="store_true", help="demo: acknowledge SEVERE immediately")
    s.set_defaults(f=cmd_live)

    s = sub.add_parser("pretrip", help="60 s pre-trip fitness check (advisory)")
    s.add_argument("--driver", required=True); s.add_argument("--history-dir", default="runs/history")
    s.add_argument("--source", default="webcam"); s.add_argument("--seed", type=int, default=11)
    s.set_defaults(f=cmd_pretrip)

    s = sub.add_parser("report", help="driver shift report from session_log.csv")
    s.add_argument("--log", required=True); s.add_argument("--outdir", default="report")
    s.add_argument("--depot-glob", help="e.g. 'logs/*.csv' -> also write the aggregate depot page")
    s.set_defaults(f=cmd_report)

    s = sub.add_parser("camera-check", help="live EAR/MAR/pose overlay to confirm the signal exists on your camera")
    s.add_argument("--index", type=int, default=0); s.set_defaults(f=cmd_camera_check)

    s = sub.add_parser("demo", help="END-TO-END on synthetic data: no dataset or webcam needed")
    s.add_argument("--outdir", default="demo"); s.add_argument("--with-eval", action="store_true")
    s.add_argument("--jobs", type=int, default=1); s.add_argument("--profile", default="ambulance_patient")
    s.set_defaults(f=cmd_demo)

    a = p.parse_args(argv)
    a.f(a)


if __name__ == "__main__":
    main()
