import json

import pandas as pd

from fatigue.cabin import get_profile
from fatigue.live import LiveSession
from fatigue.sinks import RecordingSink, SinkRouter
from fatigue.sources import FeatureSource
from fatigue.synthetic import generate_clip, make_subjects


def test_live_session_end_to_end(cfg, model, tmp_path):
    sp = make_subjects(1, 99)[0]
    clip = generate_clip(sp, "drowsy", 200, 30.0, 3)
    rec = RecordingSink()
    sinks = SinkRouter({c: [rec] for c in ("cabin", "private", "crew", "dispatch")}, rec)
    log = str(tmp_path / "log.csv")
    s = LiveSession(model, get_profile("solo_goods_truck"), sinks, "d", "v", "r", log, synthetic=True)
    ups = s.run(FeatureSource(clip.to_dict("records")))
    assert ups and max(u.level for u in ups) >= 2
    assert rec.actions and all(a.channel == "cabin" for a in rec.actions)       # solo truck: cabin audio only
    df = pd.read_csv(log)
    assert list(df.columns)[:3] == ["t", "ts", "driver_id"] and df["mode"].isin(["warmup", "full", "degraded", "blind"]).all()
    assert json.load(open(log + ".meta.json"))["synthetic"] is True
    assert s.perf.summary()["latency_ms_p95"] < 50                                 # real-time budget at 30 FPS


def test_alert_driver_gets_almost_no_alerts(cfg, model):
    clip = generate_clip(make_subjects(1, 98)[0], "alert", 240, 30.0, 4)
    rec = RecordingSink()
    s = LiveSession(model, get_profile("solo_goods_truck"), SinkRouter({}, rec), "d")
    ups = s.run(FeatureSource(clip.to_dict("records")))
    frac = sum(u.level >= 2 for u in ups) / len(ups)
    assert frac < 0.15


def test_blind_driver_triggers_visibility_notice(cfg, model):
    rows = generate_clip(make_subjects(1, 97)[0], "alert", 120, 30.0, 4).to_dict("records")
    for r in rows:
        if r["t"] > 70:
            r["face_found"] = 0
    rec = RecordingSink()
    s = LiveSession(model, get_profile("solo_goods_truck"), SinkRouter({}, rec), "d")
    s.run(FeatureSource(rows))
    assert any("not visible" in a.message for a in rec.actions)
    assert not any(a.level >= 2 for a in rec.actions)       # blindness is announced, not mistaken for drowsiness


def test_cli_synth_train_profiles(tmp_path, capsys):
    from fatigue.cli import main
    main(["synth", "--outdir", str(tmp_path), "--subjects", "4", "--duration", "150"])
    main(["train", "--features", str(tmp_path / "features.csv"), "--out", str(tmp_path / "m.joblib")])
    main(["profiles"])
    assert "ambulance_patient" in capsys.readouterr().out
    main(["live", "--model", str(tmp_path / "m.joblib"), "--source", "synthetic:prog", "--max-seconds", "100",
          "--log", str(tmp_path / "s.csv"), "--profile", "bus"])
    main(["report", "--log", str(tmp_path / "s.csv"), "--outdir", str(tmp_path / "rep")])
    assert (tmp_path / "rep" / "driver_report.html").exists()
