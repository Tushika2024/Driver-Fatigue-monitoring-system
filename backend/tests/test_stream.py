
import numpy as np
import pandas as pd
import pytest

from fatigue.stream import StreamProcessor
from fatigue.synthetic import generate_clip, make_subjects


def clip(kind="alert", dur=150, seed=1):
    return generate_clip(make_subjects(1, 9)[0], kind, dur, 30.0, seed)


def test_deterministic(cfg):
    rows = clip().to_dict("records")
    a = pd.DataFrame(StreamProcessor(cfg).run(rows))
    b = pd.DataFrame(StreamProcessor(cfg).run(rows))
    pd.testing.assert_frame_equal(a, b)


def test_causal_prefix_property(cfg):
    """Windows computed on a truncated stream are identical to the same windows of the full stream:
    nothing in the future can influence a past window."""
    rows = clip(dur=140).to_dict("records")
    full = pd.DataFrame(StreamProcessor(cfg).run(rows))
    cut = pd.DataFrame(StreamProcessor(cfg).run(rows[: 90 * 30]))
    assert len(cut) > 30
    pd.testing.assert_frame_equal(full.iloc[: len(cut)].reset_index(drop=True), cut)


def test_window_cadence(cfg):
    w = pd.DataFrame(StreamProcessor(cfg).run(clip(dur=60).to_dict("records")))
    assert np.allclose(np.diff(w["t_end"]), cfg.window.stride_s, atol=0.05)
    assert (w["t_end"] - w["t_start"]).between(cfg.window.length_s - 0.2, cfg.window.length_s).all()


def test_ref_ready_only_after_warmup(cfg):
    w = pd.DataFrame(StreamProcessor(cfg).run(clip(dur=120).to_dict("records")))
    assert w.loc[w["t_end"] < 10, "ref_ready"].eq(0).all()
    assert w.loc[w["t_end"] > 60, "ref_ready"].eq(1).all()


def test_drowsy_clip_has_more_closure_and_slower_reopening(cfg):
    a = pd.DataFrame(StreamProcessor(cfg).run(clip("alert", 200).to_dict("records")))
    d = pd.DataFrame(StreamProcessor(cfg).run(clip("drowsy", 200).to_dict("records")))
    a, d = a[a.t_end > 60], d[d.t_end > 60]
    assert d["perclos"].mean() > a["perclos"].mean()
    assert d["reopen_med_s"].mean() > a["reopen_med_s"].mean()


def test_microsleep_detected_only_for_long_continuous_closure(cfg):
    rows = []
    for i in range(30 * 30):
        t = i / 30
        ear = 0.05 if 10 <= t < 12 else 0.30           # 2 s closure
        rows.append({"t": t, "face_found": 1, "ear": ear, "ear_l": ear, "ear_r": ear, "mar": .1,
                     "pitch": 0, "yaw": 0, "head_above": .85, "tilt": 0, "wrist_dist": 2, "wrist_vis": .9})
    sp = StreamProcessor(cfg); sp.run(rows)
    assert len(sp.microsleeps) == 1 and sp.microsleeps[0] == pytest.approx(10, abs=0.2)


def test_no_face_rows_do_not_crash_and_flag_blindness(cfg):
    rows = [{"t": i / 30, "face_found": 0} for i in range(300)]
    w = pd.DataFrame(StreamProcessor(cfg).run(rows))
    assert len(w) > 0 and w["face_found_frac"].eq(0).all() and w["eye_valid_frac"].eq(0).all()


def test_fixed_mode_leaves_z_empty(cfg):
    w = pd.DataFrame(StreamProcessor(cfg, "fixed").run(clip(dur=80).to_dict("records")))
    assert w["z_reopen_med_s"].isna().all() and w["ref_ready"].eq(1).all()


def test_baseline_grid_columns_present(cfg):
    w = pd.DataFrame(StreamProcessor(cfg).run(clip(dur=40).to_dict("records")))
    assert all(f"pc_abs_{t:.2f}" in w for t in cfg.eyes.fixed_grid)
