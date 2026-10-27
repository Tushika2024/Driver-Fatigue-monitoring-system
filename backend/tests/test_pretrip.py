from datetime import datetime


from fatigue.pretrip import (CAUTION, ENROLLING, FIT, INCONCLUSIVE, NOT_FIT, HistoryStore, PretripSession, evaluate, quality_check)
from fatigue.synthetic import generate_clip, make_subjects

SP = make_subjects(1, 3)[0]
NOW = datetime(2026, 3, 2, 8, 0)


def clean_clip(kind, seed, dur):
    """A clip whose first seconds have a visible face (the quality gate would rightly refuse otherwise)."""
    for k in range(50):
        df = generate_clip(SP, kind, dur, 30.0, seed * 100 + k)
        if df[df.t < 4]["face_found"].min() == 1:
            return df
    raise AssertionError("no clean clip found")


def session(cfg, store, kind="alert", seed=1, now=NOW, dur=100, mutate=None):
    s = PretripSession(cfg, "d1", store, now)
    df = clean_clip(kind, seed, dur)
    if mutate:
        df = mutate(df)
    for r in df.to_dict("records"):
        s.push(r)
        if s.done:
            break
    return s.result()


def test_first_sessions_only_enrol(cfg, tmp_path):
    store = HistoryStore(str(tmp_path))
    for i in range(cfg.pretrip.min_sessions):
        r = session(cfg, store, seed=i)
        assert r.verdict == ENROLLING and r.index is None and r.stored
    assert len(store.load("d1")) == cfg.pretrip.min_sessions


def test_fit_then_not_fit_when_tired(cfg, tmp_path):
    store = HistoryStore(str(tmp_path))
    for i in range(5):
        session(cfg, store, seed=i)
    assert session(cfg, store, seed=40).verdict in (FIT, CAUTION)
    tired = session(cfg, store, "drowsy", seed=50)
    assert tired.verdict in (CAUTION, NOT_FIT) and tired.reason and tired.index > 1.0


def test_never_blocking_and_always_has_reason(cfg, tmp_path):
    store = HistoryStore(str(tmp_path))
    for i in range(5):
        session(cfg, store, seed=i)
    r = session(cfg, store, "drowsy", seed=51)
    assert r.blocking is False and r.advisory and r.reason


def test_dark_room_is_inconclusive_and_not_stored(cfg, tmp_path):
    store = HistoryStore(str(tmp_path))
    r = session(cfg, store, mutate=lambda d: d.assign(brightness=20.0))
    assert r.verdict == INCONCLUSIVE and "dark" in r.reason.lower()
    assert store.load("d1") == []


def test_face_too_far_and_not_found(cfg, tmp_path):
    store = HistoryStore(str(tmp_path))
    assert "closer" in session(cfg, store, mutate=lambda d: d.assign(face_w_frac=0.05)).reason
    r = session(cfg, store, mutate=lambda d: d.assign(face_found=0.0))
    assert r.verdict == INCONCLUSIVE and not store.load("d1")


def test_too_few_blinks_is_inconclusive_not_not_fit(cfg, tmp_path):
    store = HistoryStore(str(tmp_path))
    r = session(cfg, store, mutate=lambda d: d.assign(ear=0.30, ear_l=0.30, ear_r=0.30))   # eyes never blink
    assert r.verdict == INCONCLUSIVE and "blink" in r.reason.lower() and not store.load("d1")


def test_capture_extends_to_90s_when_blinks_are_few(cfg, tmp_path):
    store = HistoryStore(str(tmp_path))
    r = session(cfg, store, mutate=lambda d: d.assign(ear=0.30, ear_l=0.30, ear_r=0.30), dur=120)
    assert r.duration_s >= cfg.pretrip.extended_s - 1


def test_outlier_warning_asks_to_confirm_driver(cfg):
    hist = [{"slot": 2, "reopen_s": 0.08, "openness": 0.30, "rate_pm": 15.0, "head_micro": 0.5}] * 3 + \
           [{"slot": 2, "reopen_s": 0.081, "openness": 0.301, "rate_pm": 15.5, "head_micro": 0.52}] * 3
    odd = {"reopen_s": 0.25, "openness": 0.12, "rate_pm": 40.0, "head_micro": 4.0}
    *_, warns, _ = evaluate(odd, hist, NOW, cfg)
    assert warns and "confirm" in warns[0].lower()


def test_slot_matching_used_only_with_enough_sessions(cfg):
    mk = lambda slot, n: [{"slot": slot, "reopen_s": .08, "openness": .3, "rate_pm": 15., "head_micro": .5}] * n
    m = {"reopen_s": .08, "openness": .3, "rate_pm": 15., "head_micro": .5}
    assert evaluate(m, mk(2, 3) + mk(5, 3), NOW, cfg)[-1].startswith("time-of-day")
    assert evaluate(m, mk(2, 2) + mk(5, 4), NOW, cfg)[-1] == "all sessions"


def test_history_capped_and_atomic(tmp_path):
    s = HistoryStore(str(tmp_path))
    for i in range(10):
        s.append("a/b", {"i": i}, keep=4)
    assert [x["i"] for x in s.load("a/b")] == [6, 7, 8, 9]
    assert not [f for f in tmp_path.iterdir() if f.suffix == ".tmp"]


def test_quality_gate_messages(cfg):
    ok = [{"t": i / 30, "face_found": 1, "brightness": 130, "face_w_frac": .3, "pitch": 0, "yaw": 0} for i in range(90)]
    assert quality_check(ok, cfg) is None
    assert "straight" in quality_check([{**r, "yaw": 40} for r in ok], cfg)
    assert "bright" in quality_check([{**r, "brightness": 250} for r in ok], cfg).lower()
