
import numpy as np
import pytest

from fatigue.blink import BlinkDetector, segment_blinks
from fatigue.config import load_config

FPS = 30
CFG = load_config().blink


def shape(t, s, c, h, r, depth=0.9):
    x = np.zeros_like(t)
    a = (t >= s) & (t < s + c); x[a] = depth * (t[a] - s) / c
    b = (t >= s + c) & (t < s + c + h); x[b] = depth
    d = (t >= s + c + h) & (t < s + c + h + r); x[d] = depth * (1 - (t[d] - s - c - h) / r)
    return x


def run(env, fps=FPS):
    t = np.arange(len(env)) / fps
    return segment_blinks(t, 1 - env, CFG)


def test_timing_recovered_and_slowdown_detected():
    t = np.arange(0, 10, 1 / FPS)
    env = shape(t, 2, 0.07, 0.05, 0.14) + shape(t, 6, 0.07, 0.05, 0.30)
    b = run(env)
    assert len(b) == 2
    assert b[0].reopen_s == pytest.approx(0.14 * 0.6, abs=0.03)
    assert b[1].reopen_s > 1.8 * b[0].reopen_s       # the slowdown is visible
    assert b[1].asym > b[0].asym


def test_shallow_dip_is_not_a_blink():
    t = np.arange(0, 5, 1 / FPS)
    assert run(shape(t, 2, 0.1, 0.05, 0.1, depth=0.2)) == []


def test_long_closure_is_not_a_blink():
    t = np.arange(0, 8, 1 / FPS)
    assert run(shape(t, 2, 0.1, 3.0, 0.3)) == []


def test_short_gap_tolerated_long_gap_aborts():
    t = np.arange(0, 6, 1 / FPS)
    env = shape(t, 2, 0.07, 0.10, 0.14)
    x = 1 - env
    x2 = x.copy(); x2[int(2.04 * FPS)] = np.nan            # one dropped frame
    assert len(segment_blinks(t, x2, CFG)) == 1
    x3 = x.copy(); x3[int(2.0 * FPS):int(2.6 * FPS)] = np.nan   # 0.6 s lost mid-blink
    assert segment_blinks(t, x3, CFG) == []


def test_episode_starting_closed_is_ignored():
    t = np.arange(0, 3, 1 / FPS)
    x = np.where(t < 1.0, 0.1, 1.0)
    assert segment_blinks(t, x, CFG) == []


def test_streaming_matches_batch():
    t = np.arange(0, 12, 1 / FPS)
    env = shape(t, 2, 0.07, 0.05, 0.14) + shape(t, 5, 0.08, 0.08, 0.2) + shape(t, 9, 0.07, 0.05, 0.1)
    batch = segment_blinks(t, 1 - env, CFG)
    det, stream = BlinkDetector(CFG), []
    for ti, xi in zip(t, 1 - env):
        b = det.push(float(ti), float(xi))
        if b: stream.append(b)
    assert [round(b.t_end, 3) for b in batch] == [round(b.t_end, 3) for b in stream]


def test_hard_blink_squint_recorded():
    t = np.arange(0, 5, 1 / FPS)
    env = shape(t, 2, 0.08, 0.2, 0.2, depth=0.97)
    sq = np.where(env > 0.5, 0.8, 0.05)
    b = segment_blinks(t, 1 - env, CFG, sq)
    assert len(b) == 1 and b[0].squint == pytest.approx(0.8)
