import math


from fatigue.behaviour import EYE_RUB, HEAD_SHAKE, POSTURE_SHIFT, BehaviourDetector
from fatigue.blink import Blink
from fatigue.config import load_config

C = load_config()
FPS = 30


def feed(det, n, **kw):
    ev = []
    for i in range(n):
        t = i / FPS
        args = {"wrist_dist": 2.0, "wrist_vis": 0.9, "yaw": 0.0, "head_above": 0.85, "tilt": 0.0}
        for k, v in kw.items():
            args[k] = v(t) if callable(v) else v
        ev += det.push(t, **args)
    return ev


def new():
    return BehaviourDetector(C.behaviour, C.blink)


def test_quiet_driver_no_events():
    assert feed(new(), 300) == []


def test_eye_rub_needs_sustained_close_hand():
    assert [k for _, k in feed(new(), 90, wrist_dist=lambda t: 0.3 if 1.0 < t < 2.0 else 2.0)] == [EYE_RUB]
    assert feed(new(), 90, wrist_dist=lambda t: 0.3 if 1.0 < t < 1.1 else 2.0) == []     # brushed past
    assert feed(new(), 90, wrist_dist=0.3, wrist_vis=0.1) == []                          # unreliable wrist


def test_head_shake_detected_but_slow_turn_is_not():
    shake = lambda t: 14 * math.sin(2 * math.pi * 3 * t) if 1 < t < 1.8 else 0.0
    assert HEAD_SHAKE in [k for _, k in feed(new(), 90, yaw=shake)]
    slow_turn = lambda t: min(20.0, 6 * t)
    assert HEAD_SHAKE not in [k for _, k in feed(new(), 90, yaw=slow_turn)]


def test_posture_shift_and_refractory():
    step = lambda t: 0.85 + (0.25 if t > 1 else 0.0)
    ev = feed(new(), 150, head_above=step)
    assert [k for _, k in ev].count(POSTURE_SHIFT) == 1


def test_hard_blink_requires_squint_and_depth():
    d = new()
    assert d.from_blink(Blink(1, 1.4, .05, .2, .1, 0.95, squint=0.8)) is not None
    d = new()
    assert d.from_blink(Blink(1, 1.4, .05, .2, .1, 0.95, squint=0.1)) is None
    d = new()
    assert d.from_blink(Blink(1, 1.4, .05, .2, .1, 0.6, squint=0.9)) is None
