import math


from fatigue.alerts import (CRITICAL, MILD, MODERATE, NONE, SEVERE, AlertPolicy, LevelThresholds, Scores, raw_level)
from fatigue.config import load_config

A = load_config().alerts
TH = LevelThresholds(full={"mod": .6, "sev": .9, "early": .4, "late": .5},
                     degraded={"mild": .5, "moderate": .7, "severe": .95})


def run(pol, seq, dt=1.5):
    return [pol.update(i * dt, r).level for i, r in enumerate(seq)]


def test_two_streams_map_to_levels():
    S = lambda **k: Scores(**{"p": .1, "pe": .1, "pl": .1, "ph": .1, "mode": "full", **k})
    assert raw_level(S(), TH) == NONE
    assert raw_level(S(pe=.5), TH) == MILD                      # early only
    assert raw_level(S(pe=.5, pl=.6, p=.3), TH) == MODERATE     # early + late
    assert raw_level(S(pl=.6, p=.95), TH) == SEVERE             # late + very high fused
    assert raw_level(S(pl=.6, p=.7), TH) == MODERATE
    assert raw_level(S(p=.99, pe=.1, pl=.1), TH) == MODERATE    # fused alone never jumps to SEVERE


def test_degraded_uses_head_model_with_wider_thresholds():
    S = lambda ph: Scores(p=math.nan, pe=math.nan, pl=math.nan, ph=ph, mode="degraded")
    assert [raw_level(S(x), TH) for x in (.3, .55, .75, .97)] == [NONE, MILD, MODERATE, SEVERE]


def test_blind_never_alerts():
    assert raw_level(Scores(.9, .9, .9, .9, "blind"), TH) == NONE


def test_persistence_blocks_single_spike():
    assert run(AlertPolicy(A), [0, 2, 0, 0, 0]) == [0, 0, 0, 0, 0]
    assert run(AlertPolicy(A), [0, 2, 2, 0]) == [0, 0, 2, 2]


def test_severe_needs_three_windows():
    assert run(AlertPolicy(A), [3, 3, 3]) == [0, 2, 3]


def test_hysteresis_holds_then_clears():
    lv = run(AlertPolicy(A), [2, 2, 0, 0, 0, 0])
    assert lv == [0, 2, 2, 2, 0, 0]        # clears only after clear_windows=3 quiet windows


def test_blind_windows_are_neutral():
    p = AlertPolicy(A)
    run(p, [2, 2])
    assert p.update(10, 0, "blind").level == 2 and p.level == 2


def test_acknowledge_resets_and_requires_fresh_evidence():
    p = AlertPolicy(A)
    run(p, [3, 3, 3, 3])
    assert p.level == SEVERE
    p.acknowledge(10)
    assert p.level == NONE
    assert p.update(11, 3).level == NONE


def test_repeat_until_acknowledged():
    p = AlertPolicy(A)
    run(p, [3, 3, 3])
    fires = [p.should_repeat(t) for t in (5, 6, 12, 13, 21)]
    assert fires == [True, False, False, True, True]     # ack_repeat_s = 8
    p.acknowledge(22)
    assert not p.should_repeat(40)


def test_critical_only_when_enabled_and_unacknowledged():
    on = AlertPolicy(A, critical_enabled=True)
    off = AlertPolicy(A, critical_enabled=False)
    for pol in (on, off):
        for i in range(40):
            pol.update(i * 1.5, 3)
    assert on.level == CRITICAL and off.level == SEVERE
