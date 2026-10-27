import math

import numpy as np
import pytest

from fatigue.config import load_config
from fatigue.reference import (DriverNormaliser, PopulationNormaliser, Reference, robust_loc_scale, zscore)

CFG = load_config()


def feed(ref, ear, seconds, fps=30, pitch=0.0):
    for i in range(int(seconds * fps)):
        ref.update(i / fps, ear(i) if callable(ear) else ear, pitch, 0.85)


def test_adaptive_open_level_tracks_driver_not_population():
    r = Reference(CFG, "adaptive")
    feed(r, 0.22, 60)
    assert r.ready and r.open_level == pytest.approx(0.22, abs=0.01)
    r2 = Reference(CFG, "adaptive"); feed(r2, 0.38, 60)
    assert r2.open_level == pytest.approx(0.38, abs=0.01)


def test_open_level_robust_to_closed_eyes():
    # 25% of frames closed: the high percentile must still report the open level
    r = Reference(CFG, "adaptive")
    feed(r, lambda i: 0.05 if (i // 15) % 4 == 0 else 0.30, 90)
    assert r.open_level == pytest.approx(0.30, abs=0.015)


def test_not_ready_falls_back_to_population():
    r = Reference(CFG, "adaptive"); feed(r, 0.22, 5)
    assert not r.ready and r.open_level == CFG.eyes.fixed_open_ear


def test_fixed_mode_never_adapts():
    r = Reference(CFG, "fixed"); feed(r, 0.22, 60)
    assert r.ready and r.open_level == CFG.eyes.fixed_open_ear


def test_enrolled_freezes_after_enrolment():
    r = Reference(CFG, "enrolled")
    feed(r, 0.30, 70)
    frozen = r.open_level
    feed(r, 0.15, 60)             # later change must not move the frozen reference
    assert r.open_level == frozen


def test_profile_round_trip():
    r = Reference(CFG, "adaptive"); feed(r, 0.27, 60)
    r2 = Reference.from_profile(CFG, r.to_profile())
    assert r2.open_level == pytest.approx(r.open_level)


def test_median_iqr_resists_outliers_where_mean_std_does_not():
    x = np.r_[np.random.default_rng(0).normal(1.0, 0.1, 200), [50.0] * 5]
    lm, sm = robust_loc_scale(x, "median_iqr", 0.0)
    lx, sx = robust_loc_scale(x, "mean_std", 0.0)
    assert abs(lm - 1.0) < 0.05 and sm < 0.2
    assert sx > 5                         # inflated by the outliers


def test_zscore_nan_and_clip():
    assert math.isnan(zscore(float("nan"), 0, 1))
    assert zscore(1e6, 0, 1) == 8.0


def test_driver_normaliser_ready_after_min_windows():
    n = DriverNormaliser(CFG.reference, ["a"])
    assert math.isnan(n.transform({"a": 1.0})["z_a"])
    for v in np.linspace(0.9, 1.1, 20):
        n.update({"a": float(v)})
    assert n.ready and abs(n.transform({"a": 1.0})["z_a"]) < 1.0
    assert n.transform({"a": 2.0})["z_a"] > 3


def test_population_normaliser_uses_fit_data_only():
    import pandas as pd
    tr = pd.DataFrame({"a": np.linspace(0, 2, 100)})
    te = pd.DataFrame({"a": [1.0, 100.0]})
    out = PopulationNormaliser().fit(tr, ["a"]).transform(te)
    assert abs(out["z_a"].iloc[0]) < 0.1 and out["z_a"].iloc[1] == 8.0
