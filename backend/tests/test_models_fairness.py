import numpy as np
import pandas as pd
import pytest

from fatigue import evaluation as ev
from fatigue import fairness as fr
from fatigue.models import FusionModel, youden_threshold


@pytest.fixture(scope="module")
def fitted(cfg, windows, model):
    W, _ = windows
    return model, ev.prepare(W, cfg)


def test_threshold_ordering(fitted):
    m, _ = fitted
    f = m.th.full
    assert f["sev"] >= f["mod"]
    assert all(np.isfinite(v) for v in list(f.values()) + list(m.th.degraded.values()))
    assert m.th.degraded["severe"] >= m.th.degraded["moderate"] >= m.th.degraded["mild"]


def test_mode_routing(fitted, cfg):
    m, D = fitted
    row = D.iloc[[0]].copy()
    row["eye_valid_frac"], row["face_found_frac"] = 1.0, 1.0
    assert m.score(row)["mode"].iloc[0] == "full"
    row["eye_valid_frac"] = 0.2
    s = m.score(row).iloc[0]
    assert s["mode"] == "degraded" and np.isnan(s["p"]) and np.isfinite(s["ph"])
    row["face_found_frac"] = 0.1
    s = m.score(row).iloc[0]
    assert s["mode"] == "blind" and np.isnan(s["ph"])


def test_degraded_model_does_not_use_eye_features(fitted):
    m, _ = fitted
    assert not set(m.head_cols) & {"perclos", "longest_closure_s", "reopen_med_s", "ear_diverge", "n_hard_blink"}


def test_reason_strings_are_human_readable(fitted):
    m, D = fitted
    drowsy = D[D["y"] == 1].iloc[0]
    r = m.reason(drowsy, "full")
    assert isinstance(r, str) and len(r) > 5 and "perclos" not in r


def test_save_load_round_trip(fitted, tmp_path):
    m, D = fitted
    p = str(tmp_path / "m.joblib"); m.save(p)
    m2 = FusionModel.load(p)
    a, b = m.score(D.head(20))["p"].astype(float), m2.score(D.head(20))["p"].astype(float)
    np.testing.assert_allclose(a, b, equal_nan=True)


def test_youden_picks_separating_threshold():
    y = np.r_[np.zeros(100), np.ones(100)]
    p = np.r_[np.linspace(0, .4, 100), np.linspace(.6, 1, 100)]
    assert 0.4 <= youden_threshold(y, p) <= 0.62


def test_hgb_variant_trains(cfg, windows):
    W, _ = windows
    c = cfg.replace(model={"kind": "hgb"})
    D = ev.prepare(W, c)
    m = FusionModel.from_features(c).fit(D, D["y"].to_numpy(), D["subject"].to_numpy())
    assert m.score(D.head(5))["p"].notna().all() and isinstance(m.reason(D.iloc[0], "full"), str)


# ---------------------------------------------------------------- fairness
def toy():
    rows = []
    for s, g, fpr_hits in (("a", 0, 0), ("b", 0, 0), ("c", 1, 2), ("d", 1, 2)):
        for i in range(10):
            y = i % 2
            hit = 1 if y == 1 else (1 if (i // 2) < fpr_hits else 0)
            rows.append({"subject": s, "y": y, "hit": hit, "eye_valid_frac": 0.9 - 0.3 * g})
    return pd.DataFrame(rows), pd.DataFrame({"subject": list("abcd"), "g": [0, 0, 1, 1]})


def test_gap_values_known():
    pred, meta = toy()
    g = fr.gaps(pred, meta, "g")
    assert g["tpr_gap"] == pytest.approx(0.0)
    assert g["fpr_gap"] == pytest.approx(0.4)            # 2 of 5 alert windows flagged in group 1, 0 in group 0
    assert g["eq_odds_gap"] == pytest.approx(0.4)
    assert g["dp_diff"] == pytest.approx(0.2)


def test_report_has_gap_row_and_ci():
    pred, meta = toy()
    r = fr.fairness_report({"m": (pred, "hit")}, meta, ["g"], n_boot=50)
    gap = r[r["row"] == "GAP (1 minus 0)"].iloc[0]
    assert gap["eq_odds_gap"] == pytest.approx(0.4) and gap["eq_odds_ci_lo"] <= gap["eq_odds_gap"] <= gap["eq_odds_ci_hi"]


def test_group_threshold_correction_runs_and_is_leakage_free(cfg, windows, dataset):
    W, _ = windows
    _, meta = dataset
    gm = dict(zip(meta["subject"], meta["glasses"]))
    pred = ev.run_loso(ev.prepare(W, cfg), cfg, "adaptive", group_map=gm, prepared=True)
    assert {"hit", "hit_group"} <= set(pred.columns) and pred["hit_group"].isin([0, 1]).all()
