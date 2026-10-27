import numpy as np
import pandas as pd

from fatigue.report import (depot_table, driver_metrics, render_depot_html, render_driver_html, synthetic_depot_logs)


def log(levels, mode="full", step=3.0, driver="D1", route="A", start="2026-03-01T22:00:00"):
    n = len(levels)
    t = np.arange(n) * step
    return pd.DataFrame({"t": t, "ts": (pd.Timestamp(start) + pd.to_timedelta(t, unit="s")).astype(str),
                         "driver_id": driver, "vehicle_id": "V", "route": route,
                         "score": np.clip(np.array(levels) / 4, 0, 1), "early_score": 0.0, "late_score": 0.0,
                         "level": levels, "mode": mode, "responded": np.nan, "reason": ""})


def test_episodes_counted_at_peak_and_lead_time(cfg):
    lv = [0] * 5 + [1, 1, 2, 3, 3, 0] + [0] * 3 + [1, 0]
    m = driver_metrics(log(lv), cfg)
    assert m["alert_episodes"] == {"MILD": 1, "MODERATE": 0, "SEVERE": 1, "CRITICAL": 0}
    assert m["median_lead_to_severe_s"] == 9.0


def test_low_coverage_flag_under_60_percent_full_mode(cfg):
    df = log([0] * 10)
    df.loc[:5, "mode"] = "degraded"          # 4/10 full = 40%
    m = driver_metrics(df, cfg)
    assert m["low_coverage"] and abs(m["full_mode_frac"] - 0.4) < 1e-9
    html = render_driver_html(df, {"synthetic": False}, cfg)
    assert "Low coverage" in html
    assert "Low coverage" not in render_driver_html(log([0] * 10), {}, cfg)


def test_synthetic_stamp_on_every_page(cfg):
    assert "SYNTHETIC" in render_driver_html(log([0, 1, 2, 3] * 5), {"synthetic": True}, cfg)
    t = depot_table(synthetic_depot_logs(6, 2), cfg)
    assert "not</b> a finding" in render_depot_html(t, {"synthetic": True}, cfg)


def test_depot_hides_small_cells_and_never_leaks_driver_ids(cfg):
    logs = [log([2] * 20 + [0] * 20, driver=f"D{i}", route="R1") for i in range(2)] + \
           [log([2] * 20 + [0] * 20, driver=f"X{i}", route="R2") for i in range(4)]
    t = depot_table(logs, cfg)
    assert t[t.route == "R1"]["suppressed"].all() and t[t.route == "R1"]["events_per_100h"].isna().all()
    assert t[t.route == "R2"]["events_per_100h"].notna().all()
    assert "driver" not in t.columns
    import re
    html = re.sub(r"base64,[A-Za-z0-9+/=]+", "", render_depot_html(t, {}, cfg))     # ignore the embedded PNG
    assert "D0" not in html and "X0" not in html


def test_depot_slot_pattern_visible_in_illustrative_data(cfg):
    t = depot_table(synthetic_depot_logs(12, 6, seed=3), cfg).dropna(subset=["events_per_100h"])
    night = t[t.slot_label == "00-04h"]["pct_time_alert"].mean()
    day = t[t.slot_label == "08-12h"]["pct_time_alert"].mean() if (t.slot_label == "08-12h").any() else 0
    assert night > day
