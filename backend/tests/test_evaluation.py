import numpy as np
import pandas as pd
import pytest

from fatigue import evaluation as ev
from fatigue.alerts import BinaryPersistence


def test_loso_no_subject_in_both_sets():
    subs = ["a", "b", "c", "d"] * 3
    seen = []
    for tr, te in ev.loso_splits(subs):
        assert te not in tr and sorted(tr + [te]) == ["a", "b", "c", "d"]
        seen.append(te)
    assert seen == ["a", "b", "c", "d"]


def test_prepare_drops_warmup_unready_and_unlabelled(cfg):
    W = pd.DataFrame({"t_end": [10, 70, 70, 70], "ref_ready": [1, 1, 0, 1], "label": ["0", "0", "0", "5"]})
    D = ev.prepare(W, cfg)
    assert len(D) == 1 and D["y"].iloc[0] == 0


def test_binary_metrics():
    m = ev.binary_metrics([1, 1, 0, 0], [1, 0, 0, 1])
    assert m["acc"] == .5 and m["tpr"] == .5 and m["fpr"] == .5 and m["alert_rate"] == .5


def test_loso_run_keeps_test_subject_out(cfg, windows):
    W, _ = windows
    D = ev.prepare(W, cfg)
    pred = ev.run_loso(D, cfg, "adaptive", prepared=True)
    assert set(pred["subject"]) == set(D["subject"]) and len(pred) == len(D)
    # each subject has exactly one fold-specific decision threshold (fit without that subject)
    assert pred.groupby("subject")["decision"].nunique().eq(1).all()


def test_ours_beats_fixed_baseline_on_synthetic(cfg, windows):
    W, _ = windows
    D = ev.prepare(W, cfg)
    ours = ev.summarize(ev.run_loso(D, cfg, "adaptive", prepared=True))
    base = ev.summarize(ev.run_loso_baseline(D, cfg))
    assert ours["pooled_bal_acc"] > base["pooled_bal_acc"] + 0.05


def test_leaky_split_mixes_people(cfg, windows):
    W, _ = windows
    D = ev.prepare(W, cfg)
    leaky = ev.run_leaky(D, cfg, prepared=True)
    assert len(leaky) == len(D)          # every window evaluated once, by a model that saw its neighbours


def test_levels_use_persistence(cfg, windows):
    W, _ = windows
    pred = ev.run_loso(ev.prepare(W, cfg), cfg, "adaptive", prepared=True)
    with_p, without = ev.add_levels(pred, cfg, True), ev.add_levels(pred, cfg, False)
    assert ev.false_alarms_per_hour(with_p, cfg, "level") <= ev.false_alarms_per_hour(without, cfg, "level")
    assert with_p["level"].between(0, 3).all()


def test_earliness_counts_only_alerts_before_microsleep(cfg):
    pred = pd.DataFrame({"subject": "s", "label": "10", "video": "v", "t_end": [70., 80, 90, 100], "level": [0, 0, 1, 1]})
    ev_ = pd.DataFrame({"subject": ["s"], "label": ["10"], "video": ["v"], "t": [95.0], "kind": ["microsleep"]})
    r = ev.earliness(pred, ev_, cfg, "level", 1)
    assert r["n_clips"] == 1 and r["median_lead_s"] == pytest.approx(5.0) and r["frac_alerted_before"] == 1
    late = pred.assign(level=[0, 0, 0, 1])
    assert ev.earliness(late, ev_, cfg, "level", 1)["frac_alerted_before"] == 0


def test_baseline_persistence_filter():
    bp = BinaryPersistence(2, 3)
    assert [bp.update(h) for h in [1, 0, 1, 1, 1, 0, 0, 0, 0]] == [0, 0, 0, 1, 1, 1, 1, 0, 0]


def test_bootstrap_ci_brackets_estimate():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"subject": np.repeat(list("abcdefgh"), 20), "y": rng.integers(0, 2, 160)})
    df["hit"] = df["y"]
    lo, hi = ev.bootstrap_subject_ci(df, lambda d: ev.binary_metrics(d["y"], d["hit"])["acc"], 100, 0)
    assert lo <= 1.0 <= hi + 1e-9
