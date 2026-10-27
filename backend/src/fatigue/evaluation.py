"""Evaluation protocol: leave-one-subject-out, fixed-EAR baseline, earliness, false alarms/hour."""
from __future__ import annotations

import math
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from .alerts import AlertPolicy, BinaryPersistence, Scores, raw_level
from .alerts import LevelThresholds
from .config import Config
from .models import FusionModel, neg_quantile
from .reference import PopulationNormaliser
from .schema import Z_SOURCES

KEY = ["subject", "label", "video"]


# --------------------------------------------------------------------------- splits
def loso_splits(subjects: Sequence[str]) -> Iterator[Tuple[List[str], str]]:
    """The ONE leave-one-subject-out split. No subject is ever in both train and test."""
    subs = sorted(set(subjects))
    for s in subs:
        yield [x for x in subs if x != s], s


def prepare(W: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Keep scored windows of the labelled classes; add y (1 drowsy, 0 alert)."""
    e = cfg.evaluation
    D = W[(W["ref_ready"] > 0) & (W["t_end"] >= e.warmup_s)].copy()
    D = D[D["label"].isin(e.pos_labels + e.neg_labels)]
    D["y"] = D["label"].isin(e.pos_labels).astype(int)
    return D.reset_index(drop=True)


# --------------------------------------------------------------------------- baseline
class FixedEarBaseline:
    """One global EAR threshold -> PERCLOS -> cut. Both tuned on training subjects only."""

    def __init__(self, cfg: Config):
        self.cfg, self.col, self.cut = cfg, None, 0.5

    def fit(self, D: pd.DataFrame) -> "FixedEarBaseline":
        best = -1.0
        y = D["y"].to_numpy()
        for thr in self.cfg.eyes.fixed_grid:
            col = f"pc_abs_{thr:.2f}"
            v = D[col].to_numpy(float)
            ok = np.isfinite(v)
            for cut in np.unique(np.quantile(v[ok], np.linspace(0.05, 0.95, 37))):
                hit = v >= cut
                bal = 0.5 * (np.mean(hit[(y == 1) & ok]) + np.mean(~hit[(y == 0) & ok]))
                if bal > best:
                    best, self.col, self.cut = bal, col, float(cut)
        return self

    def score(self, D: pd.DataFrame) -> np.ndarray:
        return D[self.col].to_numpy(float)


def run_loso_baseline(D: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    out = []
    for train_s, test_s in loso_splits(D["subject"]):
        tr, te = D[D["subject"].isin(train_s)], D[D["subject"] == test_s]
        b = FixedEarBaseline(cfg).fit(tr)
        o = te[KEY + ["t_end", "y"]].copy()
        o["score"] = b.score(te)
        o["hit"] = (o["score"] >= b.cut).fillna(False).astype(int)
        o["ear_thr_col"] = b.col
        out.append(o)
    return pd.concat(out, ignore_index=True)


# --------------------------------------------------------------------------- LOSO for our model
def _group_thresholds(fm: FusionModel, groups_tr: np.ndarray, min_neg: int = 20) -> Dict[str, float]:
    """Per-subgroup decision thresholds that match each group's training FPR to the pooled FPR."""
    oof, y, mode = fm._oof["full"], fm._oof_y, fm._oof_mode
    neg = (y == 0) & (mode == "full") & np.isfinite(oof)
    if neg.sum() == 0:
        return {}
    fpr0 = float(np.mean(oof[neg] >= fm.th.decision))
    out = {}
    for g in np.unique(groups_tr):
        m = neg & (groups_tr == g)
        out[g] = neg_quantile(oof[m], fpr0) if m.sum() >= min_neg else fm.th.decision
    return out


def run_loso(W: pd.DataFrame, cfg: Config, mode: Optional[str] = None,
             cols: Optional[List[str]] = None, group_map: Optional[Dict[str, str]] = None,
             prepared: bool = False) -> pd.DataFrame:
    mode = mode or cfg.reference.mode
    D = W if prepared else prepare(W, cfg)
    pieces = []
    for train_s, test_s in loso_splits(D["subject"]):
        tr = D[D["subject"].isin(train_s)].reset_index(drop=True)
        te = D[D["subject"] == test_s].reset_index(drop=True)
        if mode == "fixed":
            pn = PopulationNormaliser(cfg.reference.norm_method, cfg.reference.scale_floor).fit(tr, Z_SOURCES)
            tr, te = pn.transform(tr), pn.transform(te)
        fm = FusionModel.from_features(cfg, cols).fit(tr, tr["y"].to_numpy(), tr["subject"].to_numpy())
        sc = fm.score(te)
        o = te[KEY + ["t_end", "y"]].copy()
        o[["p", "pe", "pl", "ph", "p_used"]] = sc[["p", "pe", "pl", "ph", "p_used"]].astype(float).to_numpy()
        o["mode"] = sc["mode"].to_numpy()
        o["eye_valid_frac"] = te["eye_valid_frac"].to_numpy()
        th = fm.th
        for k, v in th.full.items():
            o[f"th_{k}"] = v
        for k, v in th.degraded.items():
            o[f"thd_{k}"] = v
        o["decision"], o["decision_deg"] = th.decision, th.degraded_decision
        cut = np.where(o["mode"] == "full", th.decision, th.degraded_decision)
        o["hit"] = ((o["p_used"] >= cut) & (o["mode"] != "blind")).astype(int)
        if group_map is not None:
            gtr = tr["subject"].map(group_map).to_numpy()
            gth = _group_thresholds(fm, gtr)
            g_te = group_map.get(test_s)
            thr_g = gth.get(g_te, th.decision)
            cut_g = np.where(o["mode"] == "full", thr_g, th.degraded_decision)
            o["hit_group"] = ((o["p_used"] >= cut_g) & (o["mode"] != "blind")).astype(int)
        pieces.append(o)
    return pd.concat(pieces, ignore_index=True)


def run_leaky(W: pd.DataFrame, cfg: Config, prepared: bool = False, seed: int = 0) -> pd.DataFrame:
    """The deliberately WRONG protocol: random window-wise split, same person in train and test.
    Exists only to show how much it inflates accuracy."""
    D = W if prepared else prepare(W, cfg)
    pieces = []
    for tr_i, te_i in StratifiedKFold(5, shuffle=True, random_state=seed).split(D, D["y"]):
        tr, te = D.iloc[tr_i].reset_index(drop=True), D.iloc[te_i].reset_index(drop=True)
        fm = FusionModel.from_features(cfg).fit(tr, tr["y"].to_numpy(), tr["subject"].to_numpy())
        sc = fm.score(te)
        o = te[KEY + ["t_end", "y"]].copy()
        o["p_used"] = sc["p_used"].astype(float).to_numpy()
        o["mode"] = sc["mode"].to_numpy()
        cut = np.where(o["mode"] == "full", fm.th.decision, fm.th.degraded_decision)
        o["hit"] = ((o["p_used"] >= cut) & (o["mode"] != "blind")).astype(int)
        pieces.append(o)
    return pd.concat(pieces, ignore_index=True)


# --------------------------------------------------------------------------- levels / alerts
def add_levels(pred: pd.DataFrame, cfg: Config, persistence: bool = True) -> pd.DataFrame:
    """Run the real AlertPolicy over each clip's window sequence."""
    pred = pred.sort_values(KEY + ["t_end"]).reset_index(drop=True)
    raw, lvl = np.zeros(len(pred), int), np.zeros(len(pred), int)
    for _, idx in pred.groupby(KEY, sort=False).groups.items():
        pol = AlertPolicy(cfg.alerts, persistence=persistence)
        for i in idx:
            r = pred.loc[i]
            th = LevelThresholds(
                full={k: r[f"th_{k}"] for k in ("mod", "sev", "early", "late")},
                degraded={k: r[f"thd_{k}"] for k in ("mild", "moderate", "severe")})
            rl = raw_level(Scores(r["p"], r["pe"], r["pl"], r["ph"], r["mode"]), th)
            d = pol.update(r["t_end"], rl, r["mode"])
            raw[i], lvl[i] = rl, d.level
    pred["raw_level"], pred["level"] = raw, lvl
    return pred


def add_baseline_state(pred: pd.DataFrame, cfg: Config, persistence: bool = True) -> pd.DataFrame:
    pred = pred.sort_values(KEY + ["t_end"]).reset_index(drop=True)
    st = np.zeros(len(pred), int)
    for _, idx in pred.groupby(KEY, sort=False).groups.items():
        bp = BinaryPersistence(cfg.evaluation.baseline_persist, cfg.alerts.clear_windows, persistence)
        for i in idx:
            st[i] = bp.update(bool(pred.at[i, "hit"]))
    pred["state"] = st
    return pred


# --------------------------------------------------------------------------- metrics
def binary_metrics(y: np.ndarray, hit: np.ndarray) -> Dict[str, float]:
    y, hit = np.asarray(y).astype(int), np.asarray(hit).astype(int)
    tpr = float(hit[y == 1].mean()) if (y == 1).any() else math.nan
    fpr = float(hit[y == 0].mean()) if (y == 0).any() else math.nan
    return {"acc": float((hit == y).mean()), "tpr": tpr, "fpr": fpr,
            "bal_acc": float(np.nanmean([tpr, 1 - fpr])), "alert_rate": float(hit.mean()), "n": int(len(y))}


def per_subject(pred: pd.DataFrame, hit_col: str = "hit") -> pd.DataFrame:
    rows = [{"subject": s, **binary_metrics(g["y"], g[hit_col])} for s, g in pred.groupby("subject")]
    return pd.DataFrame(rows)


def summarize(pred: pd.DataFrame, hit_col: str = "hit") -> Dict[str, float]:
    ps = per_subject(pred, hit_col)
    pooled = binary_metrics(pred["y"], pred[hit_col])
    return {**{f"pooled_{k}": v for k, v in pooled.items()},
            "subj_acc_mean": float(ps["acc"].mean()), "subj_acc_std": float(ps["acc"].std(ddof=1)),
            "subj_bal_acc_mean": float(ps["bal_acc"].mean()), "subj_bal_acc_std": float(ps["bal_acc"].std(ddof=1)),
            "n_subjects": int(len(ps))}


def _onsets(sig: np.ndarray) -> int:
    s = np.asarray(sig).astype(int)
    return int(((s[1:] == 1) & (s[:-1] == 0)).sum() + (1 if len(s) and s[0] == 1 else 0))


def false_alarms_per_hour(pred: pd.DataFrame, cfg: Config, col: str) -> float:
    """Alert onsets per hour of scored time on ALERT (negative) clips."""
    neg = pred[pred["y"] == 0]
    n_on, hours = 0, 0.0
    for _, g in neg.sort_values("t_end").groupby(KEY, sort=False):
        n_on += _onsets((g[col] >= 1).to_numpy())
        hours += len(g) * cfg.window.stride_s / 3600.0
    return n_on / hours if hours > 0 else math.nan


def earliness(pred: pd.DataFrame, events: pd.DataFrame, cfg: Config, col: str, level: int = 1
              ) -> Dict[str, float]:
    """Lead time (s) between the first alert and the first microsleep, on clips that have one."""
    e = cfg.evaluation
    leads, missed, n = [], 0, 0
    for k, g in pred[pred["label"].isin(e.earliness_labels)].sort_values("t_end").groupby(KEY, sort=False):
        ev = events[(events["subject"] == k[0]) & (events["label"] == k[1]) & (events["video"] == k[2])
                    & (events["kind"] == "microsleep") & (events["t"] >= e.warmup_s)]
        if ev.empty:
            continue
        n += 1
        t_ms = float(ev["t"].min())
        a = g[(g[col] >= level) & (g["t_end"] <= t_ms)]
        if a.empty:
            missed += 1
            leads.append(0.0)
        else:
            leads.append(t_ms - float(a["t_end"].iloc[0]))
    arr = np.array(leads)
    return {"n_clips": n, "frac_alerted_before": (1 - missed / n) if n else math.nan,
            "median_lead_s": float(np.median(arr)) if n else math.nan,
            "mean_lead_s": float(arr.mean()) if n else math.nan}


def bootstrap_subject_ci(pred: pd.DataFrame, fn, n_boot: int, seed: int) -> Tuple[float, float]:
    """95% CI of fn(pred_subset) by resampling SUBJECTS (the unit that generalises)."""
    rng = np.random.default_rng(seed)
    subs = pred["subject"].unique()
    by = {s: g for s, g in pred.groupby("subject")}
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(subs, len(subs), replace=True)
        vals.append(fn(pd.concat([by[s] for s in pick])))
    vals = [v for v in vals if np.isfinite(v)]
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else (math.nan, math.nan)
