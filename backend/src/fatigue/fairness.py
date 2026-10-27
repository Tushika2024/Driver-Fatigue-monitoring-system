"""Fairness audit by objectively observable attributes (glasses, facial hair).

No inference of gender or ethnicity from video. With ~12 people the cells are tiny, so
this is a METHODOLOGY demonstration with subject-bootstrap intervals, not a certification.
"""
from __future__ import annotations

import math
from typing import Dict, List

import numpy as np
import pandas as pd

from .evaluation import binary_metrics


def group_table(pred: pd.DataFrame, meta: pd.DataFrame, attr: str, hit_col: str = "hit") -> pd.DataFrame:
    m = pred.merge(meta[["subject", attr]], on="subject", how="left")
    rows = []
    for g, d in m.groupby(attr):
        bm = binary_metrics(d["y"], d[hit_col])
        rows.append({"attr": attr, "group": int(g), "n_subjects": d["subject"].nunique(),
                     "n_windows": bm["n"], "tpr": bm["tpr"], "fpr": bm["fpr"],
                     "alert_rate": bm["alert_rate"],
                     "eye_valid_frac": float(d["eye_valid_frac"].mean()) if "eye_valid_frac" in d else math.nan})
    return pd.DataFrame(rows)


def gaps(pred: pd.DataFrame, meta: pd.DataFrame, attr: str, hit_col: str = "hit") -> Dict[str, float]:
    """Group 1 minus group 0 (e.g. glasses minus no glasses)."""
    t = group_table(pred, meta, attr, hit_col).set_index("group")
    if not {0, 1} <= set(t.index):
        return {k: math.nan for k in ("tpr_gap", "fpr_gap", "eq_odds_gap", "dp_diff")}
    d_tpr, d_fpr = t.loc[1, "tpr"] - t.loc[0, "tpr"], t.loc[1, "fpr"] - t.loc[0, "fpr"]
    return {"tpr_gap": float(d_tpr), "fpr_gap": float(d_fpr),
            "eq_odds_gap": float(max(abs(d_tpr), abs(d_fpr))),
            "dp_diff": float(t.loc[1, "alert_rate"] - t.loc[0, "alert_rate"])}


def gaps_with_ci(pred, meta, attr, hit_col="hit", n_boot=300, seed=0) -> Dict[str, float]:
    base = gaps(pred, meta, attr, hit_col)
    m = pred.merge(meta[["subject", attr]], on="subject", how="left")
    by = {s: g for s, g in m.groupby("subject")}
    grp = m.groupby("subject")[attr].first()
    pools = {g: list(grp[grp == g].index) for g in (0, 1)}
    if min(len(v) for v in pools.values()) < 2:
        return {**base, **{f"{k}_ci": (math.nan, math.nan) for k in base}}
    rng = np.random.default_rng(seed)
    acc: Dict[str, List[float]] = {k: [] for k in base}
    for _ in range(n_boot):
        pick = [s for g in (0, 1) for s in rng.choice(pools[g], len(pools[g]), replace=True)]
        # resampled subjects repeat ids; rebuild with unique ids so the merge stays 1:1
        parts, rows = [], []
        for i, s in enumerate(pick):
            d = by[s].drop(columns=[attr]).copy(); d["subject"] = f"{s}#{i}"; parts.append(d)
            rows.append({"subject": f"{s}#{i}", attr: int(grp[s])})
        res = gaps(pd.concat(parts), pd.DataFrame(rows), attr, hit_col)
        for k, v in res.items():
            if np.isfinite(v):
                acc[k].append(v)
    out = dict(base)
    for k, v in acc.items():
        out[f"{k}_ci"] = (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) if v else (math.nan, math.nan)
    return out


def fairness_report(preds: Dict[str, "tuple[pd.DataFrame, str]"], meta: pd.DataFrame,
                    attrs: List[str], n_boot: int = 300, seed: int = 0) -> pd.DataFrame:
    """preds: {method_name: (pred_df, hit_col)} -> one row per method x attribute x group, plus gap rows."""
    rows = []
    for name, (pred, col) in preds.items():
        for a in attrs:
            if a not in meta.columns:
                continue
            t = group_table(pred, meta, a, col)
            g = gaps_with_ci(pred, meta, a, col, n_boot, seed)
            for _, r in t.iterrows():
                rows.append({"method": name, "attr": a, "row": f"group={int(r['group'])}",
                             "n_subjects": r["n_subjects"], "tpr": r["tpr"], "fpr": r["fpr"],
                             "alert_rate": r["alert_rate"], "eye_valid_frac": r["eye_valid_frac"]})
            rows.append({"method": name, "attr": a, "row": "GAP (1 minus 0)",
                         "n_subjects": math.nan, "tpr": g["tpr_gap"], "fpr": g["fpr_gap"],
                         "alert_rate": g["dp_diff"], "eye_valid_frac": math.nan,
                         "eq_odds_gap": g["eq_odds_gap"],
                         "eq_odds_ci_lo": g["eq_odds_gap_ci"][0], "eq_odds_ci_hi": g["eq_odds_gap_ci"][1]})
    return pd.DataFrame(rows)
