"""Classifiers: a simple, explainable logistic regression (only ~12 people to learn from),
fused over eye / head streams, with a degraded-mode head-only model."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import schema
from .alerts import LevelThresholds
from .config import Config

REASONS = {
    "perclos": "eyes closed more of the time",
    "longest_closure_s": "long eye closures",
    "ear_rel_mean": "eyes drooping",
    "reopen_med_s": "eyes re-opening slowly",
    "closing_med_s": "slow eye closing",
    "asym": "slow re-opening after blinks",
    "blink_rate_pm": "unusual blink rate",
    "ear_diverge": "left and right eyes out of step",
    "compensatory_count": "signs of fighting sleep",
    "n_eye_rub": "rubbing eyes",
    "n_head_shake": "shaking head to stay awake",
    "n_posture_shift": "shifting or stretching in the seat",
    "n_hard_blink": "forced hard blinks",
    "yawn_frac": "yawning",
    "head_drop_frac": "head dropping",
    "pitch_rel_mean": "head tilted forward",
    "pitch_std": "head bobbing",
    "yaw_std": "head wandering",
    "head_above_rel": "slouching",
    "head_above_std": "restless posture",
    "tilt_std": "leaning",
}


def window_mode(df: pd.DataFrame, cfg: Config) -> pd.Series:
    q = cfg.quality
    mode = np.where(df["face_found_frac"] < q.blind_face_frac, "blind",
                    np.where(df["eye_valid_frac"] < q.degraded_eye_frac, "degraded", "full"))
    return pd.Series(mode, index=df.index)


def _pipeline(cfg: Config) -> Pipeline:
    if cfg.model.kind == "hgb":
        clf = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.08, max_iter=150,
                                             class_weight="balanced", random_state=0)
        return Pipeline([("clf", clf)])
    clf = LogisticRegression(C=cfg.model.C, class_weight="balanced", max_iter=1000)
    return Pipeline([("imp", SimpleImputer(strategy="median", keep_empty_features=True)),
                     ("sc", StandardScaler()), ("clf", clf)])


def _explainer(cfg: Config) -> Pipeline:
    clf = LogisticRegression(C=cfg.model.C, class_weight="balanced", max_iter=1000)
    return Pipeline([("imp", SimpleImputer(strategy="median", keep_empty_features=True)),
                     ("sc", StandardScaler()), ("clf", clf)])


def youden_threshold(y: np.ndarray, p: np.ndarray) -> float:
    """Threshold maximising TPR - FPR (robust to class imbalance)."""
    ok = np.isfinite(p)
    y, p = y[ok], p[ok]
    if len(np.unique(y)) < 2:
        return 0.5
    best, thr = -1.0, 0.5
    for t in np.unique(np.quantile(p, np.linspace(0.02, 0.98, 97))):
        j = np.mean(p[y == 1] >= t) - np.mean(p[y == 0] >= t)
        if j > best:
            best, thr = j, float(t)
    return thr


def neg_quantile(p_neg: np.ndarray, fpr: float) -> float:
    p_neg = p_neg[np.isfinite(p_neg)]
    if len(p_neg) == 0:
        return 0.5
    return float(np.quantile(p_neg, 1.0 - fpr))


@dataclass
class FusionModel:
    cfg: Config
    full_cols: List[str]
    early_cols: List[str]
    late_cols: List[str]
    head_cols: List[str]
    pipes: Dict[str, Pipeline] = field(default_factory=dict)
    explainers: Dict[str, Pipeline] = field(default_factory=dict)
    th: LevelThresholds = field(default_factory=LevelThresholds)

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_features(cls, cfg: Config, full_cols: Optional[List[str]] = None) -> "FusionModel":
        z = cfg.model.use_z
        full = schema.resolve(full_cols or schema.FULL_FEATURES, z)
        keep = lambda cols: [c for c in schema.resolve(cols, z) if c in full]
        eyeless = [c for c in full if c in schema.resolve(schema.DEGRADED_FEATURES, z)]
        return cls(cfg, full,
                   keep(schema.EARLY_FEATURES) or full, keep(schema.LATE_FEATURES) or full,
                   eyeless or full)

    def _cols(self, name: str) -> List[str]:
        return {"full": self.full_cols, "early": self.early_cols, "late": self.late_cols,
                "head": self.head_cols}[name]

    # ------------------------------------------------------------------ fitting
    def _fit_set(self, df, y, names=("full", "early", "late", "head")):
        mode = window_mode(df, self.cfg)
        pipes = {}
        for n in names:
            m = (mode != "blind") if n == "head" else (mode == "full")
            if m.sum() < 10 or len(np.unique(y[m.to_numpy()])) < 2:
                m = mode != "blind"
            pipes[n] = _pipeline(self.cfg).fit(df.loc[m, self._cols(n)].to_numpy(float),
                                               y[m.to_numpy()])
        return pipes

    def fit(self, df: pd.DataFrame, y: np.ndarray, groups: np.ndarray) -> "FusionModel":
        df = df.reset_index(drop=True)
        y = np.asarray(y).astype(int)
        # inner out-of-fold scores by subject -> threshold calibration without leakage
        k = min(self.cfg.model.inner_folds, len(np.unique(groups)))
        oof = {n: np.full(len(df), np.nan) for n in ("full", "early", "late", "head")}
        if k >= 2:
            for tr, va in GroupKFold(n_splits=k).split(df, y, groups):
                pipes = self._fit_set(df.iloc[tr], y[tr])
                for n, pp in pipes.items():
                    oof[n][va] = pp.predict_proba(df.iloc[va][self._cols(n)].to_numpy(float))[:, 1]
        self.pipes = self._fit_set(df, y)
        if self.cfg.model.kind == "hgb":
            self.explainers = {n: _explainer(self.cfg).fit(df[self._cols(n)].to_numpy(float), y)
                               for n in ("full", "head")}
        if k < 2:   # tiny training set: fall back to in-sample scores
            for n, pp in self.pipes.items():
                oof[n] = pp.predict_proba(df[self._cols(n)].to_numpy(float))[:, 1]
        self._calibrate(df, y, oof)
        return self

    def _calibrate(self, df, y, oof) -> None:
        a, mode = self.cfg.alerts, window_mode(df, self.cfg)
        neg = (y == 0)
        full_neg = neg & (mode.to_numpy() == "full")
        t = a.target_fpr
        self.th = LevelThresholds(
            full={"mod": neg_quantile(oof["full"][full_neg], t["moderate"]),
                  "sev": neg_quantile(oof["full"][full_neg], t["severe"]),
                  "early": neg_quantile(oof["early"][full_neg], t["mild"]),
                  "late": neg_quantile(oof["late"][full_neg], t["late"])},
            degraded={"mild": neg_quantile(oof["head"][neg], t["mild"] * a.degraded_fpr_scale),
                      "moderate": neg_quantile(oof["head"][neg], t["moderate"] * a.degraded_fpr_scale),
                      "severe": neg_quantile(oof["head"][neg], t["severe"] * a.degraded_fpr_scale)},
            decision=youden_threshold(y[full_neg | (y == 1) & (mode.to_numpy() == "full")],
                                      oof["full"][full_neg | (y == 1) & (mode.to_numpy() == "full")]),
            degraded_decision=youden_threshold(y, oof["head"]),
        )
        self._oof = {n: v.copy() for n, v in oof.items()}
        self._oof_y, self._oof_mode = y.copy(), mode.to_numpy()

    # ------------------------------------------------------------------ scoring
    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        mode = window_mode(df, self.cfg).to_numpy()
        out = pd.DataFrame(index=df.index, columns=["p", "pe", "pl", "ph", "mode"], dtype=object)
        for n, col in (("full", "p"), ("early", "pe"), ("late", "pl"), ("head", "ph")):
            vals = self.pipes[n].predict_proba(df[self._cols(n)].to_numpy(float))[:, 1]
            if n == "head":
                vals = np.where(mode != "blind", vals, np.nan)
            else:
                vals = np.where(mode == "full", vals, np.nan)
            out[col] = vals.astype(float)
        out["mode"] = mode
        out["p_used"] = np.where(mode == "full", out["p"].astype(float), out["ph"].astype(float))
        return out

    def explain(self, row: pd.Series, mode: str, top: int = 2) -> List[Tuple[str, float]]:
        """Top positive contributors (coef x standardised value) for the active model."""
        name = "head" if mode == "degraded" else "full"
        pipe = self.explainers.get(name, self.pipes[name])
        cols = self._cols(name)
        x = pd.DataFrame([row[cols].to_numpy(float)], columns=cols)
        xs = pipe[:-1].transform(x.to_numpy(float))[0]
        contrib = pipe[-1].coef_[0] * xs
        order = np.argsort(contrib)[::-1][:top]
        return [(cols[i], float(contrib[i])) for i in order if contrib[i] > 0]

    def reason(self, row: pd.Series, mode: str) -> str:
        items = self.explain(row, mode, top=2)
        if not items:
            return "overall fatigue pattern" if mode != "degraded" else "head and posture pattern"
        txt = " and ".join(REASONS.get(f, f) for f, _ in items)
        return txt + (" (eyes not readable: using head and posture only)" if mode == "degraded" else "")

    # ------------------------------------------------------------------ persistence
    def save(self, path: str) -> None:
        joblib.dump({"version": 1, "model": self, "cfg_hash": self.cfg.hash()}, path)

    @staticmethod
    def load(path: str) -> "FusionModel":
        blob = joblib.load(path)
        return blob["model"]
