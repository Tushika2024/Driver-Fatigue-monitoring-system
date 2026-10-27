"""Per-driver references.

fixed     population constants; z-scores come from population statistics (PopulationNormaliser)
adaptive  running high percentile of the driver's own open-eye history; median/IQR z-scores
          over a trailing window of that driver's own windows. No assumption the driver starts alert.
enrolled  adaptive during the first `enrol_s`, then frozen (ablation; can be seeded from a profile)
"""
from __future__ import annotations

import math
from collections import deque
from typing import Deque, Dict, Iterable, Optional, Tuple

import numpy as np

from .config import Config, ReferenceCfg


class RollingStat:
    """Time-bounded rolling store with percentile/median queries (subsampled for speed)."""

    def __init__(self, horizon_s: float, subsample_s: float = 0.1):
        self.h, self.dt = horizon_s, subsample_s
        self.q: Deque[Tuple[float, float]] = deque()
        self._last = -1e9

    def add(self, t: float, v: float) -> None:
        if not math.isfinite(v):
            return
        if t - self._last >= self.dt:
            self.q.append((t, v))
            self._last = t
        while self.q and t - self.q[0][0] > self.h:
            self.q.popleft()

    def span(self) -> float:
        return self.q[-1][0] - self.q[0][0] if len(self.q) > 1 else 0.0

    def percentile(self, p: float) -> float:
        if not self.q:
            return math.nan
        return float(np.percentile([v for _, v in self.q], p))


class Reference:
    """Open-eye level, neutral head pitch and neutral head height for one driver."""

    def __init__(self, cfg: Config, mode: Optional[str] = None):
        self.cfg, self.rc = cfg, cfg.reference
        self.mode = mode or self.rc.mode
        self._ear = RollingStat(self.rc.history_s)
        self._pitch = RollingStat(self.rc.pitch_history_s)
        self._ha = RollingStat(self.rc.pitch_history_s)
        self._t0: Optional[float] = None
        self._frozen: Optional[Dict[str, float]] = None
        self._cache: Dict[str, float] = {}
        self._cache_t = -1e9

    # -- lifecycle ---------------------------------------------------------------
    def update(self, t: float, ear: float, pitch: float, head_above: float) -> None:
        if self._t0 is None:
            self._t0 = t
        if self._frozen is not None or self.mode == "fixed":
            return
        self._ear.add(t, ear)
        self._pitch.add(t, pitch)
        self._ha.add(t, head_above)
        if t - self._cache_t >= 0.5:          # percentiles are costly: refresh twice a second
            self._cache_t = t
            self._cache = {"open": self._ear.percentile(self.rc.open_percentile),
                           "pitch0": self._pitch.percentile(50), "ha0": self._ha.percentile(50)}
        if self.mode == "enrolled" and t - self._t0 >= self.rc.enrol_s and self.ready:
            self.freeze()

    def freeze(self) -> None:
        self._frozen = {"open": self.open_level, "pitch0": self.pitch_zero, "ha0": self.head_above_zero}

    # -- values ------------------------------------------------------------------
    @property
    def ready(self) -> bool:
        if self.mode == "fixed" or self._frozen is not None:
            return True
        return self._ear.span() >= self.rc.min_history_s

    @property
    def open_level(self) -> float:
        if self._frozen:
            return self._frozen["open"]
        if self.mode == "fixed" or not self.ready:
            return self.cfg.eyes.fixed_open_ear
        v = self._cache.get("open", math.nan)
        return v if math.isfinite(v) and v > 0 else self.cfg.eyes.fixed_open_ear

    @property
    def pitch_zero(self) -> float:
        if self._frozen:
            return self._frozen["pitch0"]
        if self.mode == "fixed" or not self.ready:
            return 0.0
        v = self._cache.get("pitch0", math.nan)
        return v if math.isfinite(v) else 0.0

    @property
    def head_above_zero(self) -> float:
        if self._frozen:
            return self._frozen["ha0"]
        if self.mode == "fixed" or not self.ready:
            return self.cfg.head.fixed_head_above
        v = self._cache.get("ha0", math.nan)
        return v if math.isfinite(v) else self.cfg.head.fixed_head_above

    def to_profile(self) -> Dict[str, float]:
        return {"open": self.open_level, "pitch0": self.pitch_zero, "ha0": self.head_above_zero}

    @classmethod
    def from_profile(cls, cfg: Config, prof: Dict[str, float]) -> "Reference":
        r = cls(cfg, "enrolled")
        r._frozen = dict(prof)
        return r


# --- normalisation of early-signal features -----------------------------------------------

def robust_loc_scale(vals: np.ndarray, method: str, scale_floor: float) -> Tuple[float, float]:
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return math.nan, math.nan
    if method == "median_iqr":
        loc = float(np.median(vals))
        q1, q3 = np.percentile(vals, [25, 75])
        scale = float(q3 - q1) / 1.349
    else:
        loc, scale = float(np.mean(vals)), float(np.std(vals))
    return loc, max(scale, scale_floor * abs(loc), 1e-6)


def zscore(x: float, loc: float, scale: float, clip: float = 8.0) -> float:
    if not (math.isfinite(x) and math.isfinite(loc) and math.isfinite(scale)):
        return math.nan
    return float(np.clip((x - loc) / scale, -clip, clip))


class DriverNormaliser:
    """Z-scores a driver's early-signal features against that same driver's own history.

    history mode  : trailing `norm_history_windows` windows (adaptive)
    frozen mode   : stats fixed after enrolment, or loaded from a profile (enrolled)
    """

    def __init__(self, rc: ReferenceCfg, features: Iterable[str], frozen: Optional[Dict] = None):
        self.rc, self.features = rc, list(features)
        self.hist: Dict[str, Deque[float]] = {f: deque(maxlen=rc.norm_history_windows) for f in self.features}
        self.frozen: Optional[Dict[str, Tuple[float, float]]] = (
            {k: tuple(v) for k, v in frozen.items()} if frozen else None)
        self.n = 0

    @property
    def ready(self) -> bool:
        return self.frozen is not None or self.n >= self.rc.norm_min_windows

    def stats(self) -> Dict[str, Tuple[float, float]]:
        if self.frozen is not None:
            return self.frozen
        return {f: robust_loc_scale(np.array(self.hist[f]), self.rc.norm_method, self.rc.scale_floor)
                for f in self.features}

    def transform(self, feats: Dict[str, float]) -> Dict[str, float]:
        if not self.ready:
            return {f"z_{f}": math.nan for f in self.features}
        st = self.stats()
        return {f"z_{f}": zscore(feats.get(f, math.nan), *st[f]) for f in self.features}

    def update(self, feats: Dict[str, float]) -> None:
        if self.frozen is not None:
            return
        for f in self.features:
            v = feats.get(f, math.nan)
            if math.isfinite(v):
                self.hist[f].append(v)
        self.n += 1

    def freeze(self) -> None:
        if self.frozen is None:
            self.frozen = dict(self.stats())

    def to_profile(self) -> Dict[str, Tuple[float, float]]:
        return dict(self.stats())


class PopulationNormaliser:
    """'fixed' mode: loc/scale learned from TRAINING subjects only, applied to everyone."""

    def __init__(self, method: str = "median_iqr", scale_floor: float = 0.05):
        self.method, self.floor = method, scale_floor
        self.stats_: Dict[str, Tuple[float, float]] = {}

    def fit(self, df, features: Iterable[str]) -> "PopulationNormaliser":
        for f in features:
            self.stats_[f] = robust_loc_scale(df[f].to_numpy(float), self.method, self.floor)
        return self

    def transform(self, df):
        out = df.copy()
        for f, (loc, sc) in self.stats_.items():
            out[f"z_{f}"] = [zscore(x, loc, sc) for x in df[f].to_numpy(float)]
        return out
