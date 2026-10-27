"""Streaming blink segmentation with closing / closed / re-opening timing.

Input is eye openness relative to the driver's own open level (1.0 = fully open).
Timing is measured at amplitude crossings (80% / 20% of the blink's own depth) with
linear interpolation, so it is meaningful at 30 FPS.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .config import BlinkCfg


@dataclass
class Blink:
    t_start: float
    t_end: float
    closing_s: float
    closed_s: float
    reopen_s: float
    depth: float
    squint: float = math.nan

    @property
    def duration_s(self) -> float:
        return self.t_end - self.t_start

    @property
    def asym(self) -> float:
        return self.reopen_s / self.closing_s if self.closing_s > 1e-6 else math.nan


def _cross(t0: float, x0: float, t1: float, x1: float, level: float) -> float:
    if x1 == x0:
        return t1
    return t0 + (level - x0) * (t1 - t0) / (x1 - x0)


def _analyse(ts: np.ndarray, xs: np.ndarray, sq: np.ndarray, cfg: BlinkCfg) -> Optional[Blink]:
    """ts/xs cover one episode including one sample above start_level on each side."""
    i_min = int(np.argmin(xs))
    trough = float(xs[i_min])
    depth = 1.0 - trough
    if depth < cfg.min_depth:
        return None
    hi = 1.0 - 0.2 * depth       # 20% closed
    lo = 1.0 - 0.8 * depth       # 80% closed
    # last time above `hi` before the trough
    i = i_min
    while i > 0 and xs[i] < hi:
        i -= 1
    t_close0 = _cross(ts[i], xs[i], ts[i + 1], xs[i + 1], hi) if xs[i] >= hi else ts[0]
    # first time below `lo` going down
    j = i
    while j < i_min and xs[j] > lo:
        j += 1
    t_lo_down = _cross(ts[j - 1], xs[j - 1], ts[j], xs[j], lo) if j > 0 and xs[j] <= lo else ts[j]
    # last time below `lo` going up
    k = i_min
    while k < len(xs) - 1 and xs[k + 1] <= lo:
        k += 1
    if k + 1 < len(xs):
        t_lo_up = _cross(ts[k], xs[k], ts[k + 1], xs[k + 1], lo)
    else:
        return None
    m = k + 1
    while m < len(xs) - 1 and xs[m] < hi:
        m += 1
    if xs[m] < hi:
        return None
    t_hi_up = _cross(ts[m - 1], xs[m - 1], ts[m], xs[m], hi) if m > 0 else ts[m]
    closing, closed, reopen = t_lo_down - t_close0, t_lo_up - t_lo_down, t_hi_up - t_lo_up
    if closing < 0 or closed < 0 or reopen < 0:
        return None
    s = float(np.nanmax(sq)) if np.any(np.isfinite(sq)) else math.nan
    return Blink(float(t_close0), float(t_hi_up), float(closing), float(closed), float(reopen),
                 float(depth), s)


class BlinkDetector:
    """Causal state machine. push() returns a Blink when one has just finished."""

    def __init__(self, cfg: BlinkCfg):
        self.cfg = cfg
        self._reset()

    def _reset(self) -> None:
        self._prev: Optional[Tuple[float, float, float]] = None   # last good sample above level
        self._ep: List[Tuple[float, float, float]] = []
        self._last_t: Optional[float] = None

    def reset(self) -> None:
        self._reset()

    def push(self, t: float, x: float, squint: float = math.nan) -> Optional[Blink]:
        c = self.cfg
        if not math.isfinite(x):
            # invalid sample: tolerate a short gap, abort a longer one
            if self._last_t is not None and t - self._last_t > c.max_gap_s:
                self._reset()
            return None
        if self._last_t is not None and t - self._last_t > c.max_gap_s * 3 and self._ep:
            self._reset()
        self._last_t = t
        sample = (t, x, squint)
        if x >= c.start_level:
            out = None
            if self._ep:
                ep = [self._prev] + self._ep + [sample] if self._prev else self._ep + [sample]
                arr = np.array(ep, dtype=float)
                if arr[-1, 0] - arr[0, 0] <= c.max_blink_s and self._prev is not None:
                    out = _analyse(arr[:, 0], arr[:, 1], arr[:, 2], c)
                self._ep = []
            self._prev = sample
            return out
        # below start level
        if self._prev is None:
            return None   # episode began before we saw an open eye: not a clean blink
        self._ep.append(sample)
        if self._ep[-1][0] - self._ep[0][0] > c.max_blink_s:
            # sustained low openness is a closure/droop, not a blink
            self._ep = []
            self._prev = None
        return None


def segment_blinks(t: Sequence[float], x: Sequence[float], cfg: BlinkCfg,
                   squint: Optional[Sequence[float]] = None) -> List[Blink]:
    """Batch helper (pre-trip, tests) built on the same streaming detector."""
    det = BlinkDetector(cfg)
    sq = squint if squint is not None else [math.nan] * len(t)
    out = []
    for ti, xi, si in zip(t, x, sq):
        b = det.push(float(ti), float(xi), float(si))
        if b is not None:
            out.append(b)
    return out
