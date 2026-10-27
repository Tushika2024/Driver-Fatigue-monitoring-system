"""Alert policy: raw levels from model scores -> persistence + hysteresis state machine.

Levels: 0 NONE, 1 MILD, 2 MODERATE, 3 SEVERE, 4 CRITICAL (unacknowledged SEVERE, only when
the vehicle profile can escalate to someone).
Two evidence streams:
    early only                    -> MILD
    early and late, or fused high -> MODERATE
    late and fused very high      -> SEVERE
Degraded mode (eye landmarks unreliable) scores head/posture only, with thresholds widened.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional

from .config import AlertsCfg

NONE, MILD, MODERATE, SEVERE, CRITICAL = 0, 1, 2, 3, 4


@dataclass
class Scores:
    p: float = math.nan      # fused probability (full model)
    pe: float = math.nan     # early-stream probability
    pl: float = math.nan     # late-stream probability
    ph: float = math.nan     # head/posture-only probability (degraded model)
    mode: str = "full"       # full | degraded | blind


@dataclass
class LevelThresholds:
    full: Dict[str, float] = field(default_factory=dict)       # mod, sev, early, late
    degraded: Dict[str, float] = field(default_factory=dict)   # mild, moderate, severe
    decision: float = 0.5                                       # binary decision cut (Youden)
    degraded_decision: float = 0.5


def raw_level(s: Scores, th: LevelThresholds) -> int:
    """Instantaneous level for one window, before persistence."""
    if s.mode == "blind":
        return NONE
    if s.mode == "degraded":
        d = th.degraded
        if not math.isfinite(s.ph):
            return NONE
        return SEVERE if s.ph >= d["severe"] else MODERATE if s.ph >= d["moderate"] \
            else MILD if s.ph >= d["mild"] else NONE
    f = th.full
    early = math.isfinite(s.pe) and s.pe >= f["early"]
    late = math.isfinite(s.pl) and s.pl >= f["late"]
    p = s.p if math.isfinite(s.p) else 0.0
    lvl = NONE
    if early:
        lvl = MILD
    if (early and late) or p >= f["mod"]:
        lvl = max(lvl, MODERATE)
    if late and p >= f["sev"]:
        lvl = SEVERE
    return lvl


@dataclass
class Decision:
    t: float
    level: int
    raw: int
    changed: bool
    mode: str


class AlertPolicy:
    """Persistence (N consecutive windows to raise) + hysteresis (M windows to lower)."""

    def __init__(self, cfg: AlertsCfg, critical_enabled: bool = False, persistence: bool = True):
        self.cfg, self.critical_enabled, self.persistence = cfg, critical_enabled, persistence
        self.persist = {MILD: cfg.persist["mild"], MODERATE: cfg.persist["moderate"],
                        SEVERE: cfg.persist["severe"]}
        self.reset()

    def reset(self) -> None:
        self.level = NONE
        self._run = {MILD: 0, MODERATE: 0, SEVERE: 0}
        self._below = 0
        self._severe_since: Optional[float] = None
        self._last_repeat: Optional[float] = None
        self.acked_at: Optional[float] = None

    def update(self, t: float, raw: int, mode: str = "full") -> Decision:
        prev = self.level
        if mode == "blind":                       # no evidence either way: hold
            return Decision(t, self.level, raw, False, mode)
        for k in self._run:
            self._run[k] = self._run[k] + 1 if raw >= k else 0
        if self.persistence:
            target = max([k for k in self._run if self._run[k] >= self.persist[k]] or [NONE])
        else:
            target = raw
        base = min(self.level, SEVERE)
        if target > base:
            self.level, self._below = target, 0
        elif target < base:
            self._below += 1
            if self._below >= self.cfg.clear_windows or not self.persistence:
                self.level, self._below = target, 0
        else:
            self._below = 0
        # SEVERE bookkeeping and CRITICAL escalation
        if self.level >= SEVERE:
            if self._severe_since is None:
                self._severe_since = t
            if (self.critical_enabled and self.acked_at is None
                    and t - self._severe_since >= self.cfg.critical_after_s):
                self.level = CRITICAL
        else:
            self._severe_since = None
            self._last_repeat = None
        return Decision(t, self.level, raw, self.level != prev, mode)

    def acknowledge(self, t: float) -> None:
        """Driver responded: clear the alert and require fresh evidence."""
        self.level = NONE
        self._run = {k: 0 for k in self._run}
        self._below = 0
        self._severe_since = None
        self._last_repeat = None
        self.acked_at = t

    def should_repeat(self, t: float) -> bool:
        """SEVERE/CRITICAL alarms repeat until acknowledged."""
        if self.level < SEVERE:
            return False
        if self._last_repeat is None or t - self._last_repeat >= self.cfg.ack_repeat_s:
            self._last_repeat = t
            return True
        return False


class BinaryPersistence:
    """Persistence + hysteresis for a plain binary detector (used for the fixed-EAR baseline)."""

    def __init__(self, persist: int, clear: int = 3, persistence: bool = True):
        self.persist, self.clear, self.on = persist, clear, persistence
        self.state = 0
        self._up = self._down = 0

    def update(self, hit: bool) -> int:
        if not self.on:
            self.state = int(hit)
            return self.state
        if hit:
            self._up, self._down = self._up + 1, 0
            if self._up >= self.persist:
                self.state = 1
        else:
            self._down, self._up = self._down + 1, 0
            if self._down >= self.clear:
                self.state = 0
        return self.state
