"""Compensatory-behaviour detectors: the early cues that drivers fight sleep.

RLDD has no labels for these. They are rule-based, interpretable, and evaluated
only by whether they improve earliness in ablation (or on a hand-annotated subset).
"""
from __future__ import annotations

import math
from collections import deque
from typing import Deque, List, Optional, Tuple

from .blink import Blink
from .config import BehaviourCfg, BlinkCfg

EYE_RUB, HEAD_SHAKE, POSTURE_SHIFT, HARD_BLINK = "eye_rub", "head_shake", "posture_shift", "hard_blink"
KINDS = (EYE_RUB, HEAD_SHAKE, POSTURE_SHIFT, HARD_BLINK)


class BehaviourDetector:
    def __init__(self, cfg: BehaviourCfg, blink_cfg: BlinkCfg):
        self.cfg, self.bcfg = cfg, blink_cfg
        self._rub_start: Optional[float] = None
        self._rub_emitted = False
        self._yaw: Deque[Tuple[float, float]] = deque()
        self._post: Deque[Tuple[float, float, float]] = deque()
        self._last = {k: -1e9 for k in KINDS}

    def _ok(self, kind: str, t: float) -> bool:
        if t - self._last[kind] >= self.cfg.refractory_s:
            self._last[kind] = t
            return True
        return False

    def push(self, t: float, wrist_dist: float, wrist_vis: float, yaw: float,
             head_above: float, tilt: float) -> List[Tuple[float, str]]:
        c, ev = self.cfg, []
        # --- hand-to-face / eye rubbing --------------------------------------
        near = (math.isfinite(wrist_dist) and wrist_dist < c.hand_face_dist
                and wrist_vis >= c.hand_min_vis)
        if near:
            if self._rub_start is None:
                self._rub_start, self._rub_emitted = t, False
            elif not self._rub_emitted and t - self._rub_start >= c.hand_min_s:
                if self._ok(EYE_RUB, t):
                    ev.append((t, EYE_RUB))
                self._rub_emitted = True
        elif not (math.isfinite(wrist_dist) and wrist_dist < c.hand_face_dist * 1.2):
            self._rub_start = None
        # --- sharp head shake -------------------------------------------------
        if math.isfinite(yaw):
            self._yaw.append((t, yaw))
            while self._yaw and t - self._yaw[0][0] > c.shake_window_s:
                self._yaw.popleft()
            if len(self._yaw) >= 6:
                ys = [y for _, y in self._yaw]
                if max(ys) - min(ys) >= c.shake_pp_deg:
                    # count direction reversals of meaningful size (>2 deg)
                    rev, last_dir, anchor = 0, 0, ys[0]
                    for y in ys[1:]:
                        if abs(y - anchor) > 2.0:
                            d = 1 if y > anchor else -1
                            if last_dir and d != last_dir:
                                rev += 1
                            last_dir, anchor = d, y
                    if rev >= 2 and self._ok(HEAD_SHAKE, t):
                        ev.append((t, HEAD_SHAKE))
        # --- postural shift / stretch ----------------------------------------
        if math.isfinite(head_above) and math.isfinite(tilt):
            self._post.append((t, head_above, tilt))
            while self._post and t - self._post[0][0] > c.shift_window_s:
                self._post.popleft()
            if len(self._post) >= 3:
                t0, h0, k0 = self._post[0]
                if (abs(head_above - h0) >= c.shift_delta or abs(tilt - k0) >= c.shift_delta) \
                        and self._ok(POSTURE_SHIFT, t):
                    ev.append((t, POSTURE_SHIFT))
        return ev

    def from_blink(self, b: Blink) -> Optional[Tuple[float, str]]:
        """A deliberate, squeezed, near-complete blink."""
        if b.depth >= 0.85 and math.isfinite(b.squint) and b.squint >= self.bcfg.hard_squint:
            if self._ok(HARD_BLINK, b.t_end):
                return (b.t_end, HARD_BLINK)
        return None
