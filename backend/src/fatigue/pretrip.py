"""Pre-trip fitness check (advisory, never a lockout).

60 s of looking at the camera -> four measures -> compared with the DRIVER'S OWN history
(median/IQR, time-of-day matched when possible). Anything that cannot be measured cleanly is
INCONCLUSIVE: it is never stored, never treated as NOT_FIT, and never blocks the shift.
Identity is by manual profile selection; there is no face recognition or biometric template.
"""
from __future__ import annotations

import json
import math
import os
import re
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Mapping, Optional

import numpy as np
from scipy.ndimage import median_filter

from .blink import segment_blinks
from .config import Config
from .reference import robust_loc_scale

FIT, CAUTION, NOT_FIT, INCONCLUSIVE, ENROLLING = "FIT", "CAUTION", "NOT_FIT", "INCONCLUSIVE", "ENROLLING"
MEASURES = ("reopen", "openness", "rate", "head")
PHRASE = {
    "reopen": "eyes re-opening more slowly than your usual",
    "openness": "eyes less open than your usual",
    "rate": "blink rate unlike your usual",
    "head": "more head movement than your usual",
}
KEYMAP = {"reopen": "reopen_s", "openness": "openness", "rate": "rate_pm", "head": "head_micro"}


@dataclass
class PretripResult:
    verdict: str
    index: Optional[float]
    reason: str
    z: Dict[str, float] = field(default_factory=dict)
    measures: Dict[str, float] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    n_blinks: int = 0
    duration_s: float = 0.0
    stored: bool = False
    history_used: str = "none"
    advisory: bool = True          # always: the check never blocks a shift

    @property
    def blocking(self) -> bool:
        return False


class HistoryStore:
    """Per-driver JSON history, atomic writes, capped length."""

    def __init__(self, root: str):
        self.root = root
        os.makedirs(root, exist_ok=True)

    def _path(self, driver_id: str) -> str:
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", driver_id)
        return os.path.join(self.root, f"{safe}.json")

    def load(self, driver_id: str) -> List[dict]:
        try:
            with open(self._path(driver_id)) as fh:
                return json.load(fh).get("sessions", [])
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    def append(self, driver_id: str, session: dict, keep: int = 200) -> None:
        sessions = (self.load(driver_id) + [session])[-keep:]
        fd, tmp = tempfile.mkstemp(dir=self.root, suffix=".tmp")
        with os.fdopen(fd, "w") as fh:
            json.dump({"driver_id": driver_id, "sessions": sessions}, fh)
        os.replace(tmp, self._path(driver_id))


def compute_measures(rows: List[Mapping[str, float]], cfg: Config) -> Optional[Dict[str, float]]:
    """rows: analysis-period frames (already past the discard period)."""
    t = np.array([r["t"] for r in rows], float)
    face = np.array([r.get("face_found", 0) for r in rows], float)
    ear = np.array([r.get("ear", np.nan) for r in rows], float)
    e = cfg.eyes
    ok = (face > 0) & np.isfinite(ear) & (ear >= e.valid_min) & (ear < e.valid_max)
    if ok.sum() < 30:
        return None
    openness = float(np.percentile(ear[ok], 90))
    x = np.where(ok, ear / openness, np.nan)
    sq = np.array([r.get("squint", np.nan) for r in rows], float)
    blinks = segment_blinks(t, x, cfg.blink, sq)
    dur = float(t[-1] - t[0])
    out = {"n_blinks": float(len(blinks)), "openness": openness,
           "rate_pm": len(blinks) * 60.0 / dur if dur > 0 else math.nan,
           "reopen_s": float(np.median([b.reopen_s for b in blinks])) if blinks else math.nan}
    pit = np.array([r.get("pitch", np.nan) for r in rows], float)
    yaw = np.array([r.get("yaw", np.nan) for r in rows], float)
    k = max(3, int(round(len(rows) / max(dur, 1e-6))))   # ~1 s median filter
    res = []
    for a in (pit, yaw):
        m = np.isfinite(a)
        if m.sum() < 30:
            res.append(math.nan); continue
        aa = np.interp(np.arange(len(a)), np.where(m)[0], a[m])
        med = median_filter(aa, size=k | 1, mode="nearest")
        res.append(float(np.sqrt(np.mean((aa - med) ** 2))))
    out["head_micro"] = float(np.sqrt(np.nansum(np.square(res)))) if not all(math.isnan(r) for r in res) else math.nan
    out["face_frac"] = float(face.mean())
    return out


def quality_check(rows: List[Mapping[str, float]], cfg: Config) -> Optional[str]:
    """Return a human instruction if the setup is not good enough, else None."""
    p = cfg.pretrip
    if not rows:
        return "No video received."
    face = np.array([r.get("face_found", 0) for r in rows], float)
    if face.mean() < p.min_face_frac:
        return "Face not found reliably. Face the camera and make sure nothing blocks it."
    br = np.array([r.get("brightness", np.nan) for r in rows], float)
    if np.isfinite(br).any():
        b = float(np.nanmedian(br))
        if b < p.brightness_range[0]:
            return "Too dark. Turn on the cabin light."
        if b > p.brightness_range[1]:
            return "Too bright or glare. Shade the camera."
    fw = np.array([r.get("face_w_frac", np.nan) for r in rows], float)
    if np.isfinite(fw).any():
        w = float(np.nanmedian(fw))
        if w < p.face_width_range[0]:
            return "Too far from the camera. Move closer."
        if w > p.face_width_range[1]:
            return "Too close to the camera. Move back."
    pit = np.nanmedian([r.get("pitch", np.nan) for r in rows]) if rows else np.nan
    yaw = np.nanmedian([r.get("yaw", np.nan) for r in rows]) if rows else np.nan
    if (np.isfinite(pit) and abs(pit) > p.max_frontal_deg) or (np.isfinite(yaw) and abs(yaw) > p.max_frontal_deg):
        return "Please look straight at the camera."
    return None


def evaluate(measures: Dict[str, float], history: List[dict], now: datetime, cfg: Config):
    """-> (verdict, index, z (oriented: positive = worse), reason, warnings, history_used)."""
    p = cfg.pretrip
    hist = history[-p.history_sessions:]
    if len(hist) < p.min_sessions:
        return (ENROLLING, None, {}, f"Building your personal baseline ({len(hist) + 1} of {p.min_sessions}). "
                "No verdict yet.", [], "none")
    slot = now.hour // p.slot_hours
    matched = [h for h in hist if h.get("slot") == slot]
    use, used = (matched, f"time-of-day slot {slot}") if len(matched) >= p.min_slot_sessions else (hist, "all sessions")
    z, raw_z = {}, {}
    for m in MEASURES:
        key = KEYMAP[m]
        vals = np.array([h[key] for h in use if h.get(key) is not None and math.isfinite(h[key])], float)
        loc, sc = robust_loc_scale(vals, "median_iqr", 0.05)
        v = measures.get(key, math.nan)
        zz = (v - loc) / sc if math.isfinite(v) and math.isfinite(loc) else 0.0
        raw_z[m] = float(np.clip(zz, -10, 10))
        z[m] = abs(raw_z[m]) if m == "rate" else (-raw_z[m] if m == "openness" else raw_z[m])
    index = sum(p.weights[m] * min(max(0.0, z[m]), 4.0) for m in MEASURES)
    verdict = FIT if index < p.caution_at else CAUTION if index < p.not_fit_at else NOT_FIT
    contrib = {m: p.weights[m] * min(max(0.0, z[m]), 4.0) for m in MEASURES}
    top = max(contrib, key=contrib.get)
    reason = ("All measures within your usual range." if verdict == FIT or contrib[top] <= 0
              else f"Main factor: {PHRASE[top]}.")
    warns = []
    if sum(abs(v) >= p.outlier_z for v in raw_z.values()) >= 3:
        warns.append("Measures differ greatly from this profile's history. Confirm the correct driver is selected.")
    return verdict, float(index), raw_z, reason, warns, used


class PretripSession:
    """Feed frames with push(); stops asking for more when .done. Call .result() at the end."""

    def __init__(self, cfg: Config, driver_id: str, store: HistoryStore, now: Optional[datetime] = None):
        self.cfg, self.driver_id, self.store = cfg, driver_id, store
        self.now = now or datetime.now()
        self.rows: List[Mapping[str, float]] = []
        self.t0: Optional[float] = None
        self._quality: Optional[str] = None
        self._quality_done = False
        self.done = False
        self._result: Optional[PretripResult] = None
        self._last_try = -1e9

    def push(self, row: Mapping[str, float]) -> None:
        if self.done:
            return
        p = self.cfg.pretrip
        t = float(row["t"])
        if self.t0 is None:
            self.t0 = t
        self.rows.append(row)
        el = t - self.t0
        if not self._quality_done and el >= p.quality_s:
            self._quality = quality_check(self.rows, self.cfg)
            self._quality_done = True
            if self._quality:
                self.done = True
                self._finish(INCONCLUSIVE, self._quality)
                return
        if el >= p.extended_s:
            self._try_finish(extended=True)
        elif el >= p.capture_s and el - self._last_try >= 1.0:      # re-check once a second, not every frame
            self._last_try = el
            self._try_finish(extended=False)

    def _analysis_rows(self):
        p = self.cfg.pretrip
        start = self.t0 + p.discard_s
        return [r for r in self.rows if float(r["t"]) >= start]

    def _try_finish(self, extended: bool) -> bool:
        p = self.cfg.pretrip
        m = compute_measures(self._analysis_rows(), self.cfg)
        if m is None or m["face_frac"] < 0.8:
            if extended or m is None:
                self.done = True
                self._finish(INCONCLUSIVE, "Could not see the eyes clearly enough to measure.")
                return True
            return False
        if m["n_blinks"] < p.min_blinks:
            if not extended:
                return False       # keep capturing up to extended_s
            self.done = True
            self._finish(INCONCLUSIVE, f"Too few blinks observed ({int(m['n_blinks'])}) to measure reliably.")
            return True
        self.done = True
        v, idx, z, reason, warns, used = evaluate(m, self.store.load(self.driver_id), self.now, self.cfg)
        self._result = PretripResult(v, idx, reason, z, {k: m[k] for k in KEYMAP.values()}, warns,
                                     int(m["n_blinks"]), self.rows[-1]["t"] - self.t0, False, used)
        sess = {"ts": self.now.isoformat(), "slot": self.now.hour // p.slot_hours,
                **{k: m[k] for k in KEYMAP.values()}}
        self.store.append(self.driver_id, sess)
        self._result.stored = True
        return True

    def _finish(self, verdict: str, reason: str) -> None:
        self._result = PretripResult(verdict, None, reason, duration_s=(self.rows[-1]["t"] - self.t0) if self.rows else 0.0,
                                     history_used="none")   # INCONCLUSIVE: not stored

    def result(self) -> PretripResult:
        if self._result is None:     # source ended early
            self.done = True
            if not self._try_finish(extended=True):
                self._finish(INCONCLUSIVE, "The video ended before the check finished.")
        return self._result  # type: ignore[return-value]
