"""One causal frame-stream -> window-feature processor.

The SAME class is used offline (building training windows from features.csv) and
online (live webcam), so there is no train/serve skew. It only ever looks at frames
already pushed, which tests/test_stream.py verifies.
"""
from __future__ import annotations

import math
from collections import deque
from typing import Deque, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np

from .behaviour import (EYE_RUB, HARD_BLINK, HEAD_SHAKE, POSTURE_SHIFT, BehaviourDetector)
from .blink import Blink, BlinkDetector
from .config import Config
from .reference import DriverNormaliser, Reference
from .schema import Z_SOURCES

NAN = math.nan


def _nz(x, default=NAN):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return default
    return x if math.isfinite(x) else default


class StreamProcessor:
    def __init__(self, cfg: Config, mode: Optional[str] = None,
                 ref: Optional[Reference] = None, norm: Optional[DriverNormaliser] = None,
                 window_s: Optional[float] = None):
        self.cfg = cfg
        self.mode = mode or cfg.reference.mode
        self.L = window_s or cfg.window.length_s
        self.stride = cfg.window.stride_s
        self.ref = ref or Reference(cfg, self.mode)
        if self.mode == "fixed":
            self.norm: Optional[DriverNormaliser] = None
        else:
            self.norm = norm or DriverNormaliser(cfg.reference, Z_SOURCES)
        self.blinks = BlinkDetector(cfg.blink)
        self.beh = BehaviourDetector(cfg.behaviour, cfg.blink)
        self.buf: Deque[dict] = deque()
        self.blink_hist: Deque[Blink] = deque()
        self.events: Deque[Tuple[float, str]] = deque()
        self.microsleeps: List[float] = []          # onset times of closures >= microsleep_s
        self._closure_start: Optional[float] = None
        self._closure_flagged = False
        self._t0: Optional[float] = None
        self._next_emit: Optional[float] = None
        self._hold: Dict[str, Tuple[float, float]] = {}   # last good blink-shape values + time
        self._enrol_frozen = False

    # ------------------------------------------------------------------ per frame
    def push(self, row: Mapping[str, float]) -> Optional[dict]:
        cfg, e = self.cfg, self.cfg.eyes
        t = float(row["t"])
        if self._t0 is None:
            self._t0 = t
            self._next_emit = t + self.L
        face = int(_nz(row.get("face_found"), 0) > 0)
        ear, el, er = _nz(row.get("ear")), _nz(row.get("ear_l")), _nz(row.get("ear_r"))
        pitch, yaw = _nz(row.get("pitch")), _nz(row.get("yaw"))
        ha, tilt = _nz(row.get("head_above")), _nz(row.get("tilt"))
        eye_ok = bool(face and math.isfinite(ear) and e.valid_min <= ear < e.valid_max)

        self.ref.update(t, ear if eye_ok else NAN, pitch if face else NAN, ha)
        op = self.ref.open_level
        x = ear / op if eye_ok else NAN
        xl = el / op if (eye_ok and math.isfinite(el)) else NAN
        xr = er / op if (eye_ok and math.isfinite(er)) else NAN

        blink = self.blinks.push(t, x, _nz(row.get("squint")))
        if blink is not None:
            self.blink_hist.append(blink)
            hb = self.beh.from_blink(blink)
            if hb:
                self.events.append(hb)
        for ev in self.beh.push(t, _nz(row.get("wrist_dist")), _nz(row.get("wrist_vis"), 0.0),
                                yaw if face else NAN, ha, tilt):
            self.events.append(ev)

        # microsleep tracking (continuous closure on valid eye frames)
        if eye_ok and x < e.closed_frac:
            if self._closure_start is None:
                self._closure_start, self._closure_flagged = t, False
            elif not self._closure_flagged and t - self._closure_start >= e.microsleep_s:
                self.microsleeps.append(self._closure_start)
                self._closure_flagged = True
        else:
            if eye_ok or not face:     # a lost eye signal ends the run
                self._closure_start = None

        pz = self.ref.pitch_zero
        self.buf.append({
            "t": t, "face": face, "eye_ok": eye_ok, "ear": ear, "x": x, "xl": xl, "xr": xr,
            "pitch_rel": pitch - pz if (face and math.isfinite(pitch)) else NAN,
            "yaw": yaw if face else NAN, "mar": _nz(row.get("mar")) if face else NAN,
            "ha_rel": ha - self.ref.head_above_zero if math.isfinite(ha) else NAN,
            "tilt": tilt,
        })
        while self.buf and t - self.buf[0]["t"] >= self.L:
            self.buf.popleft()
        ctx = max(cfg.blink.context_s, cfg.behaviour.context_s)
        while self.blink_hist and t - self.blink_hist[0].t_end > ctx:
            self.blink_hist.popleft()
        while self.events and t - self.events[0][0] > ctx:
            self.events.popleft()

        if t + 1e-9 >= self._next_emit:
            while self._next_emit <= t + 1e-9:
                self._next_emit += self.stride
            return self._emit(t)
        return None

    def run(self, rows: Iterable[Mapping[str, float]]) -> List[dict]:
        out = []
        for r in rows:
            w = self.push(r)
            if w is not None:
                out.append(w)
        return out

    # ------------------------------------------------------------------ window
    def _emit(self, t: float) -> dict:
        cfg, e = self.cfg, self.cfg.eyes
        b = list(self.buf)
        n = len(b)
        arr = lambda k: np.array([f[k] for f in b], dtype=float)
        face, eye_ok = arr("face"), np.array([f["eye_ok"] for f in b], dtype=bool)
        x, xl, xr, ear = arr("x"), arr("xl"), arr("xr"), arr("ear")
        pr, yaw, mar = arr("pitch_rel"), arr("yaw"), arr("mar")
        har, tilt = arr("ha_rel"), arr("tilt")
        ts = arr("t")
        dt = float(np.median(np.diff(ts))) if n > 1 else 1.0 / cfg.extraction.fps

        f: Dict[str, float] = {"t_start": float(ts[0]), "t_end": t}
        f["face_found_frac"] = float(face.mean())
        f["eye_valid_frac"] = float(eye_ok.mean())
        n_eye = int(eye_ok.sum())

        closed = eye_ok & (x < e.closed_frac)
        f["perclos"] = float(closed.sum() / n_eye) if n_eye else NAN
        f["ear_rel_mean"] = float(np.nanmean(x)) if n_eye else NAN
        f["longest_closure_s"] = self._longest_run(ts, closed, dt) if n_eye else NAN
        both = np.isfinite(xl) & np.isfinite(xr)
        f["ear_diverge"] = float(np.mean(np.abs(xl[both] - xr[both]))) if both.any() else NAN
        for thr in e.fixed_grid:   # fixed-threshold baseline candidates (absolute EAR)
            ok = eye_ok
            f[f"pc_abs_{thr:.2f}"] = float(((ear < thr) & ok).sum() / ok.sum()) if ok.any() else NAN

        pv = np.isfinite(pr)
        f["pitch_rel_mean"] = float(np.mean(pr[pv])) if pv.any() else NAN
        f["pitch_std"] = float(np.std(pr[pv])) if pv.sum() > 2 else NAN
        f["head_drop_frac"] = float(np.mean(pr[pv] > cfg.head.drop_deg)) if pv.any() else NAN
        yv = np.isfinite(yaw)
        f["yaw_std"] = float(np.std(yaw[yv])) if yv.sum() > 2 else NAN
        hv = np.isfinite(har)
        f["head_above_rel"] = float(np.mean(har[hv])) if hv.any() else NAN
        f["head_above_std"] = float(np.std(har[hv])) if hv.sum() > 2 else NAN
        tv = np.isfinite(tilt)
        f["tilt_std"] = float(np.std(tilt[tv])) if tv.sum() > 2 else NAN

        mv = np.isfinite(mar)
        yawning = mv & (mar > cfg.mouth.yawn_mar)
        f["yawn_frac"] = float(yawning.sum() / mv.sum()) if mv.any() else NAN
        f["yawn_run_s"] = self._longest_run(ts, yawning, dt) if mv.any() else NAN

        # ---- trailing-context features (blink shape, behaviours) ----
        bl = [k for k in self.blink_hist if t - k.t_end <= cfg.blink.context_s]
        elapsed = min(cfg.blink.context_s, t - (self._t0 if self._t0 is not None else t))
        f["n_blinks_ctx"] = float(len(bl))
        f["blink_rate_pm"] = len(bl) * 60.0 / elapsed if elapsed >= 10 else NAN
        shape = {
            "reopen_med_s": float(np.median([k.reopen_s for k in bl])) if len(bl) >= 2 else NAN,
            "closing_med_s": float(np.median([k.closing_s for k in bl])) if len(bl) >= 2 else NAN,
            "asym": float(np.nanmedian([k.asym for k in bl])) if len(bl) >= 2 else NAN,
        }
        for k, v in shape.items():          # hold the last good value for up to 60 s
            if math.isfinite(v):
                self._hold[k] = (v, t)
            elif k in self._hold and t - self._hold[k][1] <= 60.0:
                v = self._hold[k][0]
            f[k] = v
        f["slow_blink_frac"] = (float(np.mean([k.closed_s + k.closing_s + k.reopen_s > cfg.blink.slow_blink_s
                                               for k in bl])) if bl else NAN)
        evs = [k for tt, k in self.events if t - tt <= cfg.behaviour.context_s]
        counts = {k: float(evs.count(k)) for k in (EYE_RUB, HEAD_SHAKE, POSTURE_SHIFT, HARD_BLINK)}
        f["n_eye_rub"], f["n_head_shake"] = counts[EYE_RUB], counts[HEAD_SHAKE]
        f["n_posture_shift"], f["n_hard_blink"] = counts[POSTURE_SHIFT], counts[HARD_BLINK]
        f["compensatory_count"] = float(sum(counts.values()))

        # ---- reference-relative scaling and z-scores ----
        f["open_level"] = self.ref.open_level
        ready = self.ref.ready
        if self.norm is not None:
            if self.mode == "enrolled" and self.ref._frozen is not None and not self._enrol_frozen:
                self.norm.freeze()
                self._enrol_frozen = True
            f.update(self.norm.transform(f))
            self.norm.update(f)
            ready = ready and self.norm.ready
        else:
            f.update({f"z_{k}": NAN for k in Z_SOURCES})   # filled by PopulationNormaliser
        f["ref_ready"] = float(ready)
        return f

    @staticmethod
    def _longest_run(ts: np.ndarray, mask: np.ndarray, dt: float) -> float:
        best = cur = 0
        for m in mask:
            cur = cur + 1 if m else 0
            best = max(best, cur)
        return best * dt

    # ------------------------------------------------------------------ profile
    def profile(self) -> dict:
        return {"ref": self.ref.to_profile(),
                "norm": ({k: list(v) for k, v in self.norm.to_profile().items()} if self.norm else None)}
