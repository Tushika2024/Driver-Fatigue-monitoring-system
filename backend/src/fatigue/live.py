"""Live monitoring session: frames -> features -> windows -> scores -> alert policy -> cabin routing -> sinks.

The same StreamProcessor and AlertPolicy used in evaluation run here, so what was measured is what ships.
"""
from __future__ import annotations

import csv
import math
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable, List, Optional

import numpy as np
import pandas as pd

from .alerts import (MILD, NONE, SEVERE, AlertPolicy, Scores, raw_level)
from .cabin import CABIN, Action, VehicleProfile, route
from .extractor import LandmarkExtractor
from .models import FusionModel
from .report import LOG_COLUMNS, write_meta
from .sinks import SinkRouter
from .sources import Frame
from .stream import StreamProcessor

BLIND_NOTICE_S = 10.0


@dataclass
class PerfStats:
    lat_ms: List[float] = field(default_factory=list)
    t_start: float = field(default_factory=time.perf_counter)

    def add(self, dt: float) -> None:
        self.lat_ms.append(dt * 1000.0)

    def summary(self) -> dict:
        if not self.lat_ms:
            return {}
        a = np.array(self.lat_ms)
        wall = time.perf_counter() - self.t_start
        return {"frames": len(a), "fps": len(a) / wall if wall > 0 else math.nan,
                "latency_ms_p50": float(np.percentile(a, 50)), "latency_ms_p95": float(np.percentile(a, 95)),
                "latency_ms_max": float(a.max())}


@dataclass
class Update:
    t: float
    level: int
    mode: str
    score: float
    reason: str
    actions: List[Action]
    early: float = math.nan      # early-stream probability (for display)
    late: float = math.nan       # late-stream probability (for display)


class LiveSession:
    def __init__(self, model: FusionModel, profile: VehicleProfile, sinks: Optional[SinkRouter] = None,
                 driver_id: str = "driver", vehicle_id: str = "vehicle", route_name: str = "route",
                 log_path: Optional[str] = None, extractor: Optional[LandmarkExtractor] = None,
                 start: Optional[datetime] = None, synthetic: bool = False):
        self.model, self.cfg, self.profile = model, model.cfg, profile
        self.sinks = sinks or SinkRouter.default()
        self.driver_id, self.vehicle_id, self.route_name = driver_id, vehicle_id, route_name
        self.extractor, self.start = extractor, start or datetime.now()
        self.sp = StreamProcessor(self.cfg)
        self.policy = AlertPolicy(self.cfg.alerts, critical_enabled=profile.critical_enabled)
        self.perf = PerfStats()
        self.updates: List[Update] = []
        self._last_log = -1e9
        self._blind_since: Optional[float] = None
        self._blind_notified = False
        self._acked_flag = 0
        self._last_reason = ""
        self._log_fh = None
        self.log_path = log_path
        if log_path:
            os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
            self._log_fh = open(log_path, "w", newline="")
            self._log = csv.writer(self._log_fh)
            self._log.writerow(LOG_COLUMNS)
            write_meta(log_path, synthetic=synthetic, driver_id=driver_id, vehicle_id=vehicle_id,
                       route=route_name, start=self.start.isoformat(), profile=profile.name)

    # ------------------------------------------------------------------ input
    def acknowledge(self, t: float) -> None:
        self.policy.acknowledge(t)
        self._acked_flag = 1

    def process(self, frame: Frame) -> Optional[Update]:
        t0 = time.perf_counter()
        feats = frame.features
        if feats is None:
            if self.extractor is None:
                raise RuntimeError("image frame received but no extractor configured")
            feats = self.extractor.process(frame.image, frame.t)
        w = self.sp.push(feats)
        self.perf.add(time.perf_counter() - t0)
        return self._on_window(w, frame.t) if w is not None else None

    # ------------------------------------------------------------------ per window
    def _on_window(self, w: dict, t: float) -> Update:
        row = pd.DataFrame([w])
        sc = self.model.score(row).iloc[0]
        mode = str(sc["mode"])
        warm = w.get("ref_ready", 1.0) < 1.0
        scores = Scores(float(sc["p"]), float(sc["pe"]), float(sc["pl"]), float(sc["ph"]), mode)
        actions: List[Action] = []
        reason = self._last_reason
        if warm:
            level, shown_mode, raw = NONE, "warmup", NONE
        else:
            raw = raw_level(scores, self.model.th)
            prev = self.policy.level
            dec = self.policy.update(t, raw, mode)
            level, shown_mode = dec.level, mode
            if level > NONE and (level > prev or dec.changed):
                reason = self.model.reason(row.iloc[0], mode)
                self._last_reason = reason
                actions += route(level, reason, self.profile)
            elif level >= SEVERE and self.policy.should_repeat(t):
                actions += [a for a in route(level, self._last_reason, self.profile) if a.repeat_until_ack]
        # driver-not-visible notice (never silently blind)
        if mode == "blind" and not warm:
            self._blind_since = self._blind_since if self._blind_since is not None else t
            if t - self._blind_since >= BLIND_NOTICE_S and not self._blind_notified:
                actions.append(Action(MILD, CABIN, "voice", "normal", "Driver not visible to the camera."))
                self._blind_notified = True
        else:
            self._blind_since, self._blind_notified = None, False
        if actions:
            self.sinks.emit_all(actions)
        score = float(sc["p_used"]) if math.isfinite(float(sc["p_used"])) else math.nan
        u = Update(t, level, shown_mode, score, reason if level > NONE else "", actions, scores.pe, scores.pl)
        self.updates.append(u)
        if self._log_fh and t - self._last_log >= self.cfg.alerts.log_every_s - 1e-6:
            self._last_log = t
            ts = (self.start + timedelta(seconds=t)).isoformat(timespec="seconds")
            self._log.writerow([round(t, 2), ts, self.driver_id, self.vehicle_id, self.route_name,
                                "" if math.isnan(score) else round(score, 4),
                                "" if math.isnan(scores.pe) else round(scores.pe, 4),
                                "" if math.isnan(scores.pl) else round(scores.pl, 4),
                                level, shown_mode, 1 if self._acked_flag else "", u.reason])
            self._log_fh.flush()
            self._acked_flag = 0
        return u

    # ------------------------------------------------------------------ loop
    def run(self, source, max_seconds: Optional[float] = None,
            on_update: Optional[Callable[[Update], None]] = None,
            ack_when: Optional[Callable[[Update], bool]] = None, show: bool = False) -> List[Update]:
        try:
            for frame in source:
                u = self.process(frame)
                if show and frame.image is not None:
                    import cv2
                    cv2.putText(frame.image, f"L{self.policy.level}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.imshow("fatigue monitor (a=acknowledge, q=quit)", frame.image)
                    k = cv2.waitKey(1) & 0xFF
                    if k == ord("q"):
                        break
                    if k == ord("a"):
                        self.acknowledge(frame.t)
                if u is not None:
                    if on_update:
                        on_update(u)
                    if ack_when and u.level >= SEVERE and ack_when(u):
                        self.acknowledge(u.t)
                if max_seconds is not None and frame.t >= max_seconds:
                    break
        finally:
            self.close()
        return self.updates

    def close(self) -> None:
        if self._log_fh:
            self._log_fh.close()
            self._log_fh = None
        if self.extractor:
            self.extractor.close()
