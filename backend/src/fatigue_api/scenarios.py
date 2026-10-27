"""Simulated driver scenarios. They produce *feature rows* (same schema as real extraction), so the
whole downstream pipeline (StreamProcessor -> model -> AlertPolicy -> cabin routing) is the real one.
Everything generated here is SYNTHETIC and is stamped so in logs and in the API."""
from __future__ import annotations

import zlib
from dataclasses import dataclass
from typing import Callable, Dict, List

import numpy as np
import pandas as pd

from fatigue.synthetic import generate_clip, make_subjects

FPS = 30.0


@dataclass(frozen=True)
class Scenario:
    id: str
    title: str
    blurb: str
    duration_s: float
    curve: Callable[[np.ndarray], np.ndarray]          # drowsiness 0..1 over time
    glare: tuple = ()                                    # ((start_s, end_s), ...) eyes unreadable
    away: tuple = ()                                     # ((start_s, end_s), ...) driver not visible


def _ramp(t, a, b, lo, hi):
    return lo + (hi - lo) * np.clip((t - a) / max(b - a, 1e-6), 0, 1)


SCENARIOS: Dict[str, Scenario] = {s.id: s for s in [
    Scenario("steady_alert", "Rested driver", "Alert the whole way. Nothing should fire.", 480,
             lambda t: np.full_like(t, 0.05)),
    Scenario("gradual_fatigue", "Gradual fatigue", "Slow build-up over a long motorway stretch: early signs, then late.", 900,
             lambda t: _ramp(t, 240, 780, 0.05, 0.9)),
    Scenario("sudden_onset", "Sudden onset", "Alert, then fatigue arrives fast, as on a monotonous night road.", 600,
             lambda t: _ramp(t, 150, 330, 0.08, 0.95)),
    Scenario("glare_dropout", "Sun glare / glasses", "Fatigue builds while the eyes become unreadable for a while. "
             "The system must say it is in degraded mode.", 840,
             lambda t: _ramp(t, 200, 700, 0.05, 0.85), glare=((420, 560),)),
    Scenario("driver_leaves_frame", "Driver not visible", "The driver leans out of frame. The system must say it cannot see, not guess.", 540,
             lambda t: _ramp(t, 200, 480, 0.05, 0.6), away=((300, 360),)),
]}


def subject_for(driver_id: str):
    sp = make_subjects(1, seed=zlib.crc32(driver_id.encode()) % 100000)[0]
    sp.subject = driver_id
    return sp


def build_rows(scn: Scenario, driver_id: str, seed: int = 0) -> List[dict]:
    t = np.arange(0, scn.duration_s, 1 / FPS)
    sp = subject_for(driver_id)
    df = generate_clip(sp, "prog", scn.duration_s, FPS, seed=seed, d_override=scn.curve(t))
    df = _mask(df, scn.glare, scn.away)
    return df.to_dict("records")


def _mask(df: pd.DataFrame, glare, away) -> pd.DataFrame:
    df = df.copy()
    for a, b in glare:
        m = (df["t"] >= a) & (df["t"] < b)
        df.loc[m, ["ear_l", "ear_r", "ear", "squint"]] = np.nan
    for a, b in away:
        m = (df["t"] >= a) & (df["t"] < b)
        df.loc[m, "face_found"] = 0
        df.loc[m, ["ear_l", "ear_r", "ear", "mar", "blink_l", "blink_r", "squint", "jaw_open",
                   "pitch", "yaw", "roll", "face_w_frac"]] = np.nan
    return df


PRETRIP_STATES = {
    "normal": "A normal day for this driver",
    "tired": "Driver is tired today",
    "obstructed": "Camera can't see the eyes (glare / mask / dark)",
}


def pretrip_rows(driver_id: str, state: str, seed: int, duration_s: float = 100.0) -> List[dict]:
    sp = subject_for(driver_id)
    kind = "drowsy" if state == "tired" else "alert"
    df = generate_clip(sp, kind, duration_s, FPS, seed=seed)
    if state == "obstructed":
        df = _mask(df, ((0, duration_s),), ())
    return df.to_dict("records")
