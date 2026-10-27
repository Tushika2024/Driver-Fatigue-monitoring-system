"""Synthetic driver generator.

Purpose: let the whole pipeline (windows -> LOSO evaluation -> live replay -> reports)
run and be tested with no dataset and no webcam. It emits the exact features.csv schema.
It is NOT evidence about real drivers: results on synthetic data only show the code works.
"""
from __future__ import annotations

import zlib
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

from .schema import FRAME_COLUMNS


@dataclass
class SubjectProfile:
    subject: str
    ear_open: float
    ear_noise: float
    glasses: bool
    facial_hair: bool
    blink_rate: float        # per minute when alert
    pitch0: float
    yaw0: float
    head_above0: float
    susceptibility: float    # how strongly drowsiness shows
    reopen_scale: float = 1.0   # person-specific blink speed (makes population thresholds unfair)
    pose_present: bool = True


def make_subjects(n: int, seed: int = 0) -> List[SubjectProfile]:
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        glasses = bool(rng.random() < 0.35)
        out.append(SubjectProfile(
            subject=f"S{i + 1:02d}",
            ear_open=float(np.clip(rng.normal(0.30, 0.04), 0.21, 0.40)),
            ear_noise=float(rng.uniform(0.006, 0.012) * (1.8 if glasses else 1.0)),
            glasses=glasses, facial_hair=bool(rng.random() < 0.4),
            blink_rate=float(rng.uniform(16, 26)),
            pitch0=float(rng.normal(0, 5)), yaw0=float(rng.normal(0, 5)),
            head_above0=float(rng.normal(0.85, 0.08)),
            susceptibility=float(rng.uniform(0.75, 1.25)),
            reopen_scale=float(np.clip(np.exp(rng.normal(0, 0.3)), 0.6, 1.7)),
        ))
    return out


def _shape(t: np.ndarray, s: float, c: float, h: float, r: float, depth: float) -> np.ndarray:
    x = np.zeros_like(t)
    a = (t >= s) & (t < s + c); x[a] = depth * (t[a] - s) / c
    b = (t >= s + c) & (t < s + c + h); x[b] = depth
    d = (t >= s + c + h) & (t < s + c + h + r); x[d] = depth * (1 - (t[d] - s - c - h) / r)
    return x


def drowsiness_curve(kind: str, t: np.ndarray, rng) -> np.ndarray:
    if kind == "alert":
        return np.full_like(t, 0.03)
    if kind == "drowsy":
        wob = 0.08 * np.sin(2 * np.pi * t / rng.uniform(90, 160) + rng.uniform(0, 6))
        return np.clip(0.70 + wob, 0, 1)
    if kind == "prog":           # alert start -> drowsy end
        return np.clip((t / t[-1]) ** 1.3 + 0.03, 0, 1)
    raise ValueError(kind)


def generate_clip(sp: SubjectProfile, kind: str, duration_s: float = 300.0, fps: float = 30.0,
                  seed: int = 0, label: Optional[str] = None, d_override: Optional[np.ndarray] = None
                  ) -> pd.DataFrame:
    rng = np.random.default_rng(zlib.crc32(f"{sp.subject}|{kind}|{seed}".encode()))
    n = int(duration_s * fps)
    t = np.arange(n) / fps
    d = d_override if d_override is not None else drowsiness_curve(kind, t, rng)
    d = np.clip(d * sp.susceptibility, 0, 1)

    # --- eyes -----------------------------------------------------------------
    env = np.zeros(n)
    pos = rng.uniform(0.5, 2.0)
    while pos < duration_s - 3:
        di = d[min(int(pos * fps), n - 1)]
        rate = sp.blink_rate * (1 + 0.25 * di) / 60.0
        pos += rng.exponential(1 / rate)
        close = rng.normal(0.075, 0.01)
        reopen = max(0.06, rng.normal(0.13 * sp.reopen_scale * (1 + 1.6 * di), 0.02))
        hold = rng.uniform(0.02, 0.06) + 0.25 * di * rng.random()
        env = np.maximum(env, _shape(t, pos, close, hold, reopen, rng.uniform(0.82, 0.97)))
    # long closures / microsleeps grow with drowsiness
    pos = rng.uniform(5, 20)
    while pos < duration_s - 5:
        di = d[min(int(pos * fps), n - 1)]
        gap = 60.0 / max(0.4, 7.0 * di ** 1.4 + 0.4)
        pos += rng.exponential(gap)
        dur = rng.uniform(0.7, 2.2) if di > 0.25 else rng.uniform(0.3, 0.6)
        if rng.random() < di + 0.05:
            env = np.maximum(env, _shape(t, pos, 0.2, dur, 0.4 + 0.3 * di, 0.95))
    droop = 0.10 * d
    openness = np.clip(1.0 - droop - env * (1 - droop), 0.02, 1.2)
    ear_noise = rng.normal(0, sp.ear_noise, n)
    asym = rng.normal(0, 0.004 + 0.03 * d ** 2, n)
    ear_l = sp.ear_open * openness + ear_noise + asym
    ear_r = sp.ear_open * openness + rng.normal(0, sp.ear_noise, n) - asym

    ear_l, ear_r = np.maximum(ear_l, 0.0), np.maximum(ear_r, 0.0)      # EAR is a distance ratio: never negative
    # --- compensatory behaviours (peak mid-drowsiness) ---------------------------
    wrist = np.full(n, 2.0)
    squint = np.clip(rng.normal(0.05, 0.02, n), 0, 1)
    yaw = sp.yaw0 + rng.normal(0, 0.7, n)
    pitch = sp.pitch0 + rng.normal(0, 0.8, n)
    ha = sp.head_above0 + rng.normal(0, 0.01, n)
    tilt = rng.normal(0, 0.01, n)
    comp_rate = lambda di: 0.5 + 9.0 * di * (1 - 0.6 * di)       # per minute
    pos = rng.uniform(3, 10)
    while pos < duration_s - 4:
        di = d[min(int(pos * fps), n - 1)]
        pos += rng.exponential(60.0 / comp_rate(di))
        i0 = int(pos * fps); kind_e = rng.choice(["rub", "shake", "shift", "hard"], p=[.3, .25, .25, .2])
        if kind_e == "rub":
            L = int(rng.uniform(0.8, 2.0) * fps); sl = slice(i0, min(n, i0 + L))
            wrist[sl] = rng.uniform(0.2, 0.45); ear_l[sl] *= 0.7; ear_r[sl] *= 0.7
        elif kind_e == "shake":
            L = int(0.8 * fps); sl = np.arange(i0, min(n, i0 + L))
            yaw[sl] += 14 * np.sin(2 * np.pi * 3.0 * (sl - i0) / fps)
        elif kind_e == "shift":
            L = int(rng.uniform(1.5, 3) * fps); sl = slice(i0, min(n, i0 + L))
            ha[sl] += 0.25; tilt[sl] += 0.2
        else:
            c_t = rng.uniform(0.15, 0.3)
            e2 = _shape(t, pos, 0.08, c_t, 0.2, 0.98)
            openness_h = np.clip(1.0 - droop - e2 * (1 - droop), 0.02, 1.2)
            ear_l = np.minimum(ear_l, sp.ear_open * openness_h); ear_r = np.minimum(ear_r, sp.ear_open * openness_h)
            squint[e2 > 0.5] = rng.uniform(0.6, 0.9)
    # --- head nods and yawns (late signals) ------------------------------------
    mar = np.clip(rng.normal(0.08, 0.02, n), 0, 1)
    pos = rng.uniform(10, 30)
    while pos < duration_s - 6:
        di = d[min(int(pos * fps), n - 1)]
        pos += rng.exponential(60.0 / max(0.15, 3.5 * di ** 1.5 + 0.15))
        if rng.random() < di + 0.02:
            i0 = int(pos * fps); L = int(rng.uniform(1.2, 2.6) * fps)
            sl = np.arange(i0, min(n, i0 + L))
            pitch[sl] += rng.uniform(14, 26) * np.sin(np.pi * (sl - i0) / max(L, 1)) ** 0.7
    pos = rng.uniform(15, 40)
    while pos < duration_s - 8:
        di = d[min(int(pos * fps), n - 1)]
        pos += rng.exponential(60.0 / max(0.1, 2.0 * di + 0.1))
        if rng.random() < di + 0.02:
            i0 = int(pos * fps); L = int(rng.uniform(2.5, 5) * fps)
            sl = np.arange(i0, min(n, i0 + L))
            mar[sl] = np.clip(0.75 + rng.normal(0, 0.05, len(sl)), 0, 1)
    ha -= 0.10 * d                                         # slouch
    pitch += 3.0 * d

    # --- dropouts (glasses glare, face lost) --------------------------------------
    face = np.ones(n)
    for _ in range(int(duration_s / 60 * rng.uniform(0, 1.5))):
        i0 = int(rng.uniform(0, n - 60)); face[i0:i0 + int(rng.uniform(0.3, 2.0) * fps)] = 0
    if sp.glasses:
        for _ in range(int(duration_s / 20)):
            i0 = int(rng.uniform(0, n - 60)); L = int(rng.uniform(0.5, 3.0) * fps)
            ear_l[i0:i0 + L] = np.nan; ear_r[i0:i0 + L] = np.nan     # glare: eyes unreadable
    ear = np.where(np.isnan(ear_l), ear_r, np.where(np.isnan(ear_r), ear_l, (ear_l + ear_r) / 2))

    df = pd.DataFrame({
        "subject": sp.subject, "label": label if label is not None else
        {"alert": "0", "drowsy": "10", "prog": "prog"}[kind],
        "video": f"{kind}.mp4", "frame_idx": np.arange(n), "t": t, "face_found": face,
        "ear_l": ear_l, "ear_r": ear_r, "ear": ear, "mar": mar,
        "blink_l": np.clip(1 - openness, 0, 1), "blink_r": np.clip(1 - openness, 0, 1),
        "squint": squint, "jaw_open": np.clip(mar, 0, 1),
        "pitch": pitch, "yaw": yaw, "roll": rng.normal(0, 1, n),
        "head_above": ha, "tilt": tilt, "shoulder_vis": 0.95, "wrist_dist": wrist, "wrist_vis": 0.9,
        "brightness": rng.normal(130, 4, n), "face_w_frac": rng.normal(0.33, 0.005, n),
    })
    lost = face == 0
    for c in ["ear_l", "ear_r", "ear", "mar", "blink_l", "blink_r", "squint", "jaw_open",
              "pitch", "yaw", "roll", "face_w_frac"]:
        df.loc[lost, c] = np.nan
    return df[FRAME_COLUMNS]


def make_dataset(n_subjects: int = 10, duration_s: float = 300.0, fps: float = 30.0, seed: int = 0,
                 with_prog: bool = True):
    """Returns (frames_df, subjects_df)."""
    subs = make_subjects(n_subjects, seed)
    frames = []
    for sp in subs:
        kinds = ["alert", "drowsy"] + (["prog"] if with_prog else [])
        for k in kinds:
            frames.append(generate_clip(sp, k, duration_s, fps, seed))
    meta = pd.DataFrame([{"subject": s.subject, "glasses": int(s.glasses),
                          "facial_hair": int(s.facial_hair)} for s in subs])
    return pd.concat(frames, ignore_index=True), meta
