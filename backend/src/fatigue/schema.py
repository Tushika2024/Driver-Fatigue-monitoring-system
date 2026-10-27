"""Data contracts shared by every module."""
from __future__ import annotations

# --- per-frame table (features.csv) -----------------------------------------
FRAME_COLUMNS = [
    "subject", "label", "video", "frame_idx", "t", "face_found",
    "ear_l", "ear_r", "ear", "mar", "blink_l", "blink_r", "squint", "jaw_open",
    "pitch", "yaw", "roll",
    "head_above", "tilt", "shoulder_vis", "wrist_dist", "wrist_vis",
    "brightness", "face_w_frac",
]

# --- window feature groups ---------------------------------------------------
# Raw (reference-scaled) late signals, z-scored early signals, behaviours, etc.
GROUPS = {
    "eyes_late": ["perclos", "longest_closure_s", "ear_rel_mean"],
    "eyes_early": ["reopen_med_s", "closing_med_s", "asym", "blink_rate_pm", "ear_diverge"],
    "behaviour": ["compensatory_count", "n_eye_rub", "n_head_shake", "n_posture_shift", "n_hard_blink"],
    "mouth": ["yawn_frac"],
    "head": ["head_drop_frac", "pitch_rel_mean", "pitch_std", "yaw_std"],
    "posture": ["head_above_rel", "head_above_std", "tilt_std"],
}
# Features that get z-scored against a reference (the "early" family).
Z_SOURCES = ["reopen_med_s", "closing_med_s", "asym", "blink_rate_pm", "ear_diverge"]

# Features needing valid eye landmarks (unavailable in degraded mode).
EYE_GROUPS = ["eyes_late", "eyes_early"]
# 'n_hard_blink' also needs eyes; the others in 'behaviour' do not.
EYE_ONLY_FEATURES = {"n_hard_blink", "compensatory_count"}


def cols(*groups: str) -> list[str]:
    out: list[str] = []
    for g in groups:
        out.extend(GROUPS[g])
    return out


FULL_FEATURES = cols("eyes_late", "eyes_early", "behaviour", "mouth", "head", "posture")
EARLY_FEATURES = cols("eyes_early", "behaviour")
LATE_FEATURES = cols("eyes_late", "head")
# Degraded mode: eyes unreliable -> head, posture, mouth and the hand/head behaviours.
DEGRADED_FEATURES = (cols("head", "posture", "mouth")
                     + ["n_eye_rub", "n_head_shake", "n_posture_shift"])

# Ablation variants -> feature columns
ABLATIONS = {
    "full": FULL_FEATURES,
    "no_early_signals": cols("eyes_late", "mouth", "head", "posture"),
    "perclos_only": ["perclos"],
    "blink_shape_plus_perclos": ["perclos", "reopen_med_s", "closing_med_s", "asym"],
    "eyes_only": cols("eyes_late", "eyes_early"),
    "eyes_head_posture": cols("eyes_late", "eyes_early", "head", "posture"),
}

LEVEL_NAMES = {0: "NONE", 1: "MILD", 2: "MODERATE", 3: "SEVERE", 4: "CRITICAL"}


def resolve(columns: list[str], use_z: bool) -> list[str]:
    """Swap raw early-signal columns for their z-scored counterparts when use_z is set."""
    if not use_z:
        return list(columns)
    return [f"z_{c}" if c in Z_SOURCES else c for c in columns]
