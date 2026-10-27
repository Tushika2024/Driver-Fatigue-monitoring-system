"""Pure geometric feature functions (no MediaPipe dependency, fully unit-testable).

Camera convention: x right, y down, z away from the camera. Head pose angles:
  pitch > 0  head tilted forward / nodding down
  yaw   > 0  face turned toward the image-right (subject's left)
  roll  > 0  clockwise in the image
A frontal face gives ~0,0,0.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

# MediaPipe 478-point face mesh indices
RIGHT_EYE = [33, 160, 158, 133, 153, 144]   # [outer, top1, top2, inner, bottom2, bottom1]
LEFT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [61, 291, 13, 14]                    # corners, inner upper lip, inner lower lip
PNP_IDS = [1, 152, 33, 263, 61, 291]         # nose, chin, eye outer corners, mouth corners

# Canonical face model in camera convention (mm, nose tip at origin, y down,
# the nose is closest to the camera so the other points sit at +z).
MODEL_3D = np.array(
    [
        [0.0, 0.0, 0.0],        # nose tip
        [0.0, 63.6, 12.5],      # chin
        [-43.3, -32.7, 26.0],   # right eye outer corner (image-left)
        [43.3, -32.7, 26.0],    # left eye outer corner (image-right)
        [-28.9, 28.9, 24.1],    # right mouth corner
        [28.9, 28.9, 24.1],     # left mouth corner
    ],
    dtype=np.float64,
)

# Pose landmarker indices
POSE_NOSE, POSE_L_SHOULDER, POSE_R_SHOULDER = 0, 11, 12
POSE_L_WRIST, POSE_R_WRIST = 15, 16

NAN = float("nan")


def _d(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def eye_aspect_ratio(p: np.ndarray) -> float:
    """6-point EAR. p: (6,2) pixel coords ordered [outer, top1, top2, inner, bot2, bot1]."""
    horiz = _d(p[0], p[3])
    if horiz < 1e-6:
        return NAN
    return (_d(p[1], p[5]) + _d(p[2], p[4])) / (2.0 * horiz)


def mouth_aspect_ratio(p: np.ndarray) -> float:
    """p: (4,2) = [left corner, right corner, upper inner lip, lower inner lip]."""
    horiz = _d(p[0], p[1])
    if horiz < 1e-6:
        return NAN
    return _d(p[2], p[3]) / horiz


def head_pose(img_pts: np.ndarray, w: int, h: int) -> Tuple[float, float, float]:
    """(pitch, yaw, roll) in degrees from 6 face points via solvePnP."""
    img_pts = np.ascontiguousarray(img_pts, dtype=np.float64)
    if not np.all(np.isfinite(img_pts)):
        return NAN, NAN, NAN
    f = float(w)
    cam = np.array([[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1]], dtype=np.float64)
    ok, rvec, tvec = cv2.solvePnP(MODEL_3D, img_pts, cam, np.zeros((4, 1)),
                                  flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok or tvec[2, 0] <= 0:
        return NAN, NAN, NAN
    R, _ = cv2.Rodrigues(rvec)
    # R = Rz(roll) Ry(yaw) Rx(pitch)
    pitch = math.atan2(R[2, 1], R[2, 2])
    yaw = math.atan2(-R[2, 0], math.hypot(R[2, 1], R[2, 2]))
    roll = math.atan2(R[1, 0], R[0, 0])
    return math.degrees(pitch), math.degrees(yaw), math.degrees(roll)


def project_model(pitch: float, yaw: float, roll: float, w: int, h: int,
                  z: float = 600.0) -> np.ndarray:
    """Render the canonical model at a known pose (used by tests and the simulator)."""
    p, y, r = map(math.radians, (pitch, yaw, roll))
    Rx = np.array([[1, 0, 0], [0, math.cos(p), -math.sin(p)], [0, math.sin(p), math.cos(p)]])
    Ry = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
    Rz = np.array([[math.cos(r), -math.sin(r), 0], [math.sin(r), math.cos(r), 0], [0, 0, 1]])
    R = Rz @ Ry @ Rx
    rvec, _ = cv2.Rodrigues(R)
    cam = np.array([[w, 0, w / 2.0], [0, w, h / 2.0], [0, 0, 1]], dtype=np.float64)
    pts, _ = cv2.projectPoints(MODEL_3D, rvec, np.array([0.0, 0.0, z]), cam, np.zeros((4, 1)))
    return pts.reshape(-1, 2)


def posture_features(nose: Sequence[float], ls: Sequence[float], rs: Sequence[float]
                     ) -> Tuple[float, float]:
    """(head_above, tilt), normalised by shoulder width so camera distance cancels.
    head_above: height of the nose above the shoulder line, in shoulder widths.
    tilt: signed shoulder-line tilt in shoulder widths."""
    sw = math.hypot(ls[0] - rs[0], ls[1] - rs[1])
    if sw < 1e-6:
        return NAN, NAN
    mid_y = (ls[1] + rs[1]) / 2.0
    return (mid_y - nose[1]) / sw, (ls[1] - rs[1]) / sw


def wrist_to_face(nose: Sequence[float], ls: Sequence[float], rs: Sequence[float],
                  lw: Sequence[float], rw: Sequence[float], lw_vis: float, rw_vis: float
                  ) -> Tuple[float, float]:
    """(min wrist-to-nose distance in shoulder widths, visibility of that wrist)."""
    sw = math.hypot(ls[0] - rs[0], ls[1] - rs[1])
    if sw < 1e-6:
        return NAN, 0.0
    best, vis = NAN, 0.0
    for wr, v in ((lw, lw_vis), (rw, rw_vis)):
        d = math.hypot(wr[0] - nose[0], wr[1] - nose[1]) / sw
        if math.isnan(best) or d < best:
            best, vis = d, v
    return best, vis


def face_width_fraction(pts_norm: np.ndarray) -> float:
    """Face width as a fraction of the frame width, from the outer eye corners."""
    return float(abs(pts_norm[263][0] - pts_norm[33][0]) * 2.2)


def to_px(lm: np.ndarray, ids: Sequence[int], w: int, h: int) -> np.ndarray:
    return np.stack([lm[ids, 0] * w, lm[ids, 1] * h], axis=1)


def frame_features_from_landmarks(
    face_lm: Optional[np.ndarray], blend: Optional[dict], pose_lm: Optional[np.ndarray],
    pose_vis: Optional[np.ndarray], w: int, h: int,
) -> dict:
    """Convert landmark arrays to one flat feature dict (the single place this is done)."""
    out = {k: NAN for k in (
        "ear_l", "ear_r", "ear", "mar", "blink_l", "blink_r", "squint", "jaw_open",
        "pitch", "yaw", "roll", "head_above", "tilt", "shoulder_vis", "wrist_dist",
        "wrist_vis", "face_w_frac")}
    out["face_found"] = 0
    if face_lm is not None:
        out["face_found"] = 1
        el = eye_aspect_ratio(to_px(face_lm, LEFT_EYE, w, h))
        er = eye_aspect_ratio(to_px(face_lm, RIGHT_EYE, w, h))
        out["ear_l"], out["ear_r"] = el, er
        out["ear"] = float(np.nanmean([el, er])) if not (math.isnan(el) and math.isnan(er)) else NAN
        out["mar"] = mouth_aspect_ratio(to_px(face_lm, MOUTH, w, h))
        out["pitch"], out["yaw"], out["roll"] = head_pose(to_px(face_lm, PNP_IDS, w, h), w, h)
        out["face_w_frac"] = face_width_fraction(face_lm)
        if blend:
            out["blink_l"] = blend.get("eyeBlinkLeft", NAN)
            out["blink_r"] = blend.get("eyeBlinkRight", NAN)
            out["jaw_open"] = blend.get("jawOpen", NAN)
            sq = [blend.get("eyeSquintLeft", NAN), blend.get("eyeSquintRight", NAN)]
            out["squint"] = float(np.nanmean(sq)) if not all(math.isnan(s) for s in sq) else NAN
    if pose_lm is not None:
        vis = pose_vis if pose_vis is not None else np.ones(len(pose_lm))
        nose, ls, rs = pose_lm[POSE_NOSE, :2], pose_lm[POSE_L_SHOULDER, :2], pose_lm[POSE_R_SHOULDER, :2]
        out["shoulder_vis"] = float(min(vis[POSE_L_SHOULDER], vis[POSE_R_SHOULDER]))
        out["head_above"], out["tilt"] = posture_features(nose, ls, rs)
        out["wrist_dist"], out["wrist_vis"] = wrist_to_face(
            nose, ls, rs, pose_lm[POSE_L_WRIST, :2], pose_lm[POSE_R_WRIST, :2],
            float(vis[POSE_L_WRIST]), float(vis[POSE_R_WRIST]))
    return out
