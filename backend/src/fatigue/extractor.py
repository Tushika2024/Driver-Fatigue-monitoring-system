"""MediaPipe Face + Pose wrapper -> per-frame feature dict (schema.FRAME_COLUMNS).

All maths lives in geometry.py; this file only does I/O with MediaPipe. Pose runs every
`pose_every` frames (CPU budget) and its outputs are held in between.
"""
from __future__ import annotations

import os
import urllib.request
from typing import Dict, Optional

import cv2
import numpy as np

from .config import Config
from .geometry import frame_features_from_landmarks

FACE_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
                  "face_landmarker/float16/1/face_landmarker.task")
POSE_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
                  "pose_landmarker_lite/float16/1/pose_landmarker_lite.task")
_POSE_KEYS = ("head_above", "tilt", "shoulder_vis", "wrist_dist", "wrist_vis")


def ensure_model(url: str, path: str) -> str:
    if os.path.exists(path):
        return path
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".part"
    try:
        urllib.request.urlretrieve(url, tmp)
        os.replace(tmp, path)
    except Exception as e:
        raise RuntimeError(f"Could not download {url} ({e}). Download it manually and place it at {path}.") from e
    return path


class LandmarkExtractor:
    def __init__(self, cfg: Config, models_dir: Optional[str] = None):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
        self.mp, self.cfg = mp, cfg
        d = models_dir or cfg.extraction.models_dir
        face_path = ensure_model(FACE_MODEL_URL, os.path.join(d, "face_landmarker.task"))
        VRM = vision.RunningMode.VIDEO
        self.face = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=face_path), running_mode=VRM,
            output_face_blendshapes=True, num_faces=1))
        self.pose = None
        if cfg.extraction.use_pose:
            pose_path = ensure_model(POSE_MODEL_URL, os.path.join(d, "pose_landmarker_lite.task"))
            self.pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=pose_path), running_mode=VRM, num_poses=1))
        self._n = 0
        self._last_ms = -1
        self._pose_hold: Dict[str, float] = {k: float("nan") for k in _POSE_KEYS}

    def close(self) -> None:
        for m in (self.face, self.pose):
            if m is not None:
                m.close()

    def process(self, bgr: np.ndarray, t: float) -> Dict[str, float]:
        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
        ms = max(int(t * 1000), self._last_ms + 1)      # MediaPipe needs strictly increasing ms
        self._last_ms = ms
        fr = self.face.detect_for_video(img, ms)
        face_lm = blend = None
        if fr.face_landmarks:
            face_lm = np.array([[p.x, p.y, p.z] for p in fr.face_landmarks[0]], dtype=np.float64)
            if fr.face_blendshapes:
                blend = {c.category_name: c.score for c in fr.face_blendshapes[0]}
        pose_lm = vis = None
        if self.pose is not None and self._n % max(1, self.cfg.extraction.pose_every) == 0:
            pr = self.pose.detect_for_video(img, ms)
            if pr.pose_landmarks:
                pl = pr.pose_landmarks[0]
                pose_lm = np.array([[p.x, p.y, p.z] for p in pl], dtype=np.float64)
                vis = np.array([getattr(p, "visibility", 1.0) or 0.0 for p in pl], dtype=np.float64)
            self._pose_fresh = True
        else:
            self._pose_fresh = False
        out = frame_features_from_landmarks(face_lm, blend, pose_lm, vis, w, h)
        if self._pose_fresh:
            self._pose_hold = {k: out[k] for k in _POSE_KEYS}
        else:
            out.update(self._pose_hold)
        out["brightness"] = float(cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (32, 18)).mean())
        out["t"] = float(t)
        self._n += 1
        return out
