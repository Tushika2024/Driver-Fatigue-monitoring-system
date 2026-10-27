"""The MediaPipe I/O layer cannot download models in CI, so landmarkers are replaced by fakes that
return geometrically exact landmarks. This checks the conversion, pose-hold and timestamp logic."""
import math
from types import SimpleNamespace

import mediapipe as mp
import numpy as np
import pytest

from fatigue import geometry as g
from fatigue.extractor import LandmarkExtractor

W, H = 640, 480


class Pt:
    def __init__(self, x=0.5, y=0.5, z=0.0, visibility=0.99):
        self.x, self.y, self.z, self.visibility = x, y, z, visibility


def face_landmarks(ear=0.30, pitch=20.0):
    lm = [Pt() for _ in range(478)]
    model = g.project_model(pitch, 0, 0, W, H)
    for idx, (px, py) in zip(g.PNP_IDS, model):
        lm[idx] = Pt(px / W, py / H)
    def eye(ids, p0, p3):
        """ids order [p0, top1, top2, p3, bottom2, bottom1]; p0/p3 are the horizontal corners (px)."""
        h = ear * 30.0
        x0, y0 = p0; x3, _ = p3
        pts = [p0, (x0 + (x3 - x0) / 3, y0 - h / 2), (x0 + 2 * (x3 - x0) / 3, y0 - h / 2), p3,
               (x0 + 2 * (x3 - x0) / 3, y0 + h / 2), (x0 + (x3 - x0) / 3, y0 + h / 2)]
        for i, (x, y) in zip(ids, pts):
            lm[i] = Pt(x / W, y / H)
    r0 = tuple(model[2]); eye(g.RIGHT_EYE, r0, (r0[0] + 30, r0[1]))          # 33 .. 133
    l3 = tuple(model[3]); eye(g.LEFT_EYE, (l3[0] - 30, l3[1]), l3)           # 362 .. 263
    return lm


def pose_landmarks():
    lm = [Pt(0.5, 0.9, visibility=0.9) for _ in range(33)]
    lm[g.POSE_NOSE] = Pt(0.5, 0.40)
    lm[g.POSE_L_SHOULDER], lm[g.POSE_R_SHOULDER] = Pt(0.62, 0.60), Pt(0.38, 0.60)
    lm[g.POSE_L_WRIST], lm[g.POSE_R_WRIST] = Pt(0.52, 0.45, visibility=0.9), Pt(0.9, 0.9, visibility=0.9)
    return lm


class FakeFace:
    def __init__(self, found=True): self.found, self.times = found, []
    def detect_for_video(self, img, ms):
        self.times.append(ms)
        if not self.found:
            return SimpleNamespace(face_landmarks=[], face_blendshapes=[])
        cats = [SimpleNamespace(category_name=n, score=s) for n, s in
                (("eyeBlinkLeft", 0.1), ("eyeBlinkRight", 0.2), ("jawOpen", 0.05), ("eyeSquintLeft", 0.3), ("eyeSquintRight", 0.5))]
        return SimpleNamespace(face_landmarks=[face_landmarks()], face_blendshapes=[cats])


class FakePose:
    def __init__(self): self.calls = 0
    def detect_for_video(self, img, ms):
        self.calls += 1
        return SimpleNamespace(pose_landmarks=[pose_landmarks()])


def make(cfg, found=True):
    ex = object.__new__(LandmarkExtractor)
    ex.mp, ex.cfg, ex.face, ex.pose = mp, cfg, FakeFace(found), FakePose()
    ex._n, ex._last_ms, ex._pose_hold = 0, -1, {k: float("nan") for k in ("head_above", "tilt", "shoulder_vis", "wrist_dist", "wrist_vis")}
    return ex


def test_features_from_fake_landmarks(cfg):
    ex = make(cfg)
    f = ex.process(np.full((H, W, 3), 120, np.uint8), 1.0)
    assert f["face_found"] == 1
    assert f["ear"] == pytest.approx(0.30, abs=0.01)
    assert f["pitch"] == pytest.approx(20.0, abs=1.0) and abs(f["yaw"]) < 1.0
    assert f["blink_l"] == pytest.approx(0.1) and f["squint"] == pytest.approx(0.4)
    assert f["head_above"] == pytest.approx(0.2 / 0.24)
    assert f["wrist_dist"] < 0.5 and f["wrist_vis"] == pytest.approx(0.9)
    assert f["brightness"] == pytest.approx(120, abs=1) and f["t"] == 1.0


def test_pose_runs_every_k_frames_and_values_are_held(cfg):
    ex = make(cfg)
    img = np.zeros((H, W, 3), np.uint8)
    rows = [ex.process(img, i / 30) for i in range(10)]
    assert ex.pose.calls == math.ceil(10 / cfg.extraction.pose_every)
    assert all(r["head_above"] == rows[0]["head_above"] for r in rows)       # held between pose runs


def test_timestamps_strictly_increase_even_for_repeated_t(cfg):
    ex = make(cfg)
    img = np.zeros((H, W, 3), np.uint8)
    for _ in range(4):
        ex.process(img, 0.0)
    assert ex.face.times == sorted(set(ex.face.times)) and len(ex.face.times) == 4


def test_no_face_row_is_kept_with_flag(cfg):
    ex = make(cfg, found=False)
    f = ex.process(np.zeros((H, W, 3), np.uint8), 0.5)
    assert f["face_found"] == 0 and math.isnan(f["ear"]) and f["t"] == 0.5
