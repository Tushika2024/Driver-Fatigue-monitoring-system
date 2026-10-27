import math

import numpy as np
import pytest

from fatigue import geometry as g


def eye(open_h):
    # outer, top1, top2, inner, bottom2, bottom1 ; width 30 px
    return np.array([[0, 0], [10, -open_h / 2], [20, -open_h / 2], [30, 0], [20, open_h / 2], [10, open_h / 2]], float)


def test_ear_scales_with_opening():
    assert g.eye_aspect_ratio(eye(10)) == pytest.approx(10 / 30)
    assert g.eye_aspect_ratio(eye(1)) < g.eye_aspect_ratio(eye(10))
    assert g.eye_aspect_ratio(eye(0)) == pytest.approx(0.0)


def test_ear_degenerate_returns_nan():
    p = eye(10); p[3] = p[0]
    assert math.isnan(g.eye_aspect_ratio(p))


def test_mar():
    p = np.array([[0, 0], [40, 0], [20, -10], [20, 10]], float)
    assert g.mouth_aspect_ratio(p) == pytest.approx(0.5)


@pytest.mark.parametrize("pose", [(0, 0, 0), (20, 0, 0), (-15, 0, 0), (0, 25, 0), (0, -30, 0), (0, 0, 15), (18, -12, 6)])
def test_head_pose_round_trip(pose):
    pts = g.project_model(*pose, 640, 480)
    got = g.head_pose(pts, 640, 480)
    assert got == pytest.approx(pose, abs=0.5)


def test_head_pose_sign_convention_nod_is_positive_pitch():
    pitch, yaw, _ = g.head_pose(g.project_model(25, 0, 0, 640, 480), 640, 480)
    assert pitch > 20 and abs(yaw) < 1


def test_head_pose_nan_input():
    p = g.project_model(0, 0, 0, 640, 480); p[2] = np.nan
    assert all(math.isnan(v) for v in g.head_pose(p, 640, 480))


def test_posture_invariant_to_scale():
    a = g.posture_features((0.5, 0.4), (0.6, 0.6), (0.4, 0.6))
    b = g.posture_features((0.5, 0.2), (0.7, 0.6), (0.3, 0.6))
    assert a[0] == pytest.approx(0.2 / 0.2) and b[0] == pytest.approx(0.4 / 0.4)


def test_wrist_to_face_picks_nearest():
    d, vis = g.wrist_to_face((0.5, 0.4), (0.6, 0.6), (0.4, 0.6), (0.9, 0.9), (0.52, 0.42), 0.9, 0.7)
    assert d < 0.2 and vis == 0.7


def test_frame_features_no_face_no_pose():
    out = g.frame_features_from_landmarks(None, None, None, None, 640, 480)
    assert out["face_found"] == 0 and math.isnan(out["ear"]) and math.isnan(out["pitch"])
