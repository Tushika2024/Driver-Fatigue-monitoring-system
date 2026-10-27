"""API tests: REST contracts, the live and pre-trip WebSockets (simulated feeds), camera path with a fake extractor."""
import json

import numpy as np
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient   # noqa: E402

from fatigue_api import create_app          # noqa: E402
from fatigue_api.runtime import Runtime     # noqa: E402


@pytest.fixture(scope="module")
def client(model, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("var")
    rt = Runtime(var_dir=tmp, train_on_start=False)
    rt.set_model(model)
    app = create_app(rt)
    with TestClient(app) as c:
        c.rt = rt
        yield c


def test_health_and_config(client):
    h = client.get("/api/health").json()
    assert h["ok"] and h["model"]["state"] == "ready" and h["model"]["synthetic"] is True
    c = client.get("/api/config").json()
    assert c["window_s"] > 0 and "capture_s" in c["pretrip"]
    assert 0 < c["closed_frac"] < 1 and c["reference_min_s"] > 0
    assert c["pretrip"]["caution_at"] < c["pretrip"]["not_fit_at"]


def test_profiles_matrix_matches_cabin_invariants(client):
    ps = client.get("/api/profiles").json()
    names = {p["name"] for p in ps}
    assert {"solo_goods_truck", "ambulance_patient", "school_bus"} <= names
    for p in ps:
        sev = p["matrix"]["SEVERE"]
        assert sev, f"SEVERE must never be suppressed ({p['name']})"       # redirected, not dropped
        if p["overhearing_harmful"]:
            assert sev[0]["channel"] != "cabin" or sev[0]["announce"] or any(a["channel"] in ("crew", "dispatch") for a in sev) \
                or sev[-1]["announce"]
    amb = next(p for p in ps if p["name"] == "ambulance_patient")
    assert any(a["channel"] == "crew" for a in amb["matrix"]["SEVERE"])
    assert all(a["volume"] != "loud" for a in amb["matrix"]["SEVERE"])


def test_route_custom_flags_and_validation(client):
    r = client.post("/api/cabin/route", json={"level": 3, "flags": {"overhearing_harmful": True}}).json()
    assert r["actions"] and any(a["announce"] for a in r["actions"])        # nobody to tell: falls back and says so
    assert client.post("/api/cabin/route", json={"level": 9}).status_code == 422
    assert client.post("/api/cabin/route", json={"level": 2, "flags": {"bogus": True}}).status_code == 422
    assert client.post("/api/cabin/route", json={"level": 2, "profile": "nope"}).status_code == 404


def test_scenarios_listed(client):
    s = client.get("/api/scenarios").json()
    assert {x["id"] for x in s["live"]} >= {"steady_alert", "gradual_fatigue", "glare_dropout"}
    assert {x["id"] for x in s["pretrip"]} == {"normal", "tired", "obstructed"}


def test_live_ws_simulated_end_to_end(client):
    with client.websocket_connect("/ws/live") as ws:
        ws.send_json({"source": "simulated", "scenario": "sudden_onset", "profile": "ambulance_patient",
                      "driver_id": "T01", "speed": 120})
        first = ws.receive_json()
        assert first["type"] == "started" and first["synthetic"] is True and first["profile"]["name"] == "ambulance_patient"
        levels, modes, acted, sid, ticks = [], set(), [], first["session_id"], 0
        while True:
            m = ws.receive_json()
            if m["type"] == "tick":
                ticks += 1
                assert 0 <= m["progress"] <= 1
            elif m["type"] == "update":
                levels.append(m["level"]); modes.add(m["mode"]); acted += m["actions"]
                if m["level"] >= 3 and not any(x["type"] == "acked" for x in []):
                    ws.send_json({"type": "ack"})
            elif m["type"] == "ended":
                break
            elif m["type"] == "error":
                pytest.fail(m["message"])
    assert ticks > 100 and max(levels) >= 2, "fatigue should be detected in the sudden-onset scenario"
    assert "warmup" in modes
    assert all(a["volume"] != "loud" for a in acted if a["level"] < 4), "ambulance with a patient must never use a loud alarm"
    assert any(a["channel"] in ("crew", "dispatch") for a in acted if a["level"] >= 3)
    rep = client.get(f"/api/sessions/{sid}/report").json()
    assert rep["synthetic"] is True and rep["metrics"]["empty"] is False and rep["series"]


def test_live_steady_alert_stays_quiet(client):
    with client.websocket_connect("/ws/live") as ws:
        ws.send_json({"source": "simulated", "scenario": "steady_alert", "profile": "solo_goods_truck",
                      "driver_id": "T02", "speed": 120})
        lv = []
        while True:
            m = ws.receive_json()
            if m["type"] == "update":
                lv.append(m["level"])
            if m["type"] in ("ended", "error"):
                assert m["type"] == "ended"
                break
    assert max(lv) <= 1, "a rested simulated driver should not reach MODERATE"


def test_live_glare_reports_degraded_not_silence(client):
    with client.websocket_connect("/ws/live") as ws:
        ws.send_json({"source": "simulated", "scenario": "glare_dropout", "profile": "solo_goods_truck",
                      "driver_id": "T03", "speed": 120})
        modes = set()
        while True:
            m = ws.receive_json()
            if m["type"] == "update":
                modes.add(m["mode"])
            if m["type"] in ("ended", "error"):
                break
    assert "degraded" in modes


def test_live_ws_validation(client):
    with client.websocket_connect("/ws/live") as ws:
        ws.send_json({"source": "simulated", "scenario": "nope"})
        assert ws.receive_json()["type"] == "error"
    with client.websocket_connect("/ws/live") as ws:
        ws.send_json({"source": "simulated", "driver_id": "bad id!"})
        assert ws.receive_json()["type"] == "error"
    with client.websocket_connect("/ws/live") as ws:
        ws.send_json({"source": "simulated", "profile": "nope"})
        assert "unknown vehicle profile" in ws.receive_json()["message"]


def test_pretrip_normal_tired_and_obstructed(client):
    def run(state):
        with client.websocket_connect("/ws/pretrip") as ws:
            ws.send_json({"driver_id": "P01", "state": state, "speed": 60})
            assert ws.receive_json()["type"] == "started"
            phases = set()
            while True:
                m = ws.receive_json()
                if m["type"] == "progress":
                    phases.add(m["phase"])
                elif m["type"] == "result":
                    return m["result"], phases
                elif m["type"] == "error":
                    pytest.fail(m["message"])
    normal, ph = run("normal")
    assert normal["advisory"] is True and normal["verdict"] in ("FIT", "CAUTION")
    assert ph == {"settling", "measuring"}
    tired, _ = run("tired")
    assert tired["verdict"] in ("CAUTION", "NOT_FIT")
    assert (tired["index"] or 0) > (normal["index"] or 0)
    n_before = len(client.get("/api/pretrip/history/P01").json()["sessions"])
    obs, _ = run("obstructed")
    assert obs["verdict"] == "INCONCLUSIVE" and obs["stored"] is False
    assert len(client.get("/api/pretrip/history/P01").json()["sessions"]) == n_before    # inconclusive is never stored
    assert client.delete("/api/pretrip/history/P01").json()["sessions"] == []


def test_pretrip_rejects_camera_and_bad_ids(client):
    with client.websocket_connect("/ws/pretrip") as ws:
        ws.send_json({"driver_id": "P02", "state": "normal", "source": "camera"})
        assert ws.receive_json()["type"] == "error"
    assert client.get("/api/pretrip/history/bad id").status_code in (404, 422)


def test_demo_shift_report_and_depot(client):
    r = client.post("/api/sessions/demo", json={"scenario": "sudden_onset", "driver_id": "DEMO9"})
    assert r.status_code == 200
    sid = r.json()["id"]
    assert any(s["id"] == sid and s["synthetic"] for s in client.get("/api/sessions").json())
    assert client.get("/api/sessions/nope/report").status_code == 404
    d = client.get("/api/depot?source=synthetic").json()
    assert d["synthetic"] is True and d["k"] >= 2 and d["cells"]
    for c in d["cells"]:
        assert "driver" not in c, "the depot view must never expose individual drivers"
        if c["suppressed"]:
            assert c["pct_time_alert"] is None and c["events_per_100h"] is None
    assert client.get("/api/depot?source=bogus").status_code == 422
    assert client.get("/api/depot?source=sessions").json()["n_logs"] >= 1


def test_evaluation_endpoint_reads_results(client, tmp_path):
    ev = client.get("/api/evaluation").json()
    assert ev["synthetic"] is True
    if ev["available"]:
        assert "main_comparison" in ev["tables"]
    client.rt.results_dir = tmp_path
    empty = client.get("/api/evaluation").json()
    assert empty["available"] is False


def test_camera_path_with_fake_extractor(model, tmp_path):
    """The browser-webcam path: JPEG bytes -> extractor -> session. The real MediaPipe path is NOT exercised here."""
    cv2 = pytest.importorskip("cv2")

    class FakeExtractor:
        def process(self, img, t):
            return {"t": t, "face_found": 1, "ear_l": 0.30, "ear_r": 0.30, "ear": 0.30, "mar": 0.1, "squint": 0.0,
                    "pitch": 0.0, "yaw": 0.0, "roll": 0.0, "head_above": 0.0, "tilt": 0.0, "wrist_dist": 0.5,
                    "wrist_vis": 0.9, "brightness": 130.0, "face_w_frac": 0.33, "blink_l": 0.0, "blink_r": 0.0,
                    "jaw_open": 0.0, "shoulder_vis": 0.9}

        def close(self):
            pass

    rt = Runtime(var_dir=tmp_path, train_on_start=False, extractor_factory=lambda cfg: FakeExtractor())
    rt.set_model(model)
    ok, jpg = cv2.imencode(".jpg", np.zeros((48, 64, 3), np.uint8))
    with TestClient(create_app(rt)) as c, c.websocket_connect("/ws/live") as ws:
        ws.send_json({"source": "camera", "driver_id": "C01", "profile": "taxi_empty"})
        assert ws.receive_json()["synthetic"] is False
        for _ in range(5):
            ws.send_bytes(jpg.tobytes())
        got = [ws.receive_json() for _ in range(2)]
        assert any(m["type"] == "tick" for m in got)
        ws.send_json({"type": "stop"})
        while True:
            m = ws.receive_json()
            if m["type"] == "ended":
                assert m["stopped_by_user"] is True
                break


def test_camera_unavailable_is_a_clear_error(model, tmp_path):
    def boom(cfg):
        raise RuntimeError("model download blocked")
    rt = Runtime(var_dir=tmp_path, train_on_start=False, extractor_factory=boom)
    rt.set_model(model)
    with TestClient(create_app(rt)) as c, c.websocket_connect("/ws/live") as ws:
        ws.send_json({"source": "camera"})
        m = ws.receive_json()
        assert m["type"] == "error" and "camera pipeline unavailable" in m["message"]


def test_model_not_ready_returns_503(tmp_path):
    rt = Runtime(var_dir=tmp_path, train_on_start=False)
    rt._ready.set(); rt.model_info["state"] = "error"
    app = create_app(rt)
    with TestClient(app) as c:
        # startup tries to load; with no model and no training it errors -> demo endpoints say 503
        import time; time.sleep(0.3)
        assert c.post("/api/sessions/demo", json={}).status_code == 503
        assert c.get("/api/health").json()["model"]["state"] == "error"
