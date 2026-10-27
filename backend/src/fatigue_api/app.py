"""FastAPI application: REST for static/derived data, WebSocket for live streams."""
from __future__ import annotations

import asyncio
import io
import json
import math
import os
import re
import time
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from fatigue.alerts import NONE
from fatigue.cabin import Action, VehicleProfile, get_profile, load_profiles, route
from fatigue.live import LiveSession
from fatigue.pretrip import PretripSession
from fatigue.report import (depot_table, driver_metrics, episode_starts, read_log, synthetic_depot_logs)
from fatigue.schema import LEVEL_NAMES
from fatigue.sinks import ConsoleSink, RecordingSink, SinkRouter
from fatigue.sources import Frame, FeatureSource

from .jsonutil import clean
from .runtime import ModelNotReady, Runtime
from .scenarios import FPS, PRETRIP_STATES, SCENARIOS, build_rows, pretrip_rows

ID_RE = re.compile(r"^[A-Za-z0-9_.-]{1,32}$")
CHANNEL_LABEL = {"cabin": "Cabin speaker", "private": "Private (low-volume) voice",
                 "crew": "Crew notification", "dispatch": "Dispatch desk"}


def _ident(v: str, what: str = "id") -> str:
    if not ID_RE.match(v or ""):
        raise HTTPException(422, f"{what} must be 1-32 characters of A-Z a-z 0-9 _ . -")
    return v


def _action(a: Action) -> dict:
    return {"level": a.level, "level_name": LEVEL_NAMES[a.level], "channel": a.channel,
            "channel_label": CHANNEL_LABEL.get(a.channel, a.channel), "style": a.style, "volume": a.volume,
            "message": a.message, "repeat_until_ack": a.repeat_until_ack, "announce": a.announce}


def _profile(p: VehicleProfile) -> dict:
    return {"name": p.name, "others_present": p.others_present, "second_driver": p.second_driver,
            "overhearing_harmful": p.overhearing_harmful, "has_dispatch": p.has_dispatch,
            "private_output": p.private_output, "critical_enabled": p.critical_enabled}


class RouteBody(BaseModel):
    level: int = Field(ge=1, le=4)
    reason: str = Field(default="eyes closing for longer each time", max_length=200)
    profile: Optional[str] = None
    flags: Optional[Dict[str, bool]] = None


class DemoShiftBody(BaseModel):
    profile: str = "ambulance_patient"
    driver_id: str = "DEMO01"
    scenario: str = "gradual_fatigue"
    vehicle_id: str = "TRUCK-7"
    route_name: str = "Ring"


def create_app(runtime: Optional[Runtime] = None) -> FastAPI:
    rt = runtime or Runtime()
    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        if rt._model is None:
            rt.start()
        yield

    app = FastAPI(title="Driver Fatigue Monitor API", version="1.0.0", lifespan=lifespan)
    app.state.rt = rt
    origins = os.environ.get("FATIGUE_CORS", "http://localhost:5173,http://127.0.0.1:5173").split(",")
    app.add_middleware(CORSMiddleware, allow_origins=[o.strip() for o in origins if o.strip()],
                       allow_methods=["*"], allow_headers=["*"])

    @app.exception_handler(ModelNotReady)
    async def _not_ready(_, exc):                      # noqa: ANN001
        from fastapi.responses import JSONResponse
        return JSONResponse({"detail": f"model not ready: {exc}"}, status_code=503)

    # ------------------------------------------------------------------ meta
    @app.get("/api/health")
    def health():
        return {"ok": True, "model": rt.model_info, "time": datetime.now().isoformat(timespec="seconds"),
                "config_hash": rt.cfg.hash()}

    @app.get("/api/config")
    def config():
        c = rt.cfg
        return clean({"window_s": c.window.length_s, "stride_s": c.window.stride_s, "fps": c.extraction.fps,
                      "alerts": c.alerts, "pretrip": {"capture_s": c.pretrip.capture_s,
                                                       "discard_s": c.pretrip.discard_s,
                                                       "extended_s": c.pretrip.extended_s,
                                                       "min_blinks": c.pretrip.min_blinks,
                                                       "caution_at": c.pretrip.caution_at,
                                                       "not_fit_at": c.pretrip.not_fit_at},
                      "report": c.report, "microsleep_s": c.eyes.microsleep_s,
                      "closed_frac": c.eyes.closed_frac, "reference_min_s": c.reference.min_history_s})

    # ------------------------------------------------------------------ cabin routing
    @app.get("/api/profiles")
    def profiles():
        out = []
        for name, p in load_profiles().items():
            matrix = {LEVEL_NAMES[l]: [_action(a) for a in route(l, "<reason>", p)] for l in (1, 2, 3, 4)}
            out.append({**_profile(p), "matrix": matrix})
        return out

    @app.post("/api/cabin/route")
    def cabin_route(body: RouteBody):
        if body.flags is not None:
            allowed = {"others_present", "second_driver", "overhearing_harmful", "has_dispatch", "private_output"}
            if set(body.flags) - allowed:
                raise HTTPException(422, f"unknown flags: {sorted(set(body.flags) - allowed)}")
            p = VehicleProfile(name="custom", **{k: bool(body.flags.get(k, False)) for k in allowed})
        else:
            try:
                p = get_profile(body.profile)
            except KeyError as e:
                raise HTTPException(404, str(e))
        acts = route(body.level, body.reason, p)
        return {"profile": _profile(p), "level": body.level, "level_name": LEVEL_NAMES[body.level],
                "actions": [_action(a) for a in acts]}

    # ------------------------------------------------------------------ scenarios
    @app.get("/api/scenarios")
    def scenarios():
        return {"live": [{"id": s.id, "title": s.title, "blurb": s.blurb, "duration_s": s.duration_s}
                         for s in SCENARIOS.values()],
                "pretrip": [{"id": k, "title": v} for k, v in PRETRIP_STATES.items()]}

    # ------------------------------------------------------------------ pre-trip
    @app.get("/api/pretrip/history/{driver_id}")
    def pretrip_history(driver_id: str):
        _ident(driver_id, "driver_id")
        return {"driver_id": driver_id, "sessions": rt.store.load(driver_id)}

    @app.delete("/api/pretrip/history/{driver_id}")
    def pretrip_reset(driver_id: str):
        _ident(driver_id, "driver_id")
        p = rt.store._path(driver_id)
        if os.path.exists(p):
            os.remove(p)
        return {"driver_id": driver_id, "sessions": []}

    def _seed_history(driver_id: str, n: int = 4) -> int:
        """SIMULATION ONLY: a brand-new simulated driver gets a few normal days so there is a baseline."""
        have = len(rt.store.load(driver_id))
        base = datetime.now().replace(minute=0, second=0, microsecond=0)
        for i in range(max(0, n - have)):
            s = PretripSession(rt.cfg, driver_id, rt.store, base - timedelta(days=have + i + 1))
            for r in pretrip_rows(driver_id, "normal", seed=100 + have + i):
                s.push(r)
                if s.done:
                    break
            s.result()
        return max(0, n - have)

    @app.websocket("/ws/pretrip")
    async def ws_pretrip(ws: WebSocket):
        await ws.accept()
        try:
            msg = await ws.receive_json()
            driver = _ident(str(msg.get("driver_id", "")), "driver_id")
            state = msg.get("state", "normal")
            if state not in PRETRIP_STATES:
                await ws.send_json({"type": "error", "message": f"unknown state '{state}'"})
                return
            speed = float(min(max(float(msg.get("speed", 6)), 0.5), 60))
            if msg.get("source", "simulated") != "simulated":
                await ws.send_json({"type": "error", "message": "pre-trip over the browser camera is not enabled in "
                                    "this build; use the CLI (`fatigue pretrip --source webcam`) or the simulator."})
                return
            seeded = await run_in_threadpool(_seed_history, driver)
            rows = pretrip_rows(driver, state, seed=int(time.time()) % 10_000)
            sess = PretripSession(rt.cfg, driver, rt.store, datetime.now())
            p = rt.cfg.pretrip
            await ws.send_json({"type": "started", "driver_id": driver, "state": state, "synthetic": True,
                                "seeded_history": seeded, "history_n": len(rt.store.load(driver)),
                                "capture_s": p.capture_s, "discard_s": p.discard_s, "extended_s": p.extended_s})
            i, n, step = 0, len(rows), int(FPS)
            while i < n and not sess.done:
                t_wall = time.perf_counter()
                chunk = rows[i:i + step]
                samples = []
                for k, r in enumerate(chunk):
                    sess.push(r)
                    if k % 3 == 0:
                        samples.append([r["t"], r["ear"], r["pitch"], r["face_found"]])
                    if sess.done:
                        break
                i += step
                el = (sess.rows[-1]["t"] - sess.t0) if sess.rows else 0.0
                phase = "settling" if el < p.discard_s else "measuring"
                await ws.send_json(clean({"type": "progress", "elapsed": el, "phase": phase, "samples": samples}))
                await asyncio.sleep(max(0.0, 1.0 / speed - (time.perf_counter() - t_wall)))
            res = sess.result()
            await ws.send_json(clean({"type": "result", "result": res, "history_n": len(rt.store.load(driver))}))
        except WebSocketDisconnect:
            return
        except HTTPException as e:
            await ws.send_json({"type": "error", "message": e.detail})
        finally:
            try:
                await ws.close()
            except Exception:
                pass

    # ------------------------------------------------------------------ live monitoring
    @app.websocket("/ws/live")
    async def ws_live(ws: WebSocket):
        await ws.accept()
        session: Optional[LiveSession] = None
        reader: Optional[asyncio.Task] = None
        try:
            msg = await ws.receive_json()
            driver = _ident(str(msg.get("driver_id", "D01")), "driver_id")
            vehicle = _ident(str(msg.get("vehicle_id", "V1")), "vehicle_id")
            route_name = _ident(str(msg.get("route", "route")), "route")
            source = msg.get("source", "simulated")
            try:
                profile = get_profile(msg.get("profile"))
            except KeyError as e:
                await ws.send_json({"type": "error", "message": str(e)})
                return
            if rt._model is None:
                await ws.send_json({"type": "error", "message": f"model not ready ({rt.model_info['state']})"})
                return
            synthetic = source == "simulated"
            scn = SCENARIOS.get(msg.get("scenario", "gradual_fatigue"))
            if synthetic and scn is None:
                await ws.send_json({"type": "error", "message": f"unknown scenario '{msg.get('scenario')}'"})
                return
            extractor = None
            if not synthetic:
                try:
                    extractor = await run_in_threadpool(rt.extractor_factory, rt.cfg)
                except Exception as e:
                    await ws.send_json({"type": "error", "message":
                                        f"camera pipeline unavailable: {e}. (MediaPipe model files are downloaded on first "
                                        f"use; if the server has no network, place them in the models/ folder.)"})
                    return
            start = datetime.now()
            sid = f"{start:%Y%m%d-%H%M%S}-{driver}"
            log_path = str(rt.sessions_dir / sid / "session_log.csv")
            rec = RecordingSink()
            sinks = SinkRouter({c: [rec] for c in ("cabin", "private", "crew", "dispatch")}, rec)
            session = LiveSession(rt.model, profile, sinks, driver, vehicle, route_name, log_path,
                                  extractor=extractor, start=start, synthetic=synthetic)
            state: Dict[str, Any] = {"ack": False, "stop": False, "speed": float(min(max(float(msg.get("speed", 20)), 1), 120))}
            frames: "asyncio.Queue[bytes]" = asyncio.Queue(maxsize=2)

            async def read_loop():
                try:
                    while True:
                        m = await ws.receive()
                        if m["type"] == "websocket.disconnect":
                            state["stop"] = True
                            return
                        if m.get("bytes") is not None:
                            if frames.full():
                                try: frames.get_nowait()          # drop the oldest: stay real-time
                                except asyncio.QueueEmpty: pass
                            frames.put_nowait(m["bytes"])
                        elif m.get("text"):
                            try:
                                d = json.loads(m["text"])
                            except ValueError:
                                continue
                            if d.get("type") == "ack":
                                state["ack"] = True
                            elif d.get("type") == "stop":
                                state["stop"] = True
                            elif d.get("type") == "speed":
                                state["speed"] = float(min(max(float(d.get("speed", 20)), 1), 120))
                except Exception:
                    state["stop"] = True

            reader = asyncio.create_task(read_loop())
            await ws.send_json(clean({"type": "started", "session_id": sid, "synthetic": synthetic, "source": source,
                                      "profile": _profile(profile), "scenario": scn.id if synthetic else None,
                                      "duration_s": scn.duration_s if synthetic else None,
                                      "window_s": rt.cfg.window.length_s,
                                      "reference_min_s": rt.cfg.reference.min_history_s}))

            async def emit(u, t_now: float):
                await ws.send_json(clean({
                    "type": "update", "t": u.t, "level": u.level, "level_name": LEVEL_NAMES[u.level], "mode": u.mode,
                    "score": u.score, "early": u.early, "late": u.late, "reason": u.reason,
                    "actions": [_action(a) for a in u.actions]}))

            if synthetic:
                rows = await run_in_threadpool(build_rows, scn, driver, int(time.time()) % 10_000)
                step, i, n = int(FPS), 0, len(rows)
                while i < n and not state["stop"]:
                    t_wall = time.perf_counter()
                    if state["ack"]:
                        session.acknowledge(float(rows[max(i - 1, 0)]["t"]))
                        state["ack"] = False
                        await ws.send_json({"type": "acked", "t": float(rows[max(i - 1, 0)]["t"])})
                    samples, ups = [], []
                    for k, r in enumerate(rows[i:i + step]):
                        u = session.process(Frame(float(r["t"]), features=r))
                        if u is not None:
                            ups.append(u)
                        if k % 3 == 0:
                            samples.append((r["t"], r["ear"], r["pitch"], r["mar"], r["face_found"]))
                    i += step
                    op = float(getattr(session.sp.ref, "open_level", math.nan))
                    await ws.send_json(clean({"type": "tick", "t": rows[min(i, n) - 1]["t"], "open_level": op,
                                              "samples": samples, "progress": min(i / n, 1.0)}))
                    for u in ups:
                        await emit(u, u.t)
                    await asyncio.sleep(max(0.0, 1.0 / state["speed"] - (time.perf_counter() - t_wall)))
            else:
                import cv2
                t0 = time.monotonic()
                while not state["stop"]:
                    try:
                        data = await asyncio.wait_for(frames.get(), timeout=0.5)
                    except asyncio.TimeoutError:
                        continue
                    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                    if img is None:
                        continue
                    t = time.monotonic() - t0
                    if state["ack"]:
                        session.acknowledge(t)
                        state["ack"] = False
                        await ws.send_json({"type": "acked", "t": t})
                    u = await run_in_threadpool(session.process, Frame(t, image=img))
                    op = float(getattr(session.sp.ref, "open_level", math.nan))
                    await ws.send_json(clean({"type": "tick", "t": t, "open_level": op, "samples": [], "progress": None}))
                    if u is not None:
                        await emit(u, t)
            perf = session.perf.summary()
            await ws.send_json(clean({"type": "ended", "session_id": sid, "perf": perf,
                                      "stopped_by_user": state["stop"]}))
        except WebSocketDisconnect:
            pass
        except Exception as e:                         # never leave the UI hanging
            try:
                await ws.send_json({"type": "error", "message": f"{type(e).__name__}: {e}"})
            except Exception:
                pass
        finally:
            if reader:
                reader.cancel()
            if session is not None:
                session.close()
            try:
                await ws.close()
            except Exception:
                pass

    # ------------------------------------------------------------------ sessions and reports
    def _meta_of(sid: str) -> dict:
        p = rt.session_log(sid)
        try:
            return json.load(open(str(p) + ".meta.json"))
        except (FileNotFoundError, ValueError):
            return {}

    @app.get("/api/sessions")
    def sessions():
        out = []
        for sid in rt.session_ids():
            m = _meta_of(sid)
            out.append({"id": sid, "driver_id": m.get("driver_id"), "vehicle_id": m.get("vehicle_id"),
                        "route": m.get("route"), "profile": m.get("profile"), "start": m.get("start"),
                        "synthetic": bool(m.get("synthetic", False))})
        return out

    @app.post("/api/sessions/demo")
    async def demo_shift(body: DemoShiftBody):
        _ident(body.driver_id, "driver_id"); _ident(body.vehicle_id, "vehicle_id"); _ident(body.route_name, "route_name")
        scn = SCENARIOS.get(body.scenario)
        if scn is None:
            raise HTTPException(404, f"unknown scenario '{body.scenario}'")
        try:
            profile = get_profile(body.profile)
        except KeyError as e:
            raise HTTPException(404, str(e))
        model = rt.model

        def run() -> str:
            start = datetime.now().replace(microsecond=0)
            sid = f"{start:%Y%m%d-%H%M%S}-{body.driver_id}"
            rows = build_rows(scn, body.driver_id, seed=int(time.time()) % 10_000)
            rec = RecordingSink()
            sinks = SinkRouter({c: [rec] for c in ("cabin", "private", "crew", "dispatch")}, rec)
            s = LiveSession(model, profile, sinks, body.driver_id, body.vehicle_id, body.route_name,
                            str(rt.sessions_dir / sid / "session_log.csv"), start=start, synthetic=True)
            s.run(FeatureSource(rows), ack_when=lambda u: u.t % 60 < 3)
            return sid
        return {"id": await run_in_threadpool(run)}

    def _episodes(df: pd.DataFrame) -> List[dict]:
        t, lvl = df["t"].to_numpy(float), df["level"].to_numpy(int)
        out, i = [], 0
        while i < len(lvl):
            if lvl[i] > 0:
                j = i
                while j < len(lvl) and lvl[j] > 0:
                    j += 1
                seg = df.iloc[i:j]
                reasons = [r for r in seg["reason"].astype(str) if r and r != "nan"]
                out.append({"start": float(t[i]), "end": float(t[j - 1]), "peak": int(lvl[i:j].max()),
                            "peak_name": LEVEL_NAMES[int(lvl[i:j].max())],
                            "reason": reasons[-1] if reasons else ""})
                i = j
            else:
                i += 1
        return out

    @app.get("/api/sessions/{sid}/report")
    def session_report(sid: str):
        try:
            p = rt.session_log(sid)
        except KeyError:
            raise HTTPException(404, "no such session")
        df, meta = read_log(str(p))
        m = driver_metrics(df, rt.cfg)
        step = max(1, len(df) // 400)
        d = df.iloc[::step]
        series = [{"t": float(r.t), "score": None if pd.isna(r.score) else float(r.score), "level": int(r.level),
                   "mode": str(r.mode)} for r in d.itertuples()]
        modes = df["mode"].value_counts(normalize=True).to_dict() if len(df) else {}
        return clean({"id": sid, "meta": meta, "synthetic": bool(meta.get("synthetic", False)), "metrics": m,
                      "series": series, "episodes": _episodes(df) if len(df) else [],
                      "mode_share": modes, "thresholds": {"min_full_mode_frac": rt.cfg.report.min_full_mode_frac},
                      "start_ts": str(df["ts"].iloc[0]) if len(df) else None})

    # ------------------------------------------------------------------ depot (aggregate only)
    @app.get("/api/depot")
    def depot(source: str = Query("synthetic", pattern="^(synthetic|sessions)$")):
        k = rt.cfg.report.k_anonymity
        if source == "synthetic":
            logs, synthetic = synthetic_depot_logs(12, 6, seed=3), True
        else:
            logs, synthetic = [], False
            for sid in rt.session_ids():
                df, meta = read_log(str(rt.session_log(sid)))
                logs.append(df)
                synthetic = synthetic or bool(meta.get("synthetic", False))
        tbl = depot_table(logs, rt.cfg)
        cells = [] if tbl.empty else clean(tbl.to_dict("records"))
        return {"k": k, "synthetic": synthetic, "source": source, "n_logs": len(logs),
                "slots": sorted({c["slot_label"] for c in cells}, key=lambda s: s),
                "routes": sorted({c["route"] for c in cells}), "cells": cells}

    # ------------------------------------------------------------------ offline evaluation results
    @app.get("/api/evaluation")
    def evaluation():
        d = rt.results_dir
        names = ["main_comparison", "leakage_demo", "ablations", "earliness", "false_alarms_per_hour", "fairness"]
        tables: Dict[str, Any] = {}
        for n in names:
            f = d / f"{n}.csv"
            if f.exists():
                df = pd.read_csv(f)
                tables[n] = {"columns": list(df.columns), "rows": clean(df.to_dict("records"))}
        md = d / "RESULTS.md"
        text = md.read_text() if md.exists() else ""
        synthetic = ("SYNTHETIC" in text[:400]) if text else True
        return {"available": bool(tables), "synthetic": synthetic, "dir": str(d), "tables": tables,
                "headline": text.split("\n")[2] if len(text.split("\n")) > 2 else ""}

    return app
