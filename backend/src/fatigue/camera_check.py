"""Step-1 sanity check: does the signal exist on THIS camera?  `fatigue camera-check`

Shows EAR / MAR / pitch / yaw / FPS live. Close your eyes (EAR should roughly halve),
yawn (MAR spikes), nod forward (pitch rises). Keys: c = capture your open-eye level, q = quit.
"""
from __future__ import annotations

import time
from collections import deque

import cv2
import numpy as np

from .config import Config
from .extractor import LandmarkExtractor
from .sources import WebcamSource


def run(cfg: Config, index: int = 0) -> None:
    ex, src = LandmarkExtractor(cfg), WebcamSource(index, cfg.extraction.fps)
    hist, ref, last, fps = deque(maxlen=200), None, time.perf_counter(), 0.0
    for fr in src:
        f = ex.process(fr.image, fr.t)
        now = time.perf_counter(); fps = 0.9 * fps + 0.1 / max(now - last, 1e-6); last = now
        img = fr.image
        h, w = img.shape[:2]
        hist.append(f["ear"])
        txt = ["no face found"] if not f["face_found"] else [
            f"EAR {f['ear']:.3f}   MAR {f['mar']:.3f}", f"pitch {f['pitch']:+.0f}  yaw {f['yaw']:+.0f}  roll {f['roll']:+.0f}",
            f"FPS {fps:.0f}" + ("   (need > 15)" if fps < 15 else "")]
        if ref and f["face_found"]:
            rel = f["ear"] / ref
            txt.append(f"relative to your open level: {rel:.2f}" + ("   EYES CLOSING" if rel < cfg.eyes.closed_frac + 0.25 else ""))
        for i, t in enumerate(txt):
            cv2.putText(img, t, (10, 25 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        vals = np.array([v for v in hist if np.isfinite(v)])
        if len(vals) > 2:
            pts = [(int(i * w / 200), int(h - 10 - np.clip(v, 0, 0.5) / 0.5 * 80)) for i, v in enumerate(hist) if np.isfinite(v)]
            cv2.polylines(img, [np.array(pts, np.int32)], False, (255, 200, 0), 2)
        cv2.imshow("camera check (c = set baseline, q = quit)", img)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        if k == ord("c") and len(vals):
            ref = float(np.percentile(vals, 90)); print(f"open-eye level set to {ref:.3f}")
    src.close(); ex.close(); cv2.destroyAllWindows()
