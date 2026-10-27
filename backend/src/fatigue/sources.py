"""Frame sources: webcam, video file, or pre-extracted feature rows (tests, replay, demo)."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterable, Iterator, Mapping, Optional

import numpy as np


@dataclass
class Frame:
    t: float
    image: Optional[np.ndarray] = None
    features: Optional[Mapping[str, float]] = None


class WebcamSource:
    def __init__(self, index: int = 0, fps: float = 30.0, width: int = 640, height: int = 480):
        import cv2
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open camera {index}")
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def __iter__(self) -> Iterator[Frame]:
        t0 = time.monotonic()
        while True:
            ok, img = self.cap.read()
            if not ok:
                return
            yield Frame(time.monotonic() - t0, image=img)

    def close(self) -> None:
        self.cap.release()


class VideoSource:
    """Video file -> frames at ~target_fps. Skipped frames are grabbed, not decoded/converted (faster), and
    big frames are shrunk to max_width (landmarks are normalised, so accuracy is unaffected)."""

    def __init__(self, path: str, target_fps: float = 30.0, realtime: bool = False, max_width: int = 0):
        import cv2
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open video {path}")
        self.src_fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.step = max(1, int(round(self.src_fps / target_fps)))
        self.realtime = realtime
        self.max_width = int(max_width)
        n = self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        self.total_out = int(n // self.step) if n > 0 else 0       # expected frames yielded (0 = unknown)

    def __iter__(self) -> Iterator[Frame]:
        import cv2
        i, t0 = 0, time.monotonic()
        while True:
            if not self.cap.grab():
                return
            if i % self.step == 0:
                ok, img = self.cap.retrieve()
                if not ok:
                    return
                if self.max_width and img.shape[1] > self.max_width:
                    h = int(round(img.shape[0] * self.max_width / img.shape[1]))
                    img = cv2.resize(img, (self.max_width, h), interpolation=cv2.INTER_AREA)
                t = i / self.src_fps
                if self.realtime:
                    time.sleep(max(0.0, t - (time.monotonic() - t0)))
                yield Frame(t, image=img)
            i += 1

    def close(self) -> None:
        self.cap.release()


class FeatureSource:
    """Replays rows of a features table (already extracted or synthetic)."""

    def __init__(self, rows: Iterable[Mapping[str, float]]):
        self.rows = rows

    def __iter__(self) -> Iterator[Frame]:
        for r in self.rows:
            yield Frame(float(r["t"]), features=r)

    def close(self) -> None:
        pass
