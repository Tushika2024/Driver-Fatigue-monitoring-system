"""Output sinks. The monitor only knows AlertSink.emit(action); speakers, earpieces, headrest
units, seat vibration and crew/dispatch notifications are interchangeable implementations."""
from __future__ import annotations

import json
import logging
import sys
import threading
import time
import urllib.request
from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional

from .cabin import CABIN, CREW, DISPATCH, PRIVATE, Action

log = logging.getLogger("fatigue.sinks")
VOLUME = {"soft": 0.3, "low": 0.45, "normal": 0.8, "loud": 1.0, "n/a": 0.0}


class AlertSink(ABC):
    @abstractmethod
    def emit(self, action: Action) -> None: ...


class ConsoleSink(AlertSink):
    def __init__(self, stream=None):
        self.stream = stream or sys.stdout

    def emit(self, a: Action) -> None:
        extra = f"  [{a.announce}]" if a.announce else ""
        print(f"ALERT L{a.level} -> {a.channel}/{a.style}/{a.volume}: {a.message}{extra}", file=self.stream)


class RecordingSink(AlertSink):
    def __init__(self):
        self.actions: List[Action] = []

    def emit(self, a: Action) -> None:
        self.actions.append(a)


class JsonlSink(AlertSink):
    def __init__(self, path: str):
        self.path = path

    def emit(self, a: Action) -> None:
        with open(self.path, "a") as fh:
            fh.write(json.dumps({"ts": time.time(), **a.__dict__}) + "\n")


class AudioSink(AlertSink):
    """Offline voice (pyttsx3) and tones. Degrades to the terminal bell if audio is unavailable."""

    def __init__(self, voice: bool = True):
        self._engine = None
        self._lock = threading.Lock()
        if voice:
            try:
                import pyttsx3
                self._engine = pyttsx3.init()
            except Exception as e:  # no audio device / driver
                log.warning("text-to-speech unavailable (%s); using console/bell only", e)

    def _tone(self, vol: float, long: bool) -> None:
        try:
            import winsound  # type: ignore
            winsound.Beep(int(600 + 600 * vol), 700 if long else 200)
        except Exception:
            sys.stdout.write("\a"); sys.stdout.flush()

    def emit(self, a: Action) -> None:
        threading.Thread(target=self._play, args=(a,), daemon=True).start()

    def _play(self, a: Action) -> None:
        with self._lock:
            vol = VOLUME.get(a.volume, 0.8)
            if a.style in ("chime", "alarm") or self._engine is None:
                self._tone(vol, a.style == "alarm")
            if a.style in ("voice", "alarm") and self._engine is not None:
                try:
                    self._engine.setProperty("volume", vol)
                    if a.announce:
                        self._engine.say(a.announce)
                    self._engine.say(a.message)
                    self._engine.runAndWait()
                except Exception as e:
                    log.warning("speech failed: %s", e)
                    self._tone(vol, True)


class NotifySink(AlertSink):
    """Crew / dispatch notification. Plug in a callable, or give a webhook URL (JSON POST)."""

    def __init__(self, channel: str, callback: Optional[Callable[[Action], None]] = None,
                 url: Optional[str] = None, timeout: float = 2.0):
        self.channel, self.cb, self.url, self.timeout = channel, callback, url, timeout

    def emit(self, a: Action) -> None:
        if self.cb:
            self.cb(a)
        elif self.url:
            body = json.dumps({"channel": a.channel, "level": a.level, "message": a.message}).encode()
            req = urllib.request.Request(self.url, data=body, headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=self.timeout).read()
        else:
            log.warning("[stub] would notify %s: %s", self.channel, a.message)


class SinkRouter:
    """Channel -> sinks. Never raises: an alert path must not crash the monitor."""

    def __init__(self, mapping: Optional[Dict[str, List[AlertSink]]] = None, fallback: Optional[AlertSink] = None):
        self.mapping = mapping or {}
        self.fallback = fallback or ConsoleSink()

    @classmethod
    def default(cls, audio: bool = False) -> "SinkRouter":
        con = ConsoleSink()
        out = [con] + ([AudioSink()] if audio else [])
        return cls({CABIN: out, PRIVATE: out, CREW: [con, NotifySink(CREW)],
                    DISPATCH: [con, NotifySink(DISPATCH)]}, con)

    def emit_all(self, actions: List[Action]) -> int:
        n = 0
        for a in actions:
            for s in self.mapping.get(a.channel, [self.fallback]):
                try:
                    s.emit(a); n += 1
                except Exception as e:
                    log.error("sink %s failed: %s", type(s).__name__, e)
                    try:
                        self.fallback.emit(a)
                    except Exception:
                        pass
        return n
