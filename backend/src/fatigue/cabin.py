"""Cabin context -> how to alert. Two axes: severity level x who else is in the cabin.

SEVERE is never suppressed, only REDIRECTED. If a loud alarm would harm others and nobody
else can be told, it falls back to cabin audio and says so out loud.
"""
from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from typing import Dict, List, Optional

import yaml

from .alerts import CRITICAL, MILD, MODERATE, NONE, SEVERE
from .schema import LEVEL_NAMES

CABIN, PRIVATE, CREW, DISPATCH = "cabin", "private", "crew", "dispatch"
CHIME, VOICE, ALARM, NOTIFY = "chime", "voice", "alarm", "notify"


@dataclass(frozen=True)
class VehicleProfile:
    name: str
    others_present: bool          # P1
    second_driver: bool           # P2
    overhearing_harmful: bool     # P3
    has_dispatch: bool = False
    private_output: bool = False  # a genuinely private speaker exists (else 'private' = low volume)

    @property
    def critical_enabled(self) -> bool:
        return self.has_dispatch


@dataclass(frozen=True)
class Action:
    level: int
    channel: str                  # cabin | private | crew | dispatch
    style: str                    # chime | voice | alarm | notify
    volume: str                   # soft | low | normal | loud | n/a
    message: str
    repeat_until_ack: bool = False
    announce: Optional[str] = None

    @property
    def reaches_human(self) -> bool:
        return True


def load_profiles() -> Dict[str, VehicleProfile]:
    raw = yaml.safe_load(resources.files("fatigue").joinpath("data/vehicle_profiles.yaml").read_text())
    return {k: VehicleProfile(name=k, **v) for k, v in raw["profiles"].items()}


def get_profile(name: Optional[str]) -> VehicleProfile:
    profs = load_profiles()
    raw = yaml.safe_load(resources.files("fatigue").joinpath("data/vehicle_profiles.yaml").read_text())
    key = name or raw["default"]
    if key not in profs:
        raise KeyError(f"unknown vehicle profile '{key}'. Available: {sorted(profs)}")
    return profs[key]


def route(level: int, reason: str, p: VehicleProfile) -> List[Action]:
    """Pure function: (level, reason, cabin context) -> actions to perform."""
    if level <= NONE:
        return []
    private = p.overhearing_harmful
    ch = PRIVATE if private else CABIN
    if level == MILD:
        return [Action(MILD, ch, CHIME, "soft", "chime")]
    if level == MODERATE:
        return [Action(MODERATE, ch, VOICE, "low" if private else "normal",
                       f"Signs of fatigue: {reason}. Please consider a break.")]
    # SEVERE and CRITICAL
    msg = f"Severe fatigue: {reason}. Pull over safely."
    if not private:
        acts = [Action(SEVERE, CABIN, ALARM, "loud", msg, repeat_until_ack=True)]
        if p.second_driver:
            acts.append(Action(SEVERE, CREW, NOTIFY, "n/a", f"Driver fatigue (severe): {reason}"))
    else:
        acts = [Action(SEVERE, PRIVATE, VOICE, "low" if p.private_output else "normal", msg,
                       repeat_until_ack=True)]
        redirected = False
        if p.second_driver:
            acts.append(Action(SEVERE, CREW, NOTIFY, "n/a", f"Driver fatigue (severe): {reason}"))
            redirected = True
        elif p.has_dispatch:
            acts.append(Action(SEVERE, DISPATCH, NOTIFY, "n/a", f"Driver fatigue (severe): {reason}"))
            redirected = True
        if not redirected and not p.private_output:
            acts.append(Action(SEVERE, CABIN, VOICE, "loud",
                               "Severe fatigue. Pull over safely.", repeat_until_ack=True,
                               announce="No second person or dispatch can be reached; using cabin audio."))
    if level >= CRITICAL and p.has_dispatch:
        acts.append(Action(CRITICAL, DISPATCH, NOTIFY, "n/a",
                           f"CRITICAL: driver fatigue unacknowledged. {reason}"))
    return acts


def describe(p: VehicleProfile) -> str:
    rows = []
    for lvl in (MILD, MODERATE, SEVERE, CRITICAL):
        a = route(lvl, "<reason>", p)
        rows.append(f"  {LEVEL_NAMES[lvl]:9s} -> " + ("; ".join(f"{x.channel}:{x.style}/{x.volume}" for x in a)
                                                        if a else "(no higher level)"))
    return f"{p.name}\n" + "\n".join(rows)
