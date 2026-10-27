"""Typed, validated configuration. Unknown keys are rejected (typos fail loudly)."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import typing
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


class ConfigError(ValueError):
    pass


@dataclass
class ExtractionCfg:
    fps: float = 30.0
    pose_every: int = 3
    use_pose: bool = True
    models_dir: str = "models"
    max_width: int = 640      # shrink big frames before MediaPipe (speed); 0 = keep original size


@dataclass
class WindowCfg:
    length_s: float = 3.0
    stride_s: float = 1.5


@dataclass
class EyesCfg:
    fixed_open_ear: float = 0.30
    fixed_closed_ear: float = 0.20
    closed_frac: float = 0.50
    valid_min: float = 0.0
    valid_max: float = 0.60
    microsleep_s: float = 1.0
    fixed_grid: List[float] = field(default_factory=lambda: [0.12, 0.15, 0.18, 0.20, 0.22, 0.25, 0.28])


@dataclass
class BlinkCfg:
    start_level: float = 0.80
    min_depth: float = 0.40
    max_blink_s: float = 2.0
    max_gap_s: float = 0.10
    slow_blink_s: float = 0.40
    hard_squint: float = 0.50
    context_s: float = 30.0


@dataclass
class HeadCfg:
    drop_deg: float = 12.0
    fixed_head_above: float = 0.85


@dataclass
class BehaviourCfg:
    hand_face_dist: float = 0.60
    hand_min_vis: float = 0.50
    hand_min_s: float = 0.30
    shake_pp_deg: float = 15.0
    shake_window_s: float = 0.8
    shift_delta: float = 0.15
    shift_window_s: float = 1.0
    refractory_s: float = 1.5
    context_s: float = 30.0


@dataclass
class MouthCfg:
    yawn_mar: float = 0.55
    yawn_min_s: float = 1.0


@dataclass
class ReferenceCfg:
    mode: str = "adaptive"
    history_s: float = 120.0
    open_percentile: float = 90.0
    min_history_s: float = 20.0
    pitch_history_s: float = 300.0
    norm_method: str = "median_iqr"
    norm_history_windows: int = 240
    norm_min_windows: int = 12
    scale_floor: float = 0.05
    enrol_s: float = 60.0


@dataclass
class ModelCfg:
    kind: str = "logistic"
    C: float = 1.0
    inner_folds: int = 5
    use_z: bool = False


@dataclass
class QualityCfg:
    blind_face_frac: float = 0.30
    degraded_eye_frac: float = 0.60


@dataclass
class AlertsCfg:
    target_fpr: Dict[str, float] = field(
        default_factory=lambda: {"mild": 0.10, "late": 0.05, "moderate": 0.03, "severe": 0.005})
    degraded_fpr_scale: float = 0.5
    persist: Dict[str, int] = field(default_factory=lambda: {"mild": 2, "moderate": 2, "severe": 3})
    clear_windows: int = 3
    critical_after_s: float = 20.0
    ack_repeat_s: float = 8.0
    log_every_s: float = 3.0


@dataclass
class EvaluationCfg:
    warmup_s: float = 60.0
    pos_labels: List[str] = field(default_factory=lambda: ["10"])
    neg_labels: List[str] = field(default_factory=lambda: ["0"])
    earliness_labels: List[str] = field(default_factory=lambda: ["10", "prog"])
    baseline_persist: int = 2
    seed: int = 7
    n_boot: int = 300
    window_lengths_s: List[float] = field(default_factory=lambda: [2.0, 3.0, 5.0])


@dataclass
class PretripCfg:
    discard_s: float = 10.0
    capture_s: float = 60.0
    extended_s: float = 90.0
    min_blinks: int = 8
    quality_s: float = 3.0
    min_face_frac: float = 0.90
    brightness_range: List[float] = field(default_factory=lambda: [50.0, 220.0])
    face_width_range: List[float] = field(default_factory=lambda: [0.15, 0.60])
    max_frontal_deg: float = 25.0
    history_sessions: int = 30
    min_sessions: int = 3
    slot_hours: int = 4
    min_slot_sessions: int = 3
    weights: Dict[str, float] = field(
        default_factory=lambda: {"reopen": 0.40, "openness": 0.25, "rate": 0.20, "head": 0.15})
    caution_at: float = 1.0
    not_fit_at: float = 2.0
    outlier_z: float = 4.0


@dataclass
class ReportCfg:
    min_full_mode_frac: float = 0.60
    k_anonymity: int = 3
    slot_hours: int = 4
    episode_merge_s: float = 120.0


@dataclass
class Config:
    extraction: ExtractionCfg = field(default_factory=ExtractionCfg)
    window: WindowCfg = field(default_factory=WindowCfg)
    eyes: EyesCfg = field(default_factory=EyesCfg)
    blink: BlinkCfg = field(default_factory=BlinkCfg)
    head: HeadCfg = field(default_factory=HeadCfg)
    behaviour: BehaviourCfg = field(default_factory=BehaviourCfg)
    mouth: MouthCfg = field(default_factory=MouthCfg)
    reference: ReferenceCfg = field(default_factory=ReferenceCfg)
    model: ModelCfg = field(default_factory=ModelCfg)
    quality: QualityCfg = field(default_factory=QualityCfg)
    alerts: AlertsCfg = field(default_factory=AlertsCfg)
    evaluation: EvaluationCfg = field(default_factory=EvaluationCfg)
    pretrip: PretripCfg = field(default_factory=PretripCfg)
    report: ReportCfg = field(default_factory=ReportCfg)

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def hash(self) -> str:
        blob = json.dumps(self.to_dict(), sort_keys=True).encode()
        return hashlib.sha1(blob).hexdigest()[:10]

    def replace(self, **sections: Dict[str, Any]) -> "Config":
        """Return a copy with section fields overridden: cfg.replace(window={'length_s': 5})."""
        d = self.to_dict()
        for sec, vals in sections.items():
            if sec not in d:
                raise ConfigError(f"unknown section '{sec}'")
            d[sec].update(vals)
        new = _build(Config, d)
        validate(new)
        return new


def _build(cls, data: Any, path: str = ""):
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ConfigError(f"'{path or 'root'}' must be a mapping, got {type(data).__name__}")
    hints = typing.get_type_hints(cls)
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ConfigError(f"unknown key(s) {sorted(unknown)} in '{path or 'root'}'")
    kwargs: Dict[str, Any] = {}
    for f in dataclasses.fields(cls):
        if f.name not in data:
            continue
        v, h = data[f.name], hints[f.name]
        if dataclasses.is_dataclass(h):
            kwargs[f.name] = _build(h, v, f"{path}.{f.name}" if path else f.name)
        elif h is float:
            kwargs[f.name] = float(v)
        elif typing.get_origin(h) is dict:
            base = f.default_factory()  # type: ignore[misc]
            if not isinstance(v, dict):
                raise ConfigError(f"'{f.name}' must be a mapping")
            bad = set(v) - set(base)
            if bad:
                raise ConfigError(f"unknown key(s) {sorted(bad)} in '{path}.{f.name}'")
            base.update(v)
            kwargs[f.name] = base
        else:
            kwargs[f.name] = v
    return cls(**kwargs)


def validate(cfg: Config) -> None:
    errs = []
    if cfg.window.length_s <= 0 or cfg.window.stride_s <= 0:
        errs.append("window.length_s and stride_s must be > 0")
    if cfg.window.stride_s > cfg.window.length_s:
        errs.append("window.stride_s must be <= length_s")
    if cfg.extraction.fps < 5:
        errs.append("extraction.fps must be >= 5")
    if cfg.reference.mode not in ("fixed", "adaptive", "enrolled"):
        errs.append("reference.mode must be fixed|adaptive|enrolled")
    if cfg.reference.norm_method not in ("median_iqr", "mean_std"):
        errs.append("reference.norm_method must be median_iqr|mean_std")
    if cfg.model.kind not in ("logistic", "hgb"):
        errs.append("model.kind must be logistic|hgb")
    if not 0 < cfg.eyes.closed_frac < 1:
        errs.append("eyes.closed_frac must be in (0,1)")
    if not 50 <= cfg.reference.open_percentile <= 99:
        errs.append("reference.open_percentile must be in [50,99]")
    t = cfg.alerts.target_fpr
    if not (t["severe"] < t["moderate"] < t["mild"] <= 0.5):
        errs.append("alerts.target_fpr must satisfy severe < moderate < mild <= 0.5")
    w = cfg.pretrip.weights
    if abs(sum(w.values()) - 1.0) > 1e-6:
        errs.append("pretrip.weights must sum to 1")
    if cfg.pretrip.caution_at >= cfg.pretrip.not_fit_at:
        errs.append("pretrip.caution_at must be < not_fit_at")
    if not 0 < cfg.report.min_full_mode_frac <= 1:
        errs.append("report.min_full_mode_frac must be in (0,1]")
    if 24 % cfg.pretrip.slot_hours or 24 % cfg.report.slot_hours:
        errs.append("slot_hours must divide 24")
    if errs:
        raise ConfigError("; ".join(errs))


def _read_yaml(path: Optional[str | Path]) -> dict:
    if path is None:
        text = resources.files("fatigue").joinpath("data/default.yaml").read_text()
    else:
        text = Path(path).read_text()
    return yaml.safe_load(text) or {}


def load_config(path: Optional[str | Path] = None) -> Config:
    """Load the packaged defaults, then overlay a user YAML file if given."""
    base = _read_yaml(None)
    if path is not None:
        user = _read_yaml(path)
        for sec, vals in user.items():
            if isinstance(vals, dict) and isinstance(base.get(sec), dict):
                base[sec].update(vals)
            else:
                base[sec] = vals
    cfg = _build(Config, base)
    validate(cfg)
    return cfg
