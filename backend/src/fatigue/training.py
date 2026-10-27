"""Train the deployable model on ALL subjects (after evaluation has told you what to expect)."""
from __future__ import annotations

import pandas as pd

from . import evaluation as ev
from .config import Config
from .models import FusionModel
from .windows import build_windows


def fit_final(frames: pd.DataFrame, cfg: Config, n_jobs: int = 1) -> FusionModel:
    W, _ = build_windows(frames, cfg, cfg.reference.mode, n_jobs=n_jobs)
    D = ev.prepare(W, cfg)
    if D["y"].nunique() < 2:
        raise ValueError("training data needs both alert and drowsy windows")
    return FusionModel.from_features(cfg).fit(D, D["y"].to_numpy(), D["subject"].to_numpy())
