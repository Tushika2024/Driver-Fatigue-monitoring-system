"""frames table -> windows table (and microsleep events), via the shared StreamProcessor."""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import pandas as pd
from joblib import Parallel, delayed

from .config import Config
from .reference import DriverNormaliser, Reference
from .schema import Z_SOURCES
from .stream import StreamProcessor

KEY = ["subject", "label", "video"]


def read_frames(path: str) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"subject": str, "label": str, "video": str})


def _records(g: pd.DataFrame):
    cols = [c for c in g.columns if c not in ("subject", "label", "video")]
    return g[cols].to_dict("records")


def enrol_profile(clip: pd.DataFrame, cfg: Config) -> dict:
    """Freeze a reference + normaliser from the first `enrol_s` seconds of a clip."""
    sp = StreamProcessor(cfg, "enrolled")
    t0 = float(clip["t"].iloc[0])
    for r in _records(clip[clip["t"] <= t0 + cfg.reference.enrol_s + cfg.window.stride_s]):
        sp.push(r)
    if sp.ref._frozen is None:
        sp.ref.freeze()
    if sp.norm is not None:
        sp.norm.freeze()
    return sp.profile()


def processor_from_profile(cfg: Config, prof: dict, window_s: Optional[float] = None) -> StreamProcessor:
    ref = Reference.from_profile(cfg, prof["ref"])
    norm = DriverNormaliser(cfg.reference, Z_SOURCES, frozen=prof["norm"])
    return StreamProcessor(cfg, "enrolled", ref=ref, norm=norm, window_s=window_s)


def _one_clip(g: pd.DataFrame, cfg: Config, mode: str, window_s: Optional[float],
              prof: Optional[dict]) -> Tuple[pd.DataFrame, list]:
    sp = (processor_from_profile(cfg, prof, window_s) if prof is not None
          else StreamProcessor(cfg, mode, window_s=window_s))
    w = pd.DataFrame(sp.run(_records(g)))
    ev = [(t, "microsleep") for t in sp.microsleeps]
    for k in KEY:
        w[k] = g[k].iloc[0]
    return w, [dict(zip(KEY, [g[k].iloc[0] for k in KEY]), t=t, kind=kind) for t, kind in ev]


def build_windows(frames: pd.DataFrame, cfg: Config, mode: Optional[str] = None,
                  window_s: Optional[float] = None, profiles: Optional[Dict[Tuple[str, str], dict]] = None,
                  n_jobs: int = 1) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """profiles: optional {(subject, video): enrol profile} to enrol on a DIFFERENT clip."""
    mode = mode or cfg.reference.mode
    groups = [g for _, g in frames.groupby(KEY, sort=False)]
    res = Parallel(n_jobs=n_jobs)(
        delayed(_one_clip)(g, cfg, mode, window_s,
                           (profiles or {}).get((g["subject"].iloc[0], g["video"].iloc[0])))
        for g in groups)
    W = pd.concat([r[0] for r in res], ignore_index=True)
    E = pd.DataFrame([e for r in res for e in r[1]], columns=KEY + ["t", "kind"])
    return W, E
