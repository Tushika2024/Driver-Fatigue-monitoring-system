"""Batch extraction: video folder -> features.csv (resumable, parallel across videos)."""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import List, Sequence, Tuple

import pandas as pd
from joblib import Parallel, delayed

from .config import Config
from .schema import FRAME_COLUMNS

VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}


def discover_videos(root: str, labels: Sequence[str]) -> List[Tuple[str, str, Path]]:
    """Find <anything>/<subject>/<label>.<ext>. Subject = parent folder name, label = file stem.
    (The exact RLDD folder layout is not assumed beyond that convention.)"""
    found = []
    for p in sorted(Path(root).rglob("*")):
        if p.is_file() and p.suffix.lower() in VIDEO_EXT and p.stem in set(labels):
            found.append((p.parent.name, p.stem, p))
    return found


def _one(subject: str, label: str, path: Path, part: str, cfg_dict: dict, models_dir: str,
         max_seconds: float = 0.0) -> str:
    from .config import _build, Config as C
    from .extractor import LandmarkExtractor
    from .sources import VideoSource
    cfg = _build(C, cfg_dict)
    if os.path.exists(part):
        return part
    ex = LandmarkExtractor(cfg, models_dir)
    rows = []
    src = VideoSource(str(path), cfg.extraction.fps, max_width=cfg.extraction.max_width)
    tag, t_start = f"[{subject}/{label}]", time.monotonic()
    print(f"{tag} started: {path.name}, {src.total_out or '?'} frames to process", flush=True)
    try:
        for i, fr in enumerate(src):
            if max_seconds and fr.t > max_seconds:
                break
            f = ex.process(fr.image, fr.t)
            f.update(subject=subject, label=label, video=path.name, frame_idx=i)
            rows.append(f)
            if len(rows) % 600 == 0:
                el = time.monotonic() - t_start
                rate = len(rows) / el
                tot = src.total_out
                eta = f", ~{(tot - len(rows)) / rate / 60:.0f} min left" if tot and rate > 0 else ""
                pct = f"{100 * len(rows) / tot:.0f}%" if tot else f"{len(rows)} frames"
                print(f"{tag} {pct} at {rate:.1f} frames/s{eta}", flush=True)
    finally:
        ex.close()
        src.close()
    print(f"{tag} done: {len(rows)} frames in {(time.monotonic() - t_start) / 60:.1f} min", flush=True)
    df = pd.DataFrame(rows)
    for c in FRAME_COLUMNS:
        if c not in df:
            df[c] = float("nan")
    df[FRAME_COLUMNS].to_csv(part + ".tmp", index=False)
    os.replace(part + ".tmp", part)
    return part


def extract_all(root: str, out_csv: str, cfg: Config, labels: Sequence[str] = ("0", "10"),
                jobs: int = 1, log=print, max_seconds: float = 0.0) -> int:
    vids = discover_videos(root, labels)
    if not vids:
        raise SystemExit(f"no videos named <label>.<ext> for labels {list(labels)} found under {root}")
    parts_dir = out_csv + ".parts"
    os.makedirs(parts_dir, exist_ok=True)
    jobs_list = [(s, l, p, os.path.join(parts_dir, f"{s}__{l}__{p.stem}.csv")) for s, l, p in vids]
    log(f"{len(vids)} videos, {len({s for s, _, _ in vids})} subjects; resumable parts in {parts_dir}")
    Parallel(n_jobs=jobs)(delayed(_one)(s, l, p, part, cfg.to_dict(), cfg.extraction.models_dir, max_seconds)
                          for s, l, p, part in jobs_list)
    dfs = [pd.read_csv(part, dtype={"subject": str, "label": str, "video": str}) for *_, part in jobs_list]
    pd.concat(dfs, ignore_index=True).to_csv(out_csv, index=False)
    log(f"wrote {out_csv}")
    return len(vids)
