"""JSON-safe conversion: NaN/inf -> null, numpy scalars -> python, dataclasses -> dicts."""
from __future__ import annotations

import dataclasses
import math
from typing import Any

import numpy as np


def clean(x: Any) -> Any:
    if dataclasses.is_dataclass(x) and not isinstance(x, type):
        x = dataclasses.asdict(x)
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return clean(x.tolist())
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, (float, np.floating)):
        v = float(x)
        return v if math.isfinite(v) else None
    return x
