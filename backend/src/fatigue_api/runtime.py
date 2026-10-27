"""Process-wide state: config, model (loaded or trained in the background), history store, session logs."""
from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Callable, List, Optional

from fatigue.config import Config, load_config
from fatigue.models import FusionModel
from fatigue.pretrip import HistoryStore

log = logging.getLogger("fatigue_api")


def default_var_dir() -> Path:
    return Path(os.environ.get("FATIGUE_VAR_DIR", Path.cwd() / "var"))


def default_results_dir() -> Path:
    env = os.environ.get("FATIGUE_RESULTS_DIR")
    if env:
        return Path(env)
    for candidate in (Path.cwd() / "results", Path(__file__).resolve().parents[2] / "results"):
        if candidate.exists():
            return candidate
    return Path.cwd() / "results"


class Runtime:
    def __init__(self, var_dir: Optional[Path] = None, results_dir: Optional[Path] = None,
                 model_path: Optional[str] = None,
                 extractor_factory: Optional[Callable[[Config], object]] = None,
                 train_on_start: bool = True, n_train_drivers: int = 8, train_seconds: float = 240.0):
        self.cfg: Config = load_config()
        self.var = Path(var_dir or default_var_dir())
        self.results_dir = Path(results_dir or default_results_dir())
        self.model_path = model_path or os.environ.get("FATIGUE_MODEL")
        self.extractor_factory = extractor_factory or self._default_extractor
        self.train_on_start = train_on_start
        self.n_train, self.train_s = n_train_drivers, train_seconds
        (self.var / "sessions").mkdir(parents=True, exist_ok=True)
        self.store = HistoryStore(str(self.var / "history"))
        self._model: Optional[FusionModel] = None
        self.model_info: dict = {"state": "loading", "synthetic": None, "source": None, "error": None}
        self._ready = threading.Event()

    # ---------------------------------------------------------------- model
    @staticmethod
    def _default_extractor(cfg: Config):
        from fatigue.extractor import LandmarkExtractor
        return LandmarkExtractor(cfg)

    def start(self) -> None:
        threading.Thread(target=self._load_model, daemon=True, name="model-loader").start()

    def _load_model(self) -> None:
        try:
            cached = self.var / "model.joblib"
            # Auto-detect trained bundle.joblib in models/
            bundle_candidates = [
                Path.cwd() / "models" / "bundle.joblib",
                Path(__file__).resolve().parents[2] / "models" / "bundle.joblib",
            ]
            prod_model = next((p for p in bundle_candidates if p.exists()), None)

            if self.model_path and os.path.exists(self.model_path):
                self._model = FusionModel.load(self.model_path)
                self.model_info.update(state="ready", synthetic=False, source=self.model_path)
            elif prod_model:
                self._model = FusionModel.load(str(prod_model))
                self.model_info.update(state="ready", synthetic=False, source=str(prod_model))
            elif cached.exists():
                self._model = FusionModel.load(str(cached))
                self.model_info.update(state="ready", synthetic=True, source="cached synthetic-trained model")
            elif self.train_on_start:
                from fatigue.synthetic import make_dataset
                from fatigue.training import fit_final
                log.info("no model found: training a SYNTHETIC demo model (%d drivers) ...", self.n_train)
                self.model_info.update(state="training")
                frames, _ = make_dataset(self.n_train, duration_s=self.train_s, seed=21)
                self._model = fit_final(frames, self.cfg)
                self._model.save(str(cached))
                self.model_info.update(state="ready", synthetic=True, source="trained on synthetic drivers at startup")
            else:
                raise RuntimeError("no model available (set FATIGUE_MODEL or enable train_on_start)")
        except Exception as e:                      # surfaced via /api/health, never crashes the server
            log.exception("model load failed")
            self.model_info.update(state="error", error=str(e))
        finally:
            self._ready.set()

    def set_model(self, model: FusionModel, synthetic: bool = True, source: str = "injected") -> None:
        self._model = model
        self.model_info.update(state="ready", synthetic=synthetic, source=source, error=None)
        self._ready.set()

    @property
    def model(self) -> FusionModel:
        if self._model is None:
            raise ModelNotReady(self.model_info.get("error") or self.model_info["state"])
        return self._model

    # ---------------------------------------------------------------- sessions on disk
    @property
    def sessions_dir(self) -> Path:
        return self.var / "sessions"

    def session_ids(self) -> List[str]:
        return sorted((p.name for p in self.sessions_dir.iterdir()
                       if (p / "session_log.csv").exists()), reverse=True)

    def session_log(self, sid: str) -> Path:
        if not sid.replace("-", "").replace("_", "").isalnum():
            raise KeyError(sid)
        p = self.sessions_dir / sid / "session_log.csv"
        if not p.exists():
            raise KeyError(sid)
        return p


class ModelNotReady(RuntimeError):
    pass
