
import pytest

from fatigue.config import load_config
from fatigue.synthetic import make_dataset
from fatigue.windows import build_windows


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def dataset():
    return make_dataset(6, duration_s=200, seed=4)


@pytest.fixture(scope="session")
def windows(cfg, dataset):
    fr, _ = dataset
    return build_windows(fr, cfg, "adaptive")


@pytest.fixture(scope="session")
def model(cfg, windows):
    """One trained model shared by all tests (training is the slow part)."""
    from fatigue import evaluation as ev
    from fatigue.models import FusionModel
    W, _ = windows
    D = ev.prepare(W, cfg)
    return FusionModel.from_features(cfg).fit(D, D["y"].to_numpy(), D["subject"].to_numpy())
