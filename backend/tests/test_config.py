import pytest

from fatigue.config import ConfigError, load_config


def test_defaults_load_and_hash_is_stable(cfg):
    assert cfg.hash() == load_config().hash()
    assert cfg.window.length_s == 3.0 and cfg.extraction.fps == 30.0


def test_unknown_key_rejected(cfg):
    with pytest.raises(ConfigError):
        cfg.replace(window={"lenght_s": 3})


def test_unknown_dict_key_rejected(cfg):
    with pytest.raises(ConfigError):
        cfg.replace(alerts={"persist": {"mildd": 2}})


def test_validation_catches_bad_values(cfg):
    with pytest.raises(ConfigError):
        cfg.replace(window={"stride_s": 10.0})
    with pytest.raises(ConfigError):
        cfg.replace(pretrip={"weights": {"reopen": 0.9}})
    with pytest.raises(ConfigError):
        cfg.replace(alerts={"target_fpr": {"severe": 0.2}})


def test_user_yaml_overlay(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("window:\n  length_s: 5\n  stride_s: 2.5\n")
    c = load_config(p)
    assert c.window.length_s == 5 and c.window.stride_s == 2.5
    assert c.eyes.closed_frac == 0.5      # untouched defaults survive
