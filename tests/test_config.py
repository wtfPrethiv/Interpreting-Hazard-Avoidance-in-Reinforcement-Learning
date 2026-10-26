import pytest

from experiments.config import ROOT, load_config


def test_precedence_and_unknown_keys():
    config = load_config([ROOT / "configs/rq1_randomised.yaml"], ["seed=42"])
    assert config["seed"] == 42
    assert config["environment"]["colour_mode"] == "randomised"
    with pytest.raises(ValueError, match="Unknown"):
        load_config(overrides=["training.typo=1"])
    with pytest.raises(ValueError):
        load_config(overrides=["training.batch_size=17"])
