"""Explicit configuration precedence: base < overlays < dotted overrides."""

from copy import deepcopy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def merge(base, overlay):
    result = deepcopy(base)
    for key, value in overlay.items():
        if key not in result:
            raise ValueError(f"Unknown configuration key: {key}")
        if isinstance(result[key], dict):
            if not isinstance(value, dict):
                raise ValueError(f"Expected mapping for {key}")
            result[key] = merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(paths=(), overrides=(), base=None):
    config = (
        deepcopy(base)
        if base is not None
        else yaml.safe_load((ROOT / "configs/base.yaml").read_text())
    )
    for path in paths:
        config = merge(config, yaml.safe_load(Path(path).read_text()) or {})
    for override in overrides:
        key, separator, raw = override.partition("=")
        if not separator:
            raise ValueError("Overrides must use dotted.key=value")
        node = config
        parts = key.split(".")
        for part in parts[:-1]:
            if part not in node or not isinstance(node[part], dict):
                raise ValueError(f"Unknown configuration key: {key}")
            node = node[part]
        if parts[-1] not in node:
            raise ValueError(f"Unknown configuration key: {key}")
        node[parts[-1]] = yaml.safe_load(raw)
    validate(config)
    return config


def validate(config):
    env, train, evaluation = (config[k] for k in ("environment", "training", "evaluation"))
    if env["grid_size"] != 11 or env["tile_size"] != 8:
        raise ValueError("Version 1 uses an 11x11 grid with 8-pixel tiles")
    if env["colour_mode"] not in {"correlated", "randomised", "reversed"}:
        raise ValueError("Invalid colour mode")
    if not 0 <= env["correlation"] <= 1 or not 0 <= env["hazard_probability"] <= 1:
        raise ValueError("Probabilities must lie in [0, 1]")
    if env["max_steps"] < 1 or env["safety_penalty"] < 0:
        raise ValueError("Invalid episode limit or safety penalty")
    if config["agent"]["algorithm"] != "ppo":
        raise ValueError("Only PPO is implemented")
    if not 0 < train["gamma"] <= 1:
        raise ValueError("gamma must lie in (0, 1]")
    for key in (
        "total_steps",
        "n_envs",
        "n_steps",
        "batch_size",
        "n_epochs",
        "checkpoint_every",
        "torch_threads",
    ):
        if not isinstance(train[key], int) or isinstance(train[key], bool) or train[key] < 1:
            raise ValueError(f"training.{key} must be a positive integer")
    rollout_size = train["n_envs"] * train["n_steps"]
    if train["batch_size"] < 2 or rollout_size % train["batch_size"]:
        raise ValueError("batch_size must divide rollout size and be at least 2")
    if evaluation["split"] not in {"development", "discovery", "final", "train"}:
        raise ValueError("Invalid evaluation split")
    if not evaluation["orientations"] or any(x not in range(8) for x in evaluation["orientations"]):
        raise ValueError("Orientations must be integers from 0 to 7")
    if evaluation["families"] < 1 or evaluation["repeats"] < 1:
        raise ValueError("Evaluation counts must be positive")
    if not set(evaluation["conditions"]) <= {"correlated", "randomised", "reversed"}:
        raise ValueError("Invalid evaluation condition")


def config_arguments(parser):
    parser.add_argument("--config", action="append", default=[], help="YAML overlay; repeatable")
    parser.add_argument(
        "--set", dest="overrides", action="append", default=[], help="dotted.key=value"
    )
