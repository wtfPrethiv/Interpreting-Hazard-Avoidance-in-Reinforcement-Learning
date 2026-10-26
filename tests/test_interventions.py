"""Appearance interventions only. Neural interventions are deferred."""

import numpy as np

from envs.gridworld import HazardGridworld
from envs.interventions import appearance_config
from envs.observations import BLUE, RED, tile_colours
from envs.scenarios import make_scenario
from evaluation.oracle import path_actions
from experiments.config import load_config


def test_appearance_only_preserves_complete_trajectory():
    scenario = make_scenario(10, orientation=3)
    records, observations = [], []
    for mode in ("correlated", "randomised", "reversed"):
        env = HazardGridworld(colour_mode=mode, correlation=1)
        obs, _ = env.reset(options={"scenario": scenario})
        observations.append(obs)
        record = []
        for action in path_actions(scenario.short_path):
            _, reward, terminated, truncated, info = env.step(action)
            record.append((reward, terminated, truncated, info))
        records.append(record)
    assert records[0] == records[1] == records[2]
    assert not np.array_equal(observations[0], observations[2])


def test_symbol_and_colour_independence():
    for hazardous in (False, True):
        scenario = make_scenario(0, hazardous=hazardous)
        normal = tile_colours(scenario, "correlated", 1)
        reversed_colours = tile_colours(scenario, "reversed")
        assert np.array_equal(normal[scenario.shortcut_tile], RED if hazardous else BLUE)
        assert not np.array_equal(
            normal[scenario.shortcut_tile], reversed_colours[scenario.shortcut_tile]
        )
        images = []
        for mode in ("correlated", "reversed"):
            env = HazardGridworld(colour_mode=mode, correlation=1)
            image, _ = env.reset(options={"scenario": scenario})
            y, x = scenario.shortcut_tile
            images.append(image[:, y * 8 : (y + 1) * 8, x * 8 : (x + 1) * 8])
        assert np.array_equal(np.all(images[0] == 255, axis=0), np.all(images[1] == 255, axis=0))


def test_randomised_colours_do_not_depend_on_hazard_status():
    for seed in range(100):
        safe = make_scenario(0, hazardous=False, appearance_seed=seed)
        hazard = make_scenario(0, hazardous=True, appearance_seed=seed)
        assert all(
            np.array_equal(
                tile_colours(safe, "randomised")[point], tile_colours(hazard, "randomised")[point]
            )
            for point in (safe.shortcut_tile, safe.comparison_tile)
        )


def test_appearance_config_does_not_mutate_source():
    config = load_config()["environment"]
    changed = appearance_config(config, "reversed")
    assert config["colour_mode"] == "correlated"
    assert {k: v for k, v in config.items() if k != "colour_mode"} == {
        k: v for k, v in changed.items() if k != "colour_mode"
    }
