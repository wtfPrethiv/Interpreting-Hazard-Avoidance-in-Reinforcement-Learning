"""Simulator contracts and incentive checks, independent of learned behaviour."""

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from envs.gridworld import HazardGridworld
from envs.scenarios import (
    DELTAS,
    evaluation_cases,
    families,
    family_ids,
    make_scenario,
    scenario_manifest,
)
from evaluation.oracle import path_actions, plan, route_return, route_values
from experiments.config import load_config


def test_gymnasium_contract():
    check_env(HazardGridworld(), skip_render_check=True)


def test_split_geometry_and_seed_reproducibility():
    manifest = scenario_manifest()
    assert len(families()) == 256
    assert len({entry["geometry_hash"] for entry in manifest["families"]}) == 256
    assert sum(map(len, manifest["split_families"].values())) == 256
    assert set(family_ids("train")).isdisjoint(family_ids("development"))
    a, b = HazardGridworld(), HazardGridworld()
    for seed in range(10):
        obs_a, info_a = a.reset(seed=seed)
        obs_b, info_b = b.reset(seed=seed)
        np.testing.assert_array_equal(obs_a, obs_b)
        assert info_a == info_b


def test_routes_and_incentives_all_families():
    config = load_config()
    for family_id in range(256):
        for hazardous in (False, True):
            scenario = make_scenario(family_id, family_id % 8, hazardous)
            env = HazardGridworld()
            values = route_values(scenario, config)
            assert (
                values["detour"] > values["short"]
                if hazardous
                else values["short"] > values["detour"]
            )
            assert values["detour"] > 0  # Progress is preferable to timing out.
            for path in (scenario.short_path, scenario.detour_path):
                env.reset(options={"scenario": scenario})
                discounted = 0
                for index, action in enumerate(path_actions(path)):
                    _, reward, terminated, truncated, info = env.step(action)
                    discounted += 0.99**index * reward
                assert terminated and not truncated
                assert info["success"]
                assert discounted == pytest.approx(route_return(scenario, path))
                assert info["episode_safety_cost"] == float(
                    hazardous and path == scenario.short_path
                )


def test_oracle_matches_enumerated_routes():
    for family in range(0, 256, 16):
        for hazardous in (False, True):
            scenario = make_scenario(family, hazardous=hazardous)
            indices, q = plan(scenario, horizon=30)
            expected = max(
                route_return(scenario, p) for p in (scenario.short_path, scenario.detour_path)
            )
            assert q[30, indices[scenario.start]].max() == pytest.approx(expected)


def test_hazard_reentry_blocked_action_and_accounting():
    scenario = make_scenario(0)
    env = HazardGridworld()
    env.reset(options={"scenario": scenario})
    index = scenario.short_path.index(scenario.shortcut_tile)
    actions = path_actions(scenario.short_path[: index + 1])
    for action in actions:
        _, reward, _, _, info = env.step(action)
    assert info["safety_cost"] == 1
    assert reward == pytest.approx(-1.02)
    blocked = next(
        i
        for i, (dy, dx) in enumerate(DELTAS)
        if (env.position[0] + dy, env.position[1] + dx) not in env.walkable
    )
    assert env.step(blocked)[-1]["safety_cost"] == 0
    previous = scenario.short_path[index - 1]
    back = DELTAS.index((previous[0] - env.position[0], previous[1] - env.position[1]))
    env.step(back)
    _, _, _, _, info = env.step(actions[-1])
    assert info["episode_safety_cost"] == 2
    assert info["learning_return"] == pytest.approx(info["task_return"] - 2)


def test_time_limit_terminal_and_invalid_actions():
    scenario = make_scenario(0)
    env = HazardGridworld(max_steps=1)
    env.reset(options={"scenario": scenario})
    with pytest.raises(ValueError):
        env.step(4)
    _, _, terminated, truncated, _ = env.step(0)
    assert truncated and not terminated
    with pytest.raises(RuntimeError):
        env.step(0)
    env = HazardGridworld(max_steps=len(scenario.short_path) - 1)
    env.reset(options={"scenario": scenario})
    for action in path_actions(scenario.short_path):
        result = env.step(action)
    assert result[2] and not result[3]


def test_image_and_balanced_cases():
    env = HazardGridworld()
    obs, _ = env.reset(seed=123)
    assert obs.shape == (3, 88, 88) and obs.dtype == np.uint8
    assert env.observation_space.contains(obs)
    assert env.render().shape == (88, 88, 3)
    cases = list(evaluation_cases(count=2))
    assert len(cases) == 32
    assert sum(case.hazardous for case in cases) == 16
