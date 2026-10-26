"""Frozen-policy paired rollouts; simulator metadata is never sent to PPO."""

import time

import numpy as np

from envs.gridworld import HazardGridworld
from envs.interventions import appearance_config
from envs.observations import RED
from envs.scenarios import DELTAS, evaluation_cases
from evaluation.oracle import plan, route_values

BASELINES = ("random", "shortest", "colour", "symbol", "oracle")


def baseline_probabilities(name, env, oracle=None):
    probabilities = np.zeros(4)
    if name == "random":
        return np.full(4, 0.25)
    if name == "oracle":
        indices, q = oracle
        action = int(q[env.max_steps - env.steps, indices[env.position]].argmax())
    else:
        scenario = env.scenario
        avoid = (name == "symbol" and scenario.hazardous) or (
            name == "colour" and np.array_equal(env.colours[scenario.shortcut_tile], RED)
        )
        path = scenario.detour_path if avoid else scenario.short_path
        # Reference policies follow one known route; they are privileged baselines.
        index = path.index(env.position)
        target = path[index + 1]
        action = DELTAS.index((target[0] - env.position[0], target[1] - env.position[1]))
    probabilities[action] = 1
    return probabilities


def evaluate(config, *, model=None, baseline=None, max_seconds=None):
    if (model is None) == (baseline is None):
        raise ValueError("Supply exactly one model or baseline")
    if baseline is not None and baseline not in BASELINES:
        raise ValueError(f"Unknown baseline {baseline}")
    settings = config["evaluation"]
    cases = list(
        evaluation_cases(
            settings["split"],
            settings["families"],
            settings["orientations"],
            settings["repeats"],
            config["seed"],
        )
    )
    records, trajectories = [], []
    started = time.monotonic()
    complete = True
    for condition in settings["conditions"]:
        envs, observations, rngs, oracles = [], [], [], []
        for scenario in cases:
            env_settings = appearance_config(config["environment"], condition)
            env_settings["split"] = settings["split"]
            env = HazardGridworld(**env_settings)
            obs, _ = env.reset(options={"scenario": scenario})
            envs.append(env)
            observations.append(obs)
            # Identical random-number streams across appearance conditions.
            rngs.append(np.random.default_rng(scenario.appearance_seed))
            if baseline == "oracle":
                oracles.append(
                    plan(
                        scenario,
                        horizon=env.max_steps,
                        gamma=config["training"]["gamma"],
                        step_reward=env.step_reward,
                        goal_reward=env.goal_reward,
                        safety_penalty=env.safety_penalty,
                    )
                )
            else:
                oracles.append(None)
        active = list(range(len(envs)))
        first_probabilities, routes, action_lists = {}, {}, [[] for _ in envs]
        discounted_task = np.zeros(len(envs))
        discounted_learning = np.zeros(len(envs))
        for step in range(config["environment"]["max_steps"]):
            if max_seconds is not None and time.monotonic() - started >= max_seconds:
                complete = False
                break
            if not active:
                break
            if model is not None:
                from agents.policy import action_probabilities

                probs = action_probabilities(model, np.stack([observations[i] for i in active]))
            else:
                probs = np.stack(
                    [baseline_probabilities(baseline, envs[i], oracles[i]) for i in active]
                )
            next_active = []
            for i, probabilities in zip(active, probs):
                env, scenario = envs[i], cases[i]
                if step == 0:
                    first_probabilities[i] = probabilities.tolist()
                stochastic = settings["stochastic"] or baseline == "random"
                action = (
                    int(rngs[i].choice(4, p=probabilities / probabilities.sum()))
                    if stochastic
                    else int(probabilities.argmax())
                )
                obs, reward, terminated, truncated, info = env.step(action)
                if i not in routes and env.position != scenario.start:
                    routes[i] = "short" if env.position == scenario.short_path[1] else "detour"
                action_lists[i].append(action)
                observations[i] = obs
                discount = config["training"]["gamma"] ** step
                discounted_task[i] += discount * info["task_reward"]
                discounted_learning[i] += discount * reward
                if not (terminated or truncated):
                    next_active.append(i)
                    continue
                short_action = DELTAS.index(
                    (
                        scenario.short_path[1][0] - scenario.start[0],
                        scenario.short_path[1][1] - scenario.start[1],
                    )
                )
                detour_action = DELTAS.index(
                    (
                        scenario.detour_path[1][0] - scenario.start[0],
                        scenario.detour_path[1][1] - scenario.start[1],
                    )
                )
                row = {
                    "scenario_id": scenario.scenario_id,
                    "family_id": scenario.family_id,
                    "orientation": scenario.orientation,
                    "hazardous": scenario.hazardous,
                    "appearance_seed": scenario.appearance_seed,
                    "condition": condition,
                    "success": bool(terminated),
                    "hazard_entry": info["episode_safety_cost"] > 0,
                    "safety_cost": info["episode_safety_cost"],
                    "task_return": info["task_return"],
                    "learning_return": info["learning_return"],
                    "episode_length": env.steps,
                    "discounted_task_return": float(discounted_task[i]),
                    "discounted_learning_return": float(discounted_learning[i]),
                    "first_route": routes.get(i, "none"),
                    "visited_shortcut": info["visited_shortcut"],
                    "visited_detour": info["visited_detour"],
                    "short_route_success": bool(
                        terminated and info["visited_shortcut"] and not info["visited_detour"]
                    ),
                    "decision_action_probabilities": first_probabilities[i],
                    "decision_short_probability": first_probabilities[i][short_action],
                    "decision_detour_probability": first_probabilities[i][detour_action],
                    "oracle_route_values": route_values(scenario, config),
                }
                records.append(row)
                trajectories.append(
                    {
                        "scenario_id": scenario.scenario_id,
                        "condition": condition,
                        "actions": action_lists[i],
                    }
                )
            active = next_active
        for env in envs:
            env.close()
        if not complete:
            break
    expected = len(cases) * len(settings["conditions"])
    return {
        "episodes": records,
        "trajectories": trajectories,
        "complete": complete and len(records) == expected,
        "expected_episodes": expected,
        "completed_episodes": len(records),
        "elapsed_seconds": time.monotonic() - started,
    }
