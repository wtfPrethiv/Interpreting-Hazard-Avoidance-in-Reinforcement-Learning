"""Finite-horizon planning reference using simulator truth, never policy inputs."""

import numpy as np

from envs.scenarios import DELTAS


def path_actions(path):
    return [DELTAS.index((b[0] - a[0], b[1] - a[1])) for a, b in zip(path, path[1:])]


def route_return(
    scenario, path, *, gamma=0.99, step_reward=-0.02, goal_reward=1.0, safety_penalty=1.0
):
    total = 0.0
    for step, (before, after) in enumerate(zip(path, path[1:])):
        cost = float(after != before and after in scenario.hazard_tiles)
        task = step_reward + goal_reward * (after == scenario.goal)
        total += gamma**step * (task - safety_penalty * cost)
        if after == scenario.goal:
            break
    return total


def plan(
    scenario, *, horizon=100, gamma=0.99, step_reward=-0.02, goal_reward=1.0, safety_penalty=1.0
):
    """Return optimal action values for every remaining time and position."""
    points = tuple(scenario.walkable)
    indices = {point: i for i, point in enumerate(points)}
    transitions = np.empty((len(points), 4), dtype=int)
    rewards = np.empty((len(points), 4), dtype=float)
    terminal = np.empty((len(points), 4), dtype=bool)
    for i, point in enumerate(points):
        for action, (dy, dx) in enumerate(DELTAS):
            target = (point[0] + dy, point[1] + dx)
            target = target if target in indices else point
            transitions[i, action] = indices[target]
            cost = float(target != point and target in scenario.hazard_tiles)
            rewards[i, action] = (
                step_reward + goal_reward * (target == scenario.goal) - safety_penalty * cost
            )
            terminal[i, action] = target == scenario.goal or point == scenario.goal
            if point == scenario.goal:
                rewards[i, action] = 0.0
    values = np.zeros(len(points))
    q = np.zeros((horizon + 1, len(points), 4))
    for remaining in range(1, horizon + 1):
        q[remaining] = rewards + gamma * values[transitions] * ~terminal
        values = q[remaining].max(axis=1)
    return indices, q


def route_values(scenario, config):
    settings = {
        key: config["environment"][key] for key in ("step_reward", "goal_reward", "safety_penalty")
    }
    settings["gamma"] = config["training"]["gamma"]
    return {
        "short": route_return(scenario, scenario.short_path, **settings),
        "detour": route_return(scenario, scenario.detour_path, **settings),
    }
