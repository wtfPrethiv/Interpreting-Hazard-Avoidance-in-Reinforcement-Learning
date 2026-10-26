"""Four-action gridworld with separate task rewards and safety costs."""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from envs.observations import render_observation, tile_colours
from envs.scenarios import DELTAS, Scenario, family_ids, make_scenario


class HazardGridworld(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 8}

    def __init__(
        self,
        *,
        grid_size=11,
        tile_size=8,
        max_steps=100,
        goal_reward=1.0,
        step_reward=-0.02,
        safety_penalty=1.0,
        colour_mode="correlated",
        correlation=0.95,
        hazard_probability=0.5,
        split="train",
        render_mode=None,
    ):
        if (grid_size, tile_size) != (11, 8) or max_steps < 1 or safety_penalty < 0:
            raise ValueError("Invalid grid size, tile size, episode limit, or safety penalty")
        if not 0 <= correlation <= 1 or not 0 <= hazard_probability <= 1:
            raise ValueError("Probabilities must lie in [0, 1]")
        if colour_mode not in {"correlated", "randomised", "reversed"}:
            raise ValueError("Invalid colour mode")
        if render_mode not in {None, "rgb_array"}:
            raise ValueError("Only rgb_array rendering is supported")
        self.max_steps, self.goal_reward = max_steps, goal_reward
        self.step_reward, self.safety_penalty = step_reward, safety_penalty
        self.colour_mode, self.correlation = colour_mode, correlation
        self.hazard_probability, self.split = hazard_probability, split
        self.ids = family_ids(split)
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(0, 255, shape=(3, 88, 88), dtype=np.uint8)
        self.scenario = None
        self.done = True

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        scenario = options.get("scenario")
        if scenario is None:
            scenario = make_scenario(
                int(self.np_random.choice(self.ids)),
                int(self.np_random.integers(8)),
                self.np_random.random() < self.hazard_probability,
                int(self.np_random.integers(2**32)),
            )
        if not isinstance(scenario, Scenario):
            raise TypeError("options.scenario must be a Scenario")
        self.scenario = scenario
        self.walkable = frozenset(scenario.walkable)
        self.position = scenario.start
        self.colours = tile_colours(scenario, self.colour_mode, self.correlation)
        self.steps = 0
        self.task_return = self.learning_return = self.safety_cost = 0.0
        self.visited_shortcut = self.visited_detour = False
        self.done = False
        return self._observation(), self._info(0.0, 0.0, 0.0)

    def step(self, action):
        if self.done:
            raise RuntimeError("Call reset before stepping a finished episode")
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")
        dy, dx = DELTAS[int(action)]
        candidate = (self.position[0] + dy, self.position[1] + dx)
        moved = candidate in self.walkable
        if moved:
            self.position = candidate
        cost = float(moved and self.position in self.scenario.hazard_tiles)
        self.steps += 1
        terminated = self.position == self.scenario.goal
        truncated = self.steps >= self.max_steps and not terminated
        task = self.step_reward + self.goal_reward * terminated
        reward = task - self.safety_penalty * cost
        self.task_return += task
        self.learning_return += reward
        self.safety_cost += cost
        self.visited_shortcut |= self.position == self.scenario.shortcut_tile
        self.visited_detour |= self.position == self.scenario.comparison_tile
        self.done = terminated or truncated
        return self._observation(), reward, terminated, truncated, self._info(task, cost, reward)

    def _observation(self):
        return render_observation(self.scenario, self.position, self.colours)

    def _info(self, task, cost, reward):
        return {
            "scenario_id": self.scenario.scenario_id,
            "family_id": self.scenario.family_id,
            "orientation": self.scenario.orientation,
            "hazardous": self.scenario.hazardous,
            "position": self.position,
            "task_reward": task,
            "safety_cost": cost,
            "learning_reward": reward,
            "task_return": self.task_return,
            "learning_return": self.learning_return,
            "episode_safety_cost": self.safety_cost,
            "success": self.position == self.scenario.goal,
            "visited_shortcut": self.visited_shortcut,
            "visited_detour": self.visited_detour,
        }

    def render(self):
        if self.scenario is None:
            raise RuntimeError("Call reset before rendering")
        return self._observation().transpose(1, 2, 0)
