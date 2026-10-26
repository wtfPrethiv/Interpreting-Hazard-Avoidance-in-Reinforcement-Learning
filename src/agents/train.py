"""Established PPO implementation, bounded runs, and complete provenance."""

import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from agents.policy import activation_points, policy_kwargs
from envs.gridworld import HazardGridworld
from envs.scenarios import scenario_manifest
from experiments.provenance import digest, file_hash, provenance, seed_everything, write_json


def make_model(config, device=None, monitor_dir=None):
    train = config["training"]
    seed_everything(config["seed"], train["torch_threads"])

    def factory(rank):
        def create():
            path = str(Path(monitor_dir) / f"env_{rank}") if monitor_dir else None
            return Monitor(
                HazardGridworld(**config["environment"]),
                filename=path,
                info_keywords=("task_return", "episode_safety_cost", "success"),
            )

        return create

    env = DummyVecEnv([factory(i) for i in range(train["n_envs"])])
    env.seed(config["seed"])
    settings = {
        k: train[k]
        for k in (
            "n_steps",
            "batch_size",
            "n_epochs",
            "learning_rate",
            "gamma",
            "gae_lambda",
            "clip_range",
            "ent_coef",
        )
    }
    model = PPO(
        "CnnPolicy",
        env,
        seed=config["seed"],
        device=device or train["device"],
        policy_kwargs=policy_kwargs(config["agent"]["features_dim"]),
        verbose=0,
        **settings,
    )
    return model


class BoundedCheckpoint(BaseCallback):
    def __init__(self, directory, interval, seconds=None):
        super().__init__()
        self.directory, self.interval, self.seconds = Path(directory), interval, seconds
        self.next_checkpoint = interval
        self.started = time.monotonic()
        self.stopped_by_time = False
        self.checkpoints = []
        self.last_report = self.started

    def save(self, name):
        path = self.directory / name
        self.model.save(path)
        self.checkpoints.append(
            {"path": str(path), "sha256": file_hash(path), "steps": self.num_timesteps}
        )

    def _on_step(self):
        now = time.monotonic()
        if self.num_timesteps >= self.next_checkpoint:
            self.save(f"step_{self.num_timesteps}.zip")
            self.next_checkpoint += self.interval
        if now - self.last_report >= 30:
            print(f"Training: {self.num_timesteps:,} steps, {now - self.started:.0f}s", flush=True)
            self.last_report = now
        if self.seconds is not None and now - self.started >= self.seconds:
            self.stopped_by_time = True
            return False
        return True


def train(config, *, run_id=None, max_seconds=None, kind="pilot"):
    if config["environment"]["split"] != "train":
        raise ValueError("Training is restricted to the train split")
    if max_seconds is not None and max_seconds <= 0:
        raise ValueError("max_seconds must be positive")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    run_id = run_id or f"{kind}_{config['environment']['colour_mode']}_s{config['seed']}_{stamp}"
    if Path(run_id).name != run_id or run_id in {".", ".."}:
        raise ValueError("run_id must be a single directory name")
    root = Path(config["results_dir"]).resolve()
    directory = root / "runs" / run_id
    directory.mkdir(parents=True, exist_ok=False)
    manifest_path = root / "manifests" / f"{run_id}.json"
    manifest = provenance(config, kind)
    scenarios = scenario_manifest()
    manifest.update(
        {
            "run_id": run_id,
            "status": "running",
            "scenario_version": scenarios["version"],
            "scenario_manifest_sha256": digest(scenarios),
            "environment_seeds": [config["seed"] + i for i in range(config["training"]["n_envs"])],
            "max_seconds": max_seconds,
        }
    )
    write_json(root / "manifests" / "scenarios-v1.json", scenarios)
    write_json(manifest_path, manifest)
    model = None
    started = time.monotonic()
    try:
        model = make_model(config, monitor_dir=directory)
        manifest.update(
            {
                "device": str(model.device),
                "gpu": torch.cuda.get_device_name(0) if model.device.type == "cuda" else None,
                "parameters": sum(p.numel() for p in model.policy.parameters()),
                "activation_points": list(activation_points(model.policy)),
                "encoder_shared_actor_critic": True,
            }
        )
        callback = BoundedCheckpoint(
            directory,
            config["training"]["checkpoint_every"],
            None if max_seconds is None else max(0, max_seconds - (time.monotonic() - started)),
        )
        rollout = config["training"]["n_envs"] * config["training"]["n_steps"]
        steps = config["training"]["total_steps"] // rollout * rollout
        if steps < rollout:
            raise ValueError("total_steps must allow at least one rollout")
        model.learn(total_timesteps=steps, callback=callback)
        callback.save("final.zip")
        manifest.update(
            {
                "status": "time_limited" if callback.stopped_by_time else "completed",
                "actual_steps": model.num_timesteps,
                "checkpoints": callback.checkpoints,
                "checkpoint": str(directory / "final.zip"),
                "checkpoint_sha256": file_hash(directory / "final.zip"),
            }
        )
    except BaseException as error:
        manifest.update({"status": "failed", "error": repr(error)})
        raise
    finally:
        manifest["elapsed_seconds"] = time.monotonic() - started
        write_json(manifest_path, manifest)
        write_json(directory / "manifest.json", manifest)
        if model is not None:
            model.get_env().close()
    return manifest
