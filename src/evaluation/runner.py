"""Persist complete evaluation records and their provenance."""

import json
from datetime import datetime, timezone
from pathlib import Path

from evaluation.metrics import paired_differences, readiness, summarise
from evaluation.rollouts import evaluate
from experiments.provenance import file_hash, provenance, seed_everything, write_json


def run_evaluation(
    config, *, checkpoint=None, baseline=None, output=None, max_seconds=None, kind=None
):
    seed_everything(config["seed"], config["training"]["torch_threads"])
    model = None
    if checkpoint:
        from stable_baselines3 import PPO

        model = PPO.load(checkpoint, device=config["training"]["device"])
        if kind is None:
            kind = json.loads((Path(checkpoint).parent / "manifest.json").read_text())["kind"]
    record = evaluate(config, model=model, baseline=baseline, max_seconds=max_seconds)
    summary = summarise(record["episodes"], config["seed"])
    manifest = provenance(config, kind or "reference_baseline")
    manifest.update(
        {
            "checkpoint": str(checkpoint) if checkpoint else None,
            "checkpoint_sha256": file_hash(checkpoint) if checkpoint else None,
            "baseline": baseline,
            "split": config["evaluation"]["split"],
            "uncertainty": "95% bootstrap over layout families within this policy; not training-seed uncertainty",
            "decision_states": "Constructed reachable route-branch states (episode starts)",
            "summary": summary,
            "paired_differences": paired_differences(record["episodes"]),
            "readiness": readiness(summary, record["complete"]),
            **record,
        }
    )
    if output is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
        output = (
            Path(config["results_dir"])
            / "metrics"
            / f"eval_{baseline or Path(checkpoint).parent.name}_{stamp}.json"
        )
    if Path(output).exists():
        raise FileExistsError(f"Refusing to overwrite evaluation: {output}")
    write_json(output, manifest)
    return manifest


def checkpoint_config(checkpoint):
    """Resolve the exact training configuration saved beside a checkpoint."""
    path = Path(checkpoint).parent / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"Training manifest required beside checkpoint: {path}")
    return json.loads(path.read_text())["config"]
