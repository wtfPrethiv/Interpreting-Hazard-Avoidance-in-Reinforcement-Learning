"""Run one bounded behavioural pilot per condition; never run main experiments."""

import argparse
import gc
import tempfile
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import torch

from agents.train import BoundedCheckpoint, make_model, train
from evaluation.figures import behavioural_figures
from evaluation.runner import run_evaluation
from experiments.config import config_arguments, load_config
from experiments.provenance import provenance, write_json


def benchmark(config):
    results = {}
    for device in ["cpu"] + (["cuda"] if torch.cuda.is_available() else []):
        model = None
        try:
            started = time.monotonic()
            model = make_model(config, device=device)
            with tempfile.TemporaryDirectory() as directory:
                callback = BoundedCheckpoint(directory, 10**9, seconds=30)
                model.learn(total_timesteps=2048, callback=callback)
            if device == "cuda":
                torch.cuda.synchronize()
            elapsed = time.monotonic() - started
            results[device] = {
                "steps": model.num_timesteps,
                "elapsed_seconds": elapsed,
                "steps_per_second": model.num_timesteps / elapsed,
            }
        except (RuntimeError, OSError) as error:
            results[device] = {"error": repr(error)}
        finally:
            if model is not None:
                model.get_env().close()
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    valid = {key: value for key, value in results.items() if "steps_per_second" in value}
    if not valid:
        raise RuntimeError(f"No working training device: {results}")
    return max(valid, key=lambda key: valid[key]["steps_per_second"]), results


def run_pilots(config, budget_seconds=7200):
    if not 120 <= budget_seconds <= 7200:
        raise ValueError("Pilot budget must be between 120 and 7200 seconds")
    if config["evaluation"]["split"] != "development":
        raise ValueError("Pilots must use development evaluation only")
    config = deepcopy(config)
    config["training"]["total_steps"] = min(config["training"]["total_steps"], 1000000)
    started = time.monotonic()
    deadline = started + budget_seconds
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    root = Path(config["results_dir"]).resolve()
    report_path = root / "manifests" / f"pilot_suite_{stamp}.json"
    report = provenance(config, "pilot_suite")
    report.update(
        {"status": "running", "budget_seconds": budget_seconds, "runs": [], "baselines": []}
    )
    write_json(report_path, report)
    try:
        device, timings = benchmark(config)
        config["training"]["device"] = device
        report.update({"benchmark": timings, "selected_device": device})
        print(f"Selected {device}: {timings}", flush=True)
        # Reserve time for two evaluations, baseline references, and output generation.
        reserve = min(900, budget_seconds * 0.2)
        equal_training_seconds = max(1, (deadline - time.monotonic() - reserve) / 2)
        for mode in ("correlated", "randomised"):
            condition = deepcopy(config)
            condition["environment"]["colour_mode"] = mode
            run_id = f"pilot_{mode}_s{config['seed']}_{stamp}"
            manifest = train(
                condition, run_id=run_id, max_seconds=equal_training_seconds, kind="pilot"
            )
            eval_path = root / "metrics" / f"{run_id}.json"
            remaining = max(0, deadline - time.monotonic())
            evaluation = run_evaluation(
                condition,
                checkpoint=manifest["checkpoint"],
                output=eval_path,
                max_seconds=min(reserve / 4, remaining),
            )
            report["runs"].append(
                {
                    "run_id": run_id,
                    "checkpoint": manifest["checkpoint"],
                    "training_steps": manifest["actual_steps"],
                    "training_status": manifest["status"],
                    "evaluation": str(eval_path),
                    "readiness": evaluation["readiness"],
                }
            )
            write_json(report_path, report)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        for baseline in ("random", "shortest", "colour", "symbol", "oracle"):
            remaining = deadline - time.monotonic()
            if remaining <= 5:
                break
            path = root / "metrics" / f"baseline_{baseline}_{stamp}.json"
            result = run_evaluation(
                config, baseline=baseline, output=path, max_seconds=min(reserve / 10, remaining)
            )
            report["baselines"].append(
                {"name": baseline, "evaluation": str(path), "complete": result["complete"]}
            )
        if all(run["readiness"]["checks"]["evaluation_complete"] for run in report["runs"]):
            report["figure"] = behavioural_figures(
                [run["evaluation"] for run in report["runs"]], root / "figures"
            )
        report["ready_for_protocol_review"] = all(
            run["readiness"]["ready"] for run in report["runs"]
        )
        report["status"] = "completed" if time.monotonic() < deadline else "budget_exhausted"
    except BaseException as error:
        report.update({"status": "failed", "error": repr(error)})
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        write_json(report_path, report)
    print(f"Pilot suite recorded at {report_path}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    config_arguments(parser)
    parser.add_argument("--budget-seconds", type=float, default=7200)
    args = parser.parse_args()
    run_pilots(load_config(args.config, args.overrides), args.budget_seconds)


if __name__ == "__main__":
    main()
