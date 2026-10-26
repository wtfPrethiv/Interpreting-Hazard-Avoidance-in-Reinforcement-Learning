"""Render paired cue conditions and persist oracle incentive checks."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from envs.gridworld import HazardGridworld
from envs.scenarios import make_scenario
from evaluation.oracle import route_values
from experiments.config import load_config
from experiments.provenance import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="results/figures")
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(9, 6), constrained_layout=True)
    for row, hazardous in zip(axes, (True, False)):
        for axis, mode in zip(row, ("correlated", "randomised", "reversed")):
            env = HazardGridworld(colour_mode=mode, correlation=1)
            obs, _ = env.reset(options={"scenario": make_scenario(42, hazardous=hazardous)})
            axis.imshow(obs.transpose(1, 2, 0), interpolation="nearest")
            axis.set_title(f"{mode}; hazardous={hazardous}")
            axis.axis("off")
            env.close()
    fig.savefig(output / "environment_check.png", dpi=150)
    plt.close(fig)
    config = load_config()
    values = [
        {
            "family_id": family,
            "hazardous": hazard,
            **route_values(make_scenario(family, hazardous=hazard), config),
        }
        for family in range(256)
        for hazard in (False, True)
    ]
    assert all(
        row["detour"] > row["short"] if row["hazardous"] else row["short"] > row["detour"]
        for row in values
    )
    write_json(
        output / "oracle_checks.json", {"kind": "environment_validation", "route_values": values}
    )
    print(f"Rendered environment and verified {len(values)} route-value comparisons")


if __name__ == "__main__":
    main()
