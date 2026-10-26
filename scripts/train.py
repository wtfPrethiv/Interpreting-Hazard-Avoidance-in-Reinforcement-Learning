"""Train an RQ1 policy. Install the repository with pip install -e first."""

import argparse
import json

from agents.train import train
from experiments.config import config_arguments, load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    config_arguments(parser)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--run-id")
    parser.add_argument("--max-seconds", type=float)
    parser.add_argument("--kind", choices=["smoke", "pilot", "main"], default="pilot")
    args = parser.parse_args()
    overrides = args.overrides + ([] if args.seed is None else [f"seed={args.seed}"])
    manifest = train(
        load_config(args.config, overrides),
        run_id=args.run_id,
        max_seconds=args.max_seconds,
        kind=args.kind,
    )
    print(
        json.dumps(
            {key: manifest[key] for key in ("run_id", "status", "actual_steps", "checkpoint")},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
