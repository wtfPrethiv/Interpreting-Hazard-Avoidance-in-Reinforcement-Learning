"""Evaluate a frozen policy or privileged reference baseline on paired cases."""

import argparse
import json

from evaluation.rollouts import BASELINES
from evaluation.runner import checkpoint_config, run_evaluation
from experiments.config import config_arguments, load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    config_arguments(parser)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--checkpoint")
    group.add_argument("--baseline", choices=BASELINES)
    parser.add_argument("--output")
    parser.add_argument("--max-seconds", type=float)
    args = parser.parse_args()
    if args.checkpoint:
        config = load_config(args.config, args.overrides, base=checkpoint_config(args.checkpoint))
    else:
        config = load_config(args.config, args.overrides)
    result = run_evaluation(
        config,
        checkpoint=args.checkpoint,
        baseline=args.baseline,
        output=args.output,
        max_seconds=args.max_seconds,
    )
    print(
        json.dumps(
            {k: result[k] for k in ("completed_episodes", "complete", "readiness")}, indent=2
        )
    )


if __name__ == "__main__":
    main()
