"""Generate behavioural figures from complete recorded evaluation JSON files."""

import argparse

from evaluation.figures import behavioural_figures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evaluations", nargs="+")
    parser.add_argument("--output-dir", default="results/figures")
    args = parser.parse_args()
    print(behavioural_figures(args.evaluations, args.output_dir))


if __name__ == "__main__":
    main()
