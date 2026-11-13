"""Regression experiment: Lasso feature selection on the California Housing dataset.

Usage:
    python california_housing_regression.py
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # figures are written to file, no window is opened

from src.datasets import CALIFORNIA_CSV, load_california_housing
from src.experiment import run_experiment

OUTPUT_DIR = Path(__file__).resolve().parent / "results" / "california_housing"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", default=CALIFORNIA_CSV, help="path of the CSV file")
    parser.add_argument("--folds", type=int, default=5, help="number of cross-validation folds")
    parser.add_argument("--rule", choices=["1se", "min"], default="1se",
                        help="rule used to choose alpha for the feature selection")
    parser.add_argument("--correlation", type=float, default=None,
                        help="optional correlation filter: threshold above which two "
                             "features are redundant (not used by default)")
    parser.add_argument("--repeats", type=int, default=3, help="repetitions of each timing")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    args = parser.parse_args()

    X, y, feature_names = load_california_housing(args.data)
    run_experiment("California Housing", X, y, feature_names, task="regression",
                   k=args.folds, selection_rule=args.rule,
                   correlation_threshold=args.correlation, repeats=args.repeats,
                   seed=args.seed, output_dir=OUTPUT_DIR)


if __name__ == "__main__":
    main()
