"""Classification experiment: Lasso feature selection on the Breast Cancer Wisconsin dataset.

The Lasso is used as a linear classifier: it regresses the 0/1 class label
and a sample is assigned to class 1 when the prediction is at least 0.5.
Before the Lasso selection, a correlation filter removes the features that
are almost copies of another one.

Usage:
    python breast_cancer_classification.py
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # figures are written to file, no window is opened

from src.datasets import load_breast_cancer
from src.experiment import run_experiment

OUTPUT_DIR = Path(__file__).resolve().parent / "results" / "breast_cancer"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--folds", type=int, default=5, help="number of cross-validation folds")
    parser.add_argument("--rule", choices=["1se", "min"], default="1se",
                        help="rule used to choose alpha for the feature selection")
    parser.add_argument("--correlation", type=float, default=0.95,
                        help="correlation above which two features are redundant")
    parser.add_argument("--repeats", type=int, default=3, help="repetitions of each timing")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    args = parser.parse_args()

    X, y, feature_names = load_breast_cancer()
    run_experiment("Breast Cancer Wisconsin", X, y, feature_names, task="classification",
                   k=args.folds, selection_rule=args.rule,
                   correlation_threshold=args.correlation, repeats=args.repeats,
                   seed=args.seed, output_dir=OUTPUT_DIR)


if __name__ == "__main__":
    main()
