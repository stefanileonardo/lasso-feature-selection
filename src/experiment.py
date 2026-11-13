"""End-to-end feature selection experiment, shared by the two datasets.

1. Split the data into a training set and a test set.
2. Train the predictor (Lasso tuned by K-fold cross-validation) on all the features.
3. Optionally remove redundant features with a correlation filter and retrain.
4. Select the features with the Lasso and retrain the predictor on them only.
5. Compare test performance and training time of the models.
6. Repeat the retraining for every number of features met along the Lasso path.

The test set is only used to report the final performance: it never takes
part in training, in the choice of ``alpha`` or in the selection of the
features.
"""

from pathlib import Path

import numpy as np

from .feature_selection import (evaluate, feature_ranking, select_features,
                                selection_sweep, timed, train_lasso_cv)
from .plots import plot_coefficient_path, plot_cv_curve, plot_tradeoff
from .preprocessing import Standardizer, drop_correlated_features, train_test_split

RULE_NAMES = {"min": "lowest validation error", "1se": "one-standard-error rule"}


def _metrics_text(metrics, task):
    if task == "classification":
        return f"{100 * metrics['accuracy']:8.2f} %   {metrics['errors']:6d}"
    return f"{metrics['mse']:10.4f}   {metrics['r2']:6.4f}"


def run_experiment(name, X, y, feature_names, task="regression", test_size=0.3,
                   k=5, selection_rule="1se", correlation_threshold=None,
                   repeats=3, seed=0, output_dir=None):
    """Run the whole experiment, print a report and save report and figures.

    Parameters
    ----------
    name : str
        Title of the experiment (used in the report and in the figures).
    X, y, feature_names :
        Dataset: feature matrix, target vector and names of the columns.
    task : {"regression", "classification"}
        For classification ``y`` must contain the labels 0 and 1.
    test_size : float
        Fraction of the samples held out as test set.
    k : int
        Number of cross-validation folds.
    selection_rule : {"1se", "min"}
        Rule that fixes how strongly the Lasso regularises when it selects
        the features (see :func:`src.cross_validation.select_alpha`).
    correlation_threshold : float or None
        If given, features whose absolute correlation with another feature
        exceeds this value are removed before the Lasso selection.
    repeats : int
        Number of repetitions of every time measurement (the best is kept).
    seed : int
        Seed of the train/test split and of the cross-validation folds.
    output_dir : path or None
        Folder where ``summary.txt`` and the figures are written.

    Returns
    -------
    dict with the trained models, the selected features and the sweep table.
    """
    report = []

    def log(line=""):
        print(line)
        report.append(line)

    names = np.array(feature_names)
    y = np.asarray(y, dtype=float)
    train_options = dict(k=k, seed=seed)
    rule_name = RULE_NAMES[selection_rule]

    # ------------------------------------------------------------------ data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, seed=seed, stratify=(task == "classification"))

    if task == "regression":
        # Express the target in units of its standard deviation (statistics of
        # the training set), so that an MSE of 1 means "no better than always
        # predicting the mean".
        y_scaler = Standardizer().fit(y_train)
        y_train, y_test = y_scaler.transform(y_train), y_scaler.transform(y_test)

    log(f"{name} ({task})")
    log("=" * len(report[0]))
    log(f"Training samples: {X_train.shape[0]}   Test samples: {X_test.shape[0]}   "
        f"Features: {X_train.shape[1]}")
    log(f"Alpha chosen by {k}-fold cross-validation on the training set; "
        f"training times are the best of {repeats} runs.")
    log()

    # ------------------------------------------ predictor on all the features
    stages = []  # (label, column indices, trained model, training time)

    all_features = np.arange(X_train.shape[1])
    model, seconds = timed(train_lasso_cv, X_train, y_train, repeats=repeats, **train_options)
    stages.append(("All features", all_features, model, seconds))

    # ------------------------------------------------- correlation filter
    current = all_features
    if correlation_threshold is not None:
        current = drop_correlated_features(X_train, correlation_threshold)
        model, seconds = timed(train_lasso_cv, X_train[:, current], y_train,
                               repeats=repeats, **train_options)
        stages.append((f"Correlation filter (|r| >= {correlation_threshold})",
                       current, model, seconds))
    selection_model = model  # model whose cross-validation curve drives the selection

    # ------------------------------------------------ Lasso feature selection
    picked, alpha_selection = select_features(selection_model, X_train[:, current], y_train,
                                              rule=selection_rule)
    if picked.size == 0:  # the rule is too aggressive for this dataset: fall back
        selection_rule, rule_name = "min", RULE_NAMES["min"]
        picked, alpha_selection = select_features(selection_model, X_train[:, current],
                                                  y_train, rule="min")
    selected = current[picked]

    model, seconds = timed(train_lasso_cv, X_train[:, selected], y_train,
                           repeats=repeats, **train_options)
    stages.append((f"Lasso selection ({selection_rule})", selected, model, seconds))

    # ------------------------------------------------------------ comparison
    metric_header = "accuracy   errors" if task == "classification" else "  test MSE   test R2"
    log(f"{'Model':<34}{'features':>9}{'non-zero':>10}{'alpha':>10}{'alphas':>8}"
        f"   {metric_header}   training time")
    baseline_seconds = stages[0][3]
    results = []
    for label, columns, model, seconds in stages:
        metrics = evaluate(model, X_test[:, columns], y_test, task)
        change = "" if seconds == baseline_seconds else \
            f"  ({100 * (seconds / baseline_seconds - 1):+.0f} %)"
        log(f"{label:<34}{columns.size:>9}{np.count_nonzero(model.beta):>10}{model.alpha:>10.5f}"
            f"{model.cv.alphas.size:>8}"
            f"   {_metrics_text(metrics, task)}   {seconds:8.3f} s{change}")
        results.append({"label": label, "features": columns, "model": model,
                        "time": seconds, **metrics})
    log("(features: columns given to the model; non-zero: coefficients kept by the Lasso;")
    log(" alphas: values of alpha evaluated by cross-validation before the early stop)")
    log()

    if correlation_threshold is not None:
        removed = np.setdiff1d(all_features, current)
        log(f"Removed by the correlation filter ({removed.size}): {', '.join(names[removed])}")
        log()

    log(f"Features selected by the Lasso with alpha = {alpha_selection:.5f} "
        f"({rule_name}): {selected.size} of {current.size}")
    final_model = stages[-1][2]
    for j in np.argsort(-np.abs(final_model.beta)):
        log(f"  {names[selected][j]:<28}{final_model.beta[j]:+9.4f}")
    discarded = np.setdiff1d(current, selected)
    if discarded.size:
        log(f"Discarded ({discarded.size}): {', '.join(names[discarded])}")
    log("(coefficients of the retrained model, for standardised features)")
    log()

    # ------------------------------------------------- ranking of the features
    order, path_alphas, path_betas = feature_ranking(X_train[:, current], y_train)
    log("Features in order of entry in the Lasso path (most important first):")
    log("  " + ", ".join(names[current][order]))
    log()

    # --------------------------- performance and time vs number of features
    sweep = selection_sweep(X_train[:, current], y_train, X_test[:, current], y_test,
                            task=task, repeats=repeats, **train_options)
    log("Predictor retrained on every feature subset met along the Lasso path:")
    log(f"{'features':>9}   {metric_header}   training time   features added")
    previous = np.array([], dtype=int)
    for row in sweep:
        added = np.setdiff1d(row["features"], previous)
        dropped = np.setdiff1d(previous, row["features"])
        note = "+ " + ", ".join(names[current][added])
        if dropped.size:
            note += "   - " + ", ".join(names[current][dropped])
        log(f"{row['n_features']:>9}   {_metrics_text(row, task)}   {row['time']:8.3f} s      {note}")
        previous = row["features"]

    # ---------------------------------------------------------------- output
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "summary.txt").write_text("\n".join(report) + "\n")

        plot_cv_curve(selection_model.cv, alpha_min=selection_model.alpha,
                      alpha_selected=alpha_selection, selected_label=rule_name,
                      title=f"{name}: {k}-fold cross-validation error",
                      path=output_dir / "cv_curve.png")
        plot_coefficient_path(path_alphas, path_betas, names[current], selected=picked,
                              alpha_min=selection_model.alpha, alpha_selected=alpha_selection,
                              selected_label=rule_name,
                              title=f"{name}: Lasso coefficient path",
                              path=output_dir / "coefficient_path.png")
        plot_tradeoff(sweep, task=task, n_selected=selected.size,
                      selected_label=f"selected ({rule_name})",
                      title=f"{name}: effect of the number of features",
                      path=output_dir / "tradeoff.png")

    return {"results": results, "selected": selected, "alpha_selection": alpha_selection,
            "ranking": current[order], "sweep": sweep}
