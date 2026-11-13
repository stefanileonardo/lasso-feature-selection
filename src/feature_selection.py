"""Lasso-based feature selection: train, select, retrain and compare.

The Lasso sets the coefficients of the least useful features exactly to zero,
so the features that survive (non-zero coefficient) are the selected ones.
The stronger the regularisation ``alpha``, the fewer features are kept.
"""

import time
from dataclasses import dataclass

import numpy as np

from .cross_validation import CVResult, k_fold_cv_lasso, select_alpha
from .lasso import MAX_ITER, TOL, alpha_grid, fit_lasso, lasso_path, predict
from .metrics import accuracy, mean_squared_error, r2_score
from .preprocessing import Standardizer


@dataclass
class LassoModel:
    """A trained predictor: standardisation + Lasso coefficients."""

    scaler: Standardizer  # standardisation learned on the training set
    beta: np.ndarray      # coefficients (for the standardised features)
    intercept: float
    alpha: float          # regularisation strength chosen by cross-validation
    cv: CVResult          # cross-validation curves used to choose alpha

    def predict(self, X):
        return predict(self.scaler.transform(X), self.beta, self.intercept)


def train_lasso_cv(X_train, y_train, rule="min", n_alphas=50, alpha_ratio=1e-3,
                   k=5, max_iter=MAX_ITER, tol=TOL, patience=5, min_delta=1e-3, seed=0):
    """Complete training procedure of the predictor.

    1. build the grid of ``alpha`` values, from the empty model downwards;
    2. choose ``alpha`` by K-fold cross-validation;
    3. fit the final model on the whole training set with the chosen ``alpha``.

    Returns
    -------
    LassoModel
    """
    scaler = Standardizer().fit(X_train)
    X_std = scaler.transform(X_train)

    alphas = alpha_grid(X_std, y_train, n_alphas=n_alphas, ratio=alpha_ratio)
    cv = k_fold_cv_lasso(X_train, y_train, alphas, k=k, max_iter=max_iter, tol=tol,
                         patience=patience, min_delta=min_delta, seed=seed)
    alpha = select_alpha(cv, rule)

    beta, intercept = fit_lasso(X_std, y_train, alpha, max_iter=max_iter, tol=tol)
    return LassoModel(scaler, beta, intercept, alpha, cv)


def select_features(model, X_train, y_train, rule="1se", max_iter=MAX_ITER, tol=TOL):
    """Select the features with a non-zero Lasso coefficient.

    The Lasso is fitted on the whole training set with the ``alpha`` given by
    ``rule`` on the cross-validation curve of ``model`` (see
    :func:`select_alpha`): ``"1se"`` regularises more strongly than ``"min"``
    and therefore selects fewer features.

    Returns
    -------
    selected : ndarray of int
        Indices of the selected columns.
    alpha : float
        Regularisation strength used for the selection.
    """
    alpha = select_alpha(model.cv, rule)
    beta, _ = fit_lasso(model.scaler.transform(X_train), y_train, alpha,
                        max_iter=max_iter, tol=tol)
    return np.flatnonzero(beta), alpha


def evaluate(model, X, y, task="regression"):
    """Performance of ``model`` on the data ``(X, y)``.

    For regression: mean squared error and R^2. For classification the model
    regresses the 0/1 class label and the prediction is thresholded at 0.5:
    accuracy and number of misclassified samples.
    """
    y_pred = model.predict(X)
    if task == "classification":
        labels = (y_pred >= 0.5).astype(int)
        return {"accuracy": accuracy(y, labels), "errors": int(np.sum(labels != y))}
    return {"mse": mean_squared_error(y, y_pred), "r2": r2_score(y, y_pred)}


def timed(function, *args, repeats=3, **kwargs):
    """Run ``function`` ``repeats`` times; return its result and the best time.

    The minimum over a few repetitions is the least sensitive to whatever
    else the machine is doing.
    """
    best = np.inf
    for _ in range(repeats):
        start = time.perf_counter()
        result = function(*args, **kwargs)
        best = min(best, time.perf_counter() - start)
    return result, best


def feature_ranking(X_train, y_train, n_alphas=50, alpha_ratio=1e-3, max_iter=MAX_ITER, tol=TOL):
    """Rank the features by the order in which they enter the Lasso path.

    Going from strong to weak regularisation, the most important features are
    the first ones to get a non-zero coefficient.

    Returns
    -------
    order : ndarray of int
        Feature indices, from the first to the last one entering the model
        (features that never enter are placed at the end).
    alphas : ndarray of shape (n_alphas,)
    betas : ndarray of shape (n_alphas, n_features)
        The coefficient path (useful for plotting).
    """
    X_std = Standardizer().fit_transform(X_train)
    alphas = alpha_grid(X_std, y_train, n_alphas=n_alphas, ratio=alpha_ratio)
    alphas, betas, _ = lasso_path(X_std, y_train, alphas, max_iter=max_iter, tol=tol)

    active = betas != 0.0
    # Index of the first alpha at which each feature is active (n_alphas if never).
    first_active = np.where(active.any(axis=0), active.argmax(axis=0), alphas.size)
    # Ties (features entering at the same alpha) are broken by coefficient size.
    order = np.lexsort((-np.abs(betas[-1]), first_active))
    return order, alphas, betas


def selection_sweep(X_train, y_train, X_test, y_test, task="regression",
                    repeats=3, **train_options):
    """Performance and training time as a function of the number of features.

    Every distinct set of active features met along the Lasso path (from the
    sparsest to the densest) is used to retrain the predictor from scratch;
    the retrained model is evaluated on the test set and its training time is
    measured.

    Returns
    -------
    list of dict, one per feature subset, with keys ``n_features``,
    ``features`` (column indices), ``alpha`` (where the subset appears on the
    path; 0 for the complete feature set when the path does not reach it),
    ``time`` (training time in seconds) and the metrics of :func:`evaluate`.
    """
    _, alphas, betas = feature_ranking(X_train, y_train)

    # Distinct feature subsets along the path, one per subset size.
    subsets = {}
    for alpha, beta in zip(alphas, betas):
        features = np.flatnonzero(beta)
        if features.size > 0 and features.size not in subsets:
            subsets[features.size] = (float(alpha), features)
    # The complete feature set closes the comparison even if the path does not reach it.
    n_features = X_train.shape[1]
    subsets.setdefault(n_features, (0.0, np.arange(n_features)))

    rows = []
    for size in sorted(subsets):
        alpha, features = subsets[size]
        model, seconds = timed(train_lasso_cv, X_train[:, features], y_train,
                               repeats=repeats, **train_options)
        rows.append({"n_features": size, "features": features, "alpha": alpha,
                     "time": seconds, **evaluate(model, X_test[:, features], y_test, task)})
    return rows
