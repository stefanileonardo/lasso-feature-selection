"""Tests of the K-fold cross-validation."""

import numpy as np
import pytest
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

from src.cross_validation import CVResult, k_fold_cv_lasso, k_fold_indices, select_alpha
from src.lasso import alpha_grid


def diabetes():
    X, y = load_diabetes(return_X_y=True)
    return X, (y - y.mean()) / y.std()


def test_folds_are_a_partition():
    folds = k_fold_indices(103, k=5, seed=0)
    assert len(folds) == 5

    validation = np.concatenate([val for _, val in folds])
    assert sorted(validation) == list(range(103))  # every sample validated exactly once
    for train, val in folds:
        assert len(set(train) & set(val)) == 0
        assert len(train) + len(val) == 103
        assert len(val) in (20, 21)


def test_cross_validation_matches_scikit_learn():
    """Same folds, same preprocessing, same model: the errors must coincide."""
    X, y = diabetes()
    alphas = [0.3, 0.05, 0.005]
    cv = k_fold_cv_lasso(X, y, alphas, k=5, seed=3, patience=None, max_iter=100_000, tol=1e-10)

    expected_val = np.zeros((3, 5))
    expected_train = np.zeros((3, 5))
    for f, (train, val) in enumerate(k_fold_indices(len(y), k=5, seed=3)):
        scaler = StandardScaler().fit(X[train])
        for a, alpha in enumerate(alphas):
            model = Lasso(alpha=alpha, tol=1e-14, max_iter=1_000_000)
            model.fit(scaler.transform(X[train]), y[train])
            expected_val[a, f] = np.mean((y[val] - model.predict(scaler.transform(X[val]))) ** 2)
            expected_train[a, f] = np.mean((y[train] - model.predict(scaler.transform(X[train]))) ** 2)

    assert np.allclose(cv.alphas, alphas)
    assert np.allclose(cv.val_errors, expected_val.mean(axis=1), atol=1e-7)
    assert np.allclose(cv.train_errors, expected_train.mean(axis=1), atol=1e-7)
    assert np.allclose(cv.val_std_errors, expected_val.std(axis=1, ddof=1) / np.sqrt(5), atol=1e-7)


def test_alphas_are_visited_in_decreasing_order():
    X, y = diabetes()
    cv = k_fold_cv_lasso(X, y, [0.01, 0.5, 0.1], patience=None)
    assert np.allclose(cv.alphas, [0.5, 0.1, 0.01])
    assert cv.n_active[0] <= cv.n_active[-1]


def test_early_stopping():
    X, y = diabetes()
    X_std = StandardScaler().fit_transform(X)
    alphas = alpha_grid(X_std, y, n_alphas=60, ratio=1e-5)

    complete = k_fold_cv_lasso(X, y, alphas, patience=None)
    stopped = k_fold_cv_lasso(X, y, alphas, patience=5, min_delta=1e-3)

    assert len(complete.alphas) == 60
    assert len(stopped.alphas) < 60
    # The evaluated part of the curve is identical and the minimum is not missed.
    assert np.allclose(stopped.val_errors, complete.val_errors[: len(stopped.alphas)])
    assert np.isclose(stopped.val_errors.min(), complete.val_errors.min(), rtol=5e-3)


def test_early_stopping_ignores_the_empty_models_at_the_start():
    """A grid that starts far above alpha_max gives many identical (empty)
    models: this flat start must not be mistaken for convergence."""
    X, y = diabetes()
    alphas = np.logspace(2, -3, 60)  # the first ~20 alphas give an empty model
    cv = k_fold_cv_lasso(X, y, alphas, patience=5, min_delta=1e-3)
    assert cv.n_active[-1] > 0
    assert cv.val_errors.min() < 0.6 * cv.val_errors[0]


def test_select_alpha():
    cv = CVResult(alphas=np.array([1.0, 0.5, 0.2, 0.1, 0.05]),
                  train_errors=np.zeros(5),
                  val_errors=np.array([0.90, 0.52, 0.49, 0.45, 0.47]),
                  val_std_errors=np.array([0.05, 0.05, 0.05, 0.05, 0.05]),
                  n_active=np.arange(5.0))

    assert select_alpha(cv, "min") == 0.1
    # Threshold 0.45 + 0.05 = 0.50: alpha 0.2 qualifies, alpha 0.5 does not.
    assert select_alpha(cv, "1se") == 0.2
    with pytest.raises(ValueError):
        select_alpha(cv, "unknown")
