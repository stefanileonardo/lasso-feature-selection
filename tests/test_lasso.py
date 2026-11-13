"""Tests of the from-scratch Lasso solver.

The reference is scikit-learn's ``Lasso``, which minimises exactly the same
objective: 1/(2n) * ||y - X beta - b||^2 + alpha * ||beta||_1.
"""

import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer, load_diabetes, make_regression
from sklearn.linear_model import Lasso

from src.lasso import (alpha_grid, alpha_max, fit_lasso, lasso_coordinate_descent,
                       lasso_path, predict, soft_threshold)

TIGHT = dict(max_iter=200_000, tol=1e-10)  # solver settings for exact comparisons


def standardise(X):
    return (X - X.mean(axis=0)) / X.std(axis=0)


def dataset(name):
    if name == "diabetes":
        X, y = load_diabetes(return_X_y=True)
        return standardise(X), (y - y.mean()) / y.std()
    if name == "breast_cancer":  # strongly correlated features, 0/1 target
        X, y = load_breast_cancer(return_X_y=True)
        return standardise(X), y.astype(float)
    if name == "raw_scale":  # features with very different scales and non-zero means
        X, y = make_regression(n_samples=200, n_features=25, n_informative=6,
                               noise=3.0, random_state=0)
        rng = np.random.default_rng(0)
        return X * rng.uniform(0.1, 20.0, size=25) + rng.normal(0.0, 5.0, size=25), y + 40.0
    raise ValueError(name)


def test_soft_threshold():
    assert soft_threshold(3.0, 1.0) == 2.0
    assert soft_threshold(-3.0, 1.0) == -2.0
    assert soft_threshold(0.5, 1.0) == 0.0
    assert soft_threshold(-0.5, 1.0) == 0.0
    assert soft_threshold(2.0, 0.0) == 2.0


@pytest.mark.parametrize("name", ["diabetes", "breast_cancer", "raw_scale"])
@pytest.mark.parametrize("fraction", [0.5, 0.1, 0.01])
def test_matches_scikit_learn(name, fraction):
    X, y = dataset(name)
    alpha = fraction * alpha_max(X, y)

    beta, intercept = fit_lasso(X, y, alpha, **TIGHT)
    reference = Lasso(alpha=alpha, tol=1e-14, max_iter=2_000_000).fit(X, y)

    scale = np.abs(reference.coef_).max()
    assert np.allclose(beta, reference.coef_, atol=1e-5 * scale)
    assert np.isclose(intercept, reference.intercept_, atol=1e-5 * max(1.0, abs(reference.intercept_)))
    assert np.array_equal(beta != 0, reference.coef_ != 0)  # same selected features


@pytest.mark.parametrize("name", ["diabetes", "breast_cancer", "raw_scale"])
def test_optimality_conditions(name):
    """KKT conditions of the Lasso: with r the residual and g_j = x_j^T r / n,
    g_j = alpha * sign(beta_j) where beta_j != 0 and |g_j| <= alpha elsewhere."""
    X, y = dataset(name)
    alpha = 0.05 * alpha_max(X, y)
    beta, intercept = fit_lasso(X, y, alpha, **TIGHT)

    gradient = X.T @ (y - predict(X, beta, intercept)) / X.shape[0]
    active = beta != 0
    assert active.any() and (~active).any()
    assert np.allclose(gradient[active], alpha * np.sign(beta[active]), rtol=1e-5, atol=1e-7 * alpha)
    assert np.all(np.abs(gradient[~active]) <= alpha * (1 + 1e-9))


def test_alpha_max_is_the_threshold_of_the_empty_model():
    X, y = dataset("diabetes")
    a_max = alpha_max(X, y)
    assert not fit_lasso(X, y, 1.001 * a_max)[0].any()
    assert fit_lasso(X, y, 0.95 * a_max)[0].any()


def test_alpha_zero_is_ordinary_least_squares():
    X, y = dataset("diabetes")
    beta, intercept = fit_lasso(X, y, 0.0, **TIGHT)
    design = np.column_stack([np.ones(len(y)), X])
    least_squares = np.linalg.lstsq(design, y, rcond=None)[0]
    assert np.allclose(np.concatenate([[intercept], beta]), least_squares, atol=1e-6)


def test_intercept_is_not_penalised():
    X, y = dataset("diabetes")
    alpha = 0.1 * alpha_max(X, y)
    beta, intercept = fit_lasso(X, y, alpha, **TIGHT)
    beta_shifted, intercept_shifted = fit_lasso(X, y + 100.0, alpha, **TIGHT)
    assert np.allclose(beta, beta_shifted, atol=1e-8)
    assert np.isclose(intercept_shifted, intercept + 100.0)


def test_warm_start_reaches_the_same_solution_in_fewer_sweeps():
    X, y = dataset("breast_cancer")
    X, y = X - X.mean(axis=0), y - y.mean()
    alpha = 0.02 * alpha_max(X, y)
    cold, sweeps_cold = lasso_coordinate_descent(X, y, alpha, return_n_iter=True, **TIGHT)
    start = lasso_coordinate_descent(X, y, 1.2 * alpha, **TIGHT)
    warm, sweeps_warm = lasso_coordinate_descent(X, y, alpha, beta_init=start,
                                                 return_n_iter=True, **TIGHT)
    assert np.allclose(cold, warm, atol=1e-7)
    assert sweeps_warm < sweeps_cold


def test_lasso_path():
    X, y = dataset("diabetes")
    alphas, betas, intercepts = lasso_path(X, y, alpha_grid(X, y, n_alphas=20), **TIGHT)

    assert np.all(np.diff(alphas) < 0)  # decreasing grid
    assert np.isclose(alphas[0], alpha_max(X, y))
    assert np.isclose(alphas[-1], 1e-3 * alpha_max(X, y))
    assert np.count_nonzero(betas[0]) <= 1  # (almost) empty model at alpha_max
    assert np.count_nonzero(betas[-1]) == X.shape[1]  # full model at the end
    for i in (5, 12, 19):  # same result as independent fits
        beta, intercept = fit_lasso(X, y, alphas[i], **TIGHT)
        assert np.allclose(betas[i], beta, atol=1e-6)
        assert np.isclose(intercepts[i], intercept, atol=1e-6)


def test_constant_column_is_ignored():
    X, y = dataset("diabetes")
    X = np.column_stack([X, np.ones(len(y))])  # e.g. a bias column
    beta, _ = fit_lasso(X, y, 0.01)
    assert beta[-1] == 0.0
    assert np.all(np.isfinite(beta))


def test_warning_when_not_converged():
    X, y = dataset("breast_cancer")
    with pytest.warns(RuntimeWarning, match="did not converge"):
        fit_lasso(X, y, 1e-4, max_iter=2, tol=1e-12)
