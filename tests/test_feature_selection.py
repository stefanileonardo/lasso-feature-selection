"""Tests of the feature selection pipeline on data with a known answer."""

import numpy as np

from src.feature_selection import (evaluate, feature_ranking, select_features,
                                   selection_sweep, timed, train_lasso_cv)


def synthetic_regression(n_samples=400, n_features=12, seed=0):
    """Only features 0, 1 and 2 influence the target (in decreasing order)."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, n_features)) * rng.uniform(0.5, 10.0, size=n_features)
    signal = 3.0 * X[:, 0] / X[:, 0].std() + 2.0 * X[:, 1] / X[:, 1].std() \
        - 1.0 * X[:, 2] / X[:, 2].std()
    y = signal + rng.normal(0.0, 0.5, size=n_samples)
    return X, (y - y.mean()) / y.std()


def test_training_gives_an_accurate_model():
    X, y = synthetic_regression()
    model = train_lasso_cv(X[:300], y[:300])

    metrics = evaluate(model, X[300:], y[300:])
    assert metrics["r2"] > 0.95
    assert np.isclose(metrics["mse"], np.mean((y[300:] - model.predict(X[300:])) ** 2))
    assert model.alpha in model.cv.alphas
    assert set(np.flatnonzero(model.beta)) >= {0, 1, 2}


def test_selection_keeps_exactly_the_informative_features():
    X, y = synthetic_regression()
    model = train_lasso_cv(X, y)
    selected, alpha = select_features(model, X, y, rule="1se")

    assert list(selected) == [0, 1, 2]
    assert alpha >= model.alpha  # the 1-SE rule never regularises less than the minimum


def test_feature_ranking_follows_the_importance():
    X, y = synthetic_regression()
    order, alphas, betas = feature_ranking(X, y)

    assert list(order[:3]) == [0, 1, 2]
    assert sorted(order) == list(range(X.shape[1]))
    assert betas.shape == (len(alphas), X.shape[1])


def test_sweep_retrains_on_growing_feature_subsets():
    X, y = synthetic_regression()
    rows = selection_sweep(X[:300], y[:300], X[300:], y[300:], repeats=1)

    sizes = [row["n_features"] for row in rows]
    assert sizes == sorted(set(sizes))
    assert sizes[0] == 1 and sizes[-1] == X.shape[1]
    assert list(rows[0]["features"]) == [0]
    assert all(row["time"] > 0 for row in rows)
    by_size = {row["n_features"]: row for row in rows}
    # Three features are enough: more features do not improve the test error.
    assert by_size[1]["mse"] > 2 * by_size[3]["mse"]
    assert abs(by_size[3]["mse"] - rows[-1]["mse"]) < 0.01


def test_classification_metrics():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 5))
    y = (X[:, 0] + 0.3 * rng.normal(size=300) > 0).astype(float)
    model = train_lasso_cv(X[:200], y[:200])

    metrics = evaluate(model, X[200:], y[200:], task="classification")
    labels = (model.predict(X[200:]) >= 0.5).astype(int)
    assert metrics["accuracy"] > 0.85
    assert metrics["errors"] == int(np.sum(labels != y[200:]))
    assert np.isclose(metrics["accuracy"], 1 - metrics["errors"] / 100)


def test_timed_returns_result_and_duration():
    result, seconds = timed(sum, [1, 2, 3], repeats=2)
    assert result == 6 and seconds >= 0.0
