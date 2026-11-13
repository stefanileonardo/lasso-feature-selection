"""Tests of the data preparation functions."""

import numpy as np
from sklearn.preprocessing import StandardScaler

from src.preprocessing import Standardizer, drop_correlated_features, train_test_split


def test_standardizer_matches_scikit_learn():
    rng = np.random.default_rng(0)
    X_train = rng.normal(5.0, 3.0, size=(100, 4))
    X_test = rng.normal(5.0, 3.0, size=(30, 4))

    scaler = Standardizer().fit(X_train)
    reference = StandardScaler().fit(X_train)
    assert np.allclose(scaler.transform(X_train), reference.transform(X_train))
    assert np.allclose(scaler.transform(X_test), reference.transform(X_test))
    assert np.allclose(scaler.transform(X_train).mean(axis=0), 0.0)
    assert np.allclose(scaler.transform(X_train).std(axis=0), 1.0)
    assert np.allclose(scaler.inverse_transform(scaler.transform(X_test)), X_test)


def test_standardizer_constant_column_and_vector():
    X = np.column_stack([np.arange(5.0), np.full(5, 7.0)])
    Z = Standardizer().fit_transform(X)
    assert np.all(np.isfinite(Z))
    assert np.allclose(Z[:, 1], 0.0)

    y = np.array([1.0, 2.0, 3.0, 4.0])
    z = Standardizer().fit_transform(y)
    assert z.shape == y.shape
    assert np.isclose(z.mean(), 0.0) and np.isclose(z.std(), 1.0)


def test_train_test_split_is_a_reproducible_partition():
    X = np.arange(200).reshape(100, 2)
    y = np.arange(100)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, seed=1)

    assert len(y_test) == 30 and len(y_train) == 70
    assert sorted(np.concatenate([y_train, y_test])) == list(range(100))
    assert np.array_equal(X_train[:, 0], 2 * y_train)  # rows stay aligned with the targets
    assert not np.array_equal(y_train, np.sort(y_train))  # shuffled

    again = train_test_split(X, y, test_size=0.3, seed=1)
    assert np.array_equal(again[2], y_train)
    other = train_test_split(X, y, test_size=0.3, seed=2)
    assert not np.array_equal(other[2], y_train)


def test_stratified_split_keeps_class_proportions():
    y = np.array([0] * 40 + [1] * 160)
    X = np.arange(200).reshape(200, 1)
    _, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, seed=0, stratify=True)

    assert len(y_test) == 50 and len(y_train) == 150
    assert y_test.sum() == 40 and y_train.sum() == 120  # 80 % of class 1 in both sets
    assert np.array_equal(y[X_test[:, 0]], y_test)


def test_correlation_filter_removes_redundant_features():
    rng = np.random.default_rng(0)
    a, b, c = rng.normal(size=(3, 500))
    X = np.column_stack([a, b, a + 0.01 * rng.normal(size=500),  # copy of feature 0
                         c, 2.0 * b + 0.01 * rng.normal(size=500),  # copy of feature 1
                         -a + 0.01 * rng.normal(size=500)])  # negative copy of feature 0

    kept = drop_correlated_features(X, threshold=0.95)

    assert len(kept) == 3  # one feature per group survives
    assert 3 in kept  # the independent feature is untouched
    assert len(set(kept) & {0, 2, 5}) == 1 and len(set(kept) & {1, 4}) == 1
    correlation = np.abs(np.corrcoef(X[:, kept], rowvar=False))
    np.fill_diagonal(correlation, 0.0)
    assert correlation.max() < 0.95


def test_correlation_filter_respects_the_threshold():
    rng = np.random.default_rng(1)
    a, b = rng.normal(size=(2, 20_000))
    X = np.column_stack([a, a + 0.25 * rng.normal(size=20_000),   # correlation ~0.97 with a
                         b, b + 0.48 * rng.normal(size=20_000)])  # correlation ~0.90 with b

    assert np.array_equal(drop_correlated_features(X, threshold=0.99), [0, 1, 2, 3])
    assert len(drop_correlated_features(X, threshold=0.95)) == 3
    assert set(drop_correlated_features(X, threshold=0.95)) >= {2, 3}
    assert len(drop_correlated_features(X, threshold=0.85)) == 2


def test_correlation_filter_drops_the_more_redundant_feature_of_a_pair():
    rng = np.random.default_rng(2)
    a, c = rng.normal(size=(2, 20_000))
    b = a + 0.15 * c  # almost a copy of a, but also correlated with c
    X = np.column_stack([a, b, c])
    # a and b are redundant; b is on average more correlated with the rest, so it goes.
    assert np.array_equal(drop_correlated_features(X, threshold=0.95), [0, 2])


def test_correlation_filter_keeps_everything_below_threshold():
    X = np.random.default_rng(1).normal(size=(300, 6))
    assert np.array_equal(drop_correlated_features(X, threshold=0.95), np.arange(6))
