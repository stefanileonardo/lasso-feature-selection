"""Data preparation: train/test split, standardisation and correlation filter."""

import numpy as np


def train_test_split(X, y, test_size=0.3, seed=0, stratify=False):
    """Randomly split the samples into a training set and a test set.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
    y : ndarray of shape (n_samples,)
    test_size : float
        Fraction of the samples assigned to the test set.
    seed : int
        Seed of the random permutation, for reproducible splits.
    stratify : bool
        If True, every class of ``y`` keeps the same proportion in the two
        sets (useful for classification problems).

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    rng = np.random.default_rng(seed)
    n_samples = X.shape[0]

    if stratify:
        test_parts = []
        for label in np.unique(y):
            members = rng.permutation(np.flatnonzero(y == label))
            test_parts.append(members[: int(round(test_size * members.size))])
        test_idx = np.concatenate(test_parts)
        is_test = np.zeros(n_samples, dtype=bool)
        is_test[test_idx] = True
        test_idx = rng.permutation(test_idx)
        train_idx = rng.permutation(np.flatnonzero(~is_test))
    else:
        permutation = rng.permutation(n_samples)
        n_test = int(round(test_size * n_samples))
        test_idx, train_idx = permutation[:n_test], permutation[n_test:]

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


class Standardizer:
    """Z-score normalisation: ``z = (x - mean) / std``.

    The statistics are learned with :meth:`fit` on the training data only and
    then applied unchanged to validation and test data, so that no information
    leaks from the data used to evaluate the model.
    """

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        std = X.std(axis=0)
        # A constant feature has zero standard deviation: leave it unscaled.
        self.std_ = np.where(std == 0.0, 1.0, std)
        return self

    def transform(self, X):
        return (np.asarray(X, dtype=float) - self.mean_) / self.std_

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, Z):
        return np.asarray(Z, dtype=float) * self.std_ + self.mean_


def drop_correlated_features(X, threshold=0.95):
    """Remove redundant features with a correlation filter.

    Two features whose absolute Pearson correlation is at least ``threshold``
    carry almost the same information. As long as such a pair exists, the
    most correlated pair is taken and, of the two, the feature that is on
    average more correlated with all the remaining ones is discarded (the
    other one carries the more varied information).

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Training data (the filter must not look at the test set).
    threshold : float
        Absolute correlation above which two features are considered
        redundant.

    Returns
    -------
    kept : ndarray of int
        Sorted indices of the columns to keep.
    """
    corr = np.abs(np.corrcoef(np.asarray(X, dtype=float), rowvar=False))
    corr = np.nan_to_num(corr)  # constant columns give NaN correlations
    np.fill_diagonal(corr, 0.0)

    kept = list(range(corr.shape[0]))
    while len(kept) > 1:
        sub = corr[np.ix_(kept, kept)]
        i, j = np.unravel_index(np.argmax(sub), sub.shape)
        if sub[i, j] < threshold:
            break
        # Mean correlation of each feature of the pair with all the others.
        mean_corr = sub.sum(axis=1) / (len(kept) - 1)
        del kept[i if mean_corr[i] >= mean_corr[j] else j]

    return np.array(kept)
