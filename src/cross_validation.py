"""K-fold cross-validation to choose the regularisation strength of the Lasso."""

from dataclasses import dataclass

import numpy as np

from .lasso import MAX_ITER, TOL, lasso_coordinate_descent
from .metrics import mean_squared_error
from .preprocessing import Standardizer


@dataclass
class CVResult:
    """Cross-validation curves, one value per ``alpha`` (largest alpha first)."""

    alphas: np.ndarray          # alpha values actually evaluated
    train_errors: np.ndarray    # training MSE, averaged over the folds
    val_errors: np.ndarray      # validation MSE, averaged over the folds
    val_std_errors: np.ndarray  # standard error of the validation MSE over the folds
    n_active: np.ndarray        # number of non-zero coefficients, averaged over the folds


def k_fold_indices(n_samples, k=5, seed=0):
    """Split the sample indices into ``k`` folds.

    The indices are shuffled once and then cut into ``k`` disjoint folds of
    (almost) equal size, so every sample is used for validation exactly once.

    Returns
    -------
    list of (train_indices, validation_indices) pairs, one per fold.
    """
    indices = np.random.default_rng(seed).permutation(n_samples)
    folds = np.array_split(indices, k)
    return [
        (np.concatenate([folds[j] for j in range(k) if j != i]), folds[i])
        for i in range(k)
    ]


def k_fold_cv_lasso(X, y, alphas, k=5, max_iter=MAX_ITER, tol=TOL,
                    patience=5, min_delta=1e-3, seed=0):
    """K-fold cross-validation of the Lasso over a grid of ``alpha`` values.

    For every fold the features are standardised with the statistics of the
    training part only, the Lasso is fitted on the training part and its mean
    squared error is measured on both the training and the validation part.

    The grid is visited from the largest to the smallest ``alpha`` and each
    fold warm-starts from its previous solution. The search stops early when
    the validation error has stopped changing.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Training data (not standardised: this is done inside every fold).
    y : ndarray of shape (n_samples,)
        Target values.
    alphas : array-like
        Candidate regularisation strengths.
    k : int
        Number of folds.
    max_iter, tol :
        Parameters of the coordinate descent solver.
    patience : int or None
        Stop when the validation error has changed by less than ``min_delta``
        (relative change) for ``patience`` consecutive values of ``alpha``.
        ``None`` disables early stopping.
    min_delta : float
        Relative change of the validation error below which two consecutive
        models are considered equivalent.
    seed : int
        Seed used to shuffle the samples before building the folds.

    Returns
    -------
    CVResult
    """
    if k < 2:
        raise ValueError("K-fold cross-validation needs at least 2 folds.")
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    alphas = np.sort(np.asarray(alphas, dtype=float))[::-1]  # strongest regularisation first

    # Prepare the data of every fold once: standardised features and centred target.
    folds = []
    for train_idx, val_idx in k_fold_indices(X.shape[0], k, seed):
        scaler = Standardizer().fit(X[train_idx])
        y_mean = y[train_idx].mean()  # intercept of the model (features are centred)
        folds.append((np.asfortranarray(scaler.transform(X[train_idx])), y[train_idx],
                      scaler.transform(X[val_idx]), y[val_idx], y_mean))

    betas = [None] * k  # last solution of every fold, used as warm start
    alphas_tried, train_errors, val_errors, val_std_errors, n_active = [], [], [], [], []
    stop_counter = 0

    for alpha in alphas:
        fold_train = np.empty(k)
        fold_val = np.empty(k)
        fold_active = np.empty(k)

        for f, (X_train, y_train, X_val, y_val, y_mean) in enumerate(folds):
            beta = lasso_coordinate_descent(X_train, y_train - y_mean, alpha,
                                            max_iter=max_iter, tol=tol, beta_init=betas[f])
            betas[f] = beta
            fold_train[f] = mean_squared_error(y_train, X_train @ beta + y_mean)
            fold_val[f] = mean_squared_error(y_val, X_val @ beta + y_mean)
            fold_active[f] = np.count_nonzero(beta)

        alphas_tried.append(alpha)
        train_errors.append(fold_train.mean())
        val_errors.append(fold_val.mean())
        val_std_errors.append(fold_val.std(ddof=1) / np.sqrt(k))
        n_active.append(fold_active.mean())

        # Early stopping: count the consecutive alphas for which the validation
        # error did not change significantly. The check only starts once the
        # model is not empty, otherwise the flat start of the path (where all
        # the coefficients are still zero) would stop the search immediately.
        if patience is not None and len(val_errors) > 1 and n_active[-1] > 0:
            change = abs(val_errors[-1] - val_errors[-2])
            if change < min_delta * val_errors[-2]:
                stop_counter += 1
            else:
                stop_counter = 0
            if stop_counter >= patience:
                break

    return CVResult(np.array(alphas_tried), np.array(train_errors), np.array(val_errors),
                    np.array(val_std_errors), np.array(n_active))


def select_alpha(cv, rule="min"):
    """Choose ``alpha`` from the cross-validation curve.

    Parameters
    ----------
    cv : CVResult
    rule : {"min", "1se"}
        ``"min"``: the alpha with the lowest validation error.
        ``"1se"``: one-standard-error rule, i.e. the largest alpha (the
        sparsest model) whose validation error is within one standard error
        of the minimum. It trades a statistically insignificant loss of
        accuracy for a simpler model.

    Returns
    -------
    float
    """
    best = int(np.argmin(cv.val_errors))
    if rule == "min":
        return float(cv.alphas[best])
    if rule == "1se":
        threshold = cv.val_errors[best] + cv.val_std_errors[best]
        # alphas are sorted in decreasing order: move towards stronger
        # regularisation while the error stays below the threshold.
        i = best
        while i > 0 and cv.val_errors[i - 1] <= threshold:
            i -= 1
        return float(cv.alphas[i])
    raise ValueError(f"Unknown rule {rule!r}: use 'min' or '1se'.")
