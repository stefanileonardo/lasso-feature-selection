"""Lasso regression implemented from scratch with cyclic coordinate descent.

The Lasso fits a linear model by minimising

    1 / (2 n) * || y - X beta ||^2  +  alpha * || beta ||_1

where ``n`` is the number of samples and ``alpha >= 0`` is the regularisation
strength. The L1 penalty shrinks the coefficients and sets the least useful
ones exactly to zero, which is why the Lasso can be used as a feature
selection method.

Coordinate descent minimises the objective with respect to one coefficient at
a time while keeping all the others fixed. For coefficient ``j`` the
minimiser has a closed form:

    rho_j  = 1/n * x_j^T (y - X beta + x_j beta_j)    # fit of feature j to the partial residual
    beta_j = soft_threshold(rho_j, alpha) / (1/n * ||x_j||^2)

For standardised features ``1/n * ||x_j||^2 = 1`` and the update reduces to a
plain soft-thresholding step.
"""

import warnings

import numpy as np

# Default settings of the coordinate descent solver, shared by the whole project.
MAX_ITER = 10_000  # maximum number of sweeps over the coefficients
TOL = 1e-5         # stop when the coefficients change less than this in one sweep


def soft_threshold(z, alpha):
    """Soft-thresholding operator ``S(z, alpha) = sign(z) * max(|z| - alpha, 0)``.

    It moves ``z`` towards zero by ``alpha`` and returns exactly zero when
    ``|z| <= alpha``.
    """
    return np.sign(z) * max(abs(z) - alpha, 0.0)


def lasso_coordinate_descent(X, y, alpha, max_iter=MAX_ITER, tol=TOL,
                             beta_init=None, return_n_iter=False):
    """Solve the Lasso problem (without intercept) by cyclic coordinate descent.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Data matrix. Columns should be centred (see :func:`fit_lasso`) and are
        usually standardised, so that the penalty treats all features alike.
    y : ndarray of shape (n_samples,)
        Centred target vector.
    alpha : float
        Regularisation strength (``alpha = 0`` gives ordinary least squares).
    max_iter : int
        Maximum number of sweeps over all the coefficients.
    tol : float
        The algorithm stops when the Euclidean norm of the change of ``beta``
        over one full sweep is smaller than ``tol``.
    beta_init : ndarray of shape (n_features,), optional
        Starting point (warm start). Defaults to all zeros.
    return_n_iter : bool
        If True, also return the number of sweeps performed.

    Returns
    -------
    beta : ndarray of shape (n_features,)
        Estimated coefficients.
    n_iter : int
        Number of sweeps performed (only if ``return_n_iter`` is True).
    """
    # Column-major layout makes every column x_j a contiguous block of memory.
    X = np.asfortranarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n_samples, n_features = X.shape

    # 1/n * ||x_j||^2 for every column (equal to 1 for standardised features).
    col_scale = np.einsum("ij,ij->j", X, X) / n_samples

    if beta_init is None:
        beta = np.zeros(n_features)
    else:
        beta = np.array(beta_init, dtype=float)
        beta[col_scale == 0.0] = 0.0

    # Residual of the current model. It is updated after every coordinate
    # step, so it never has to be recomputed from scratch.
    residual = y - X @ beta

    converged = False
    n_iter = 0
    for n_iter in range(1, max_iter + 1):  # first stop condition: maximum number of sweeps
        beta_old = beta.copy()

        for j in range(n_features):
            if col_scale[j] == 0.0:
                continue  # an all-zero column carries no information
            x_j = X[:, j]

            # Least-squares fit of feature j to the partial residual, i.e. the
            # residual computed without the contribution of feature j.
            rho_j = x_j @ residual / n_samples + col_scale[j] * beta[j]

            # Soft-thresholding: shrink the coefficient by alpha and set it
            # exactly to zero when it is smaller than alpha in absolute value.
            beta_j = soft_threshold(rho_j, alpha) / col_scale[j]

            if beta_j != beta[j]:
                residual += x_j * (beta[j] - beta_j)
                beta[j] = beta_j

        # Second stop condition: negligible change between two sweeps.
        if np.linalg.norm(beta - beta_old) < tol:
            converged = True
            break

    if not converged:
        warnings.warn(
            f"Coordinate descent did not converge in {max_iter} sweeps "
            f"(alpha={alpha:.3g}). Increase max_iter or tol.",
            RuntimeWarning,
            stacklevel=2,
        )

    if return_n_iter:
        return beta, n_iter
    return beta


def fit_lasso(X, y, alpha, max_iter=MAX_ITER, tol=TOL, beta_init=None):
    """Fit a Lasso model with an (unpenalised) intercept.

    The intercept is handled by centring: the coefficients are estimated on
    centred data and the intercept is recovered afterwards, so it is never
    shrunk by the L1 penalty.

    Returns
    -------
    beta : ndarray of shape (n_features,)
    intercept : float
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    x_mean = X.mean(axis=0)
    y_mean = y.mean()

    beta = lasso_coordinate_descent(X - x_mean, y - y_mean, alpha,
                                    max_iter=max_iter, tol=tol, beta_init=beta_init)
    intercept = y_mean - x_mean @ beta
    return beta, intercept


def predict(X, beta, intercept=0.0):
    """Prediction of the linear model: ``X @ beta + intercept``."""
    return np.asarray(X, dtype=float) @ beta + intercept


def alpha_max(X, y):
    """Smallest ``alpha`` for which every Lasso coefficient is zero.

    Starting from ``beta = 0``, coefficient ``j`` stays at zero as long as
    ``|1/n * x_j^T y| <= alpha`` (on centred data), hence the maximum over
    the features.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    X_centred = X - X.mean(axis=0)
    y_centred = y - y.mean()
    return float(np.max(np.abs(X_centred.T @ y_centred)) / X.shape[0])


def alpha_grid(X, y, n_alphas=50, ratio=1e-3):
    """Logarithmic grid of ``alpha`` values, from ``alpha_max`` downwards.

    The grid is sorted in decreasing order: it starts from the empty model
    (all coefficients equal to zero) and ends at ``ratio * alpha_max``, where
    the model is close to ordinary least squares.
    """
    a_max = alpha_max(X, y)
    if a_max == 0.0:
        raise ValueError("The target is constant: there is nothing to fit.")
    return a_max * np.logspace(0.0, np.log10(ratio), n_alphas)


def lasso_path(X, y, alphas, max_iter=MAX_ITER, tol=TOL):
    """Lasso coefficients for a whole grid of ``alpha`` values.

    The grid is visited from the largest to the smallest ``alpha`` and every
    fit starts from the solution of the previous one (warm start), which is
    much faster than restarting from zero each time.

    Returns
    -------
    alphas : ndarray of shape (n_alphas,)
        The grid, sorted in decreasing order.
    betas : ndarray of shape (n_alphas, n_features)
        Coefficients for every ``alpha``.
    intercepts : ndarray of shape (n_alphas,)
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    alphas = np.sort(np.asarray(alphas, dtype=float))[::-1]

    x_mean = X.mean(axis=0)
    y_mean = y.mean()
    X_centred = np.asfortranarray(X - x_mean)
    y_centred = y - y_mean

    betas = np.empty((alphas.size, X.shape[1]))
    intercepts = np.empty(alphas.size)
    beta = None
    for i, alpha in enumerate(alphas):
        beta = lasso_coordinate_descent(X_centred, y_centred, alpha,
                                        max_iter=max_iter, tol=tol, beta_init=beta)
        betas[i] = beta
        intercepts[i] = y_mean - x_mean @ beta
    return alphas, betas, intercepts
