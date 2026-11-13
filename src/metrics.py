"""Evaluation metrics for regression and classification."""

import numpy as np


def mean_squared_error(y_true, y_pred):
    """Mean of the squared prediction errors."""
    return float(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2))


def r2_score(y_true, y_pred):
    """Coefficient of determination: fraction of the variance of ``y_true``
    explained by the predictions (1 is a perfect fit, 0 is the mean predictor).
    """
    y_true = np.asarray(y_true, dtype=float)
    residual_sum = np.sum((y_true - np.asarray(y_pred)) ** 2)
    total_sum = np.sum((y_true - y_true.mean()) ** 2)
    return float(1.0 - residual_sum / total_sum)


def accuracy(y_true, y_pred):
    """Fraction of correctly classified samples."""
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))
