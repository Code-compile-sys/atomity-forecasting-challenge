from __future__ import annotations

import numpy as np


def _valid_pairs(*values) -> tuple[np.ndarray, ...]:
    arrays = tuple(np.asarray(value, dtype=float) for value in values)
    if not arrays:
        raise ValueError("at least one metric input is required")
    if any(array.shape != arrays[0].shape for array in arrays[1:]):
        raise ValueError("metric inputs must have the same shape")
    if any(np.isinf(array).any() for array in arrays):
        raise ValueError("metric inputs must not contain infinite values")
    valid = np.logical_and.reduce([np.isfinite(array) for array in arrays])
    if not valid.any():
        raise ValueError("metric inputs have no valid paired observations")
    return tuple(array[valid] for array in arrays)


def mae(y_true, y_pred) -> float:
    a, b = _valid_pairs(y_true, y_pred)
    return float(np.mean(np.abs(a - b)))


def rmse(y_true, y_pred) -> float:
    a, b = _valid_pairs(y_true, y_pred)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def smape(y_true, y_pred, eps: float = 1e-8) -> float:
    """Return sMAPE in percent, excluding missing pairs and protecting zero values."""
    if eps <= 0:
        raise ValueError("eps must be positive")
    a, b = _valid_pairs(y_true, y_pred)
    denom = np.maximum(np.abs(a) + np.abs(b), eps)
    return float(np.mean(2.0 * np.abs(a - b) / denom) * 100.0)


def normalized_mae(y_true, y_pred, baseline_pred) -> float:
    """Return model MAE divided by baseline MAE on shared valid observations."""
    a, prediction, baseline = _valid_pairs(y_true, y_pred, baseline_pred)
    model_mae = float(np.mean(np.abs(a - prediction)))
    baseline_mae = float(np.mean(np.abs(a - baseline)))
    zero_tolerance = np.finfo(float).eps * max(1.0, float(np.mean(np.abs(a)))) * 10
    if baseline_mae <= zero_tolerance:
        return 0.0 if model_mae <= zero_tolerance else float("inf")
    return float(model_mae / baseline_mae)
