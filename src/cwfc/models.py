"""Statistical and machine-learning forecasting models."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing


_MIN_HISTORY = 12


def _past_value_features(history: np.ndarray) -> np.ndarray:
    """Create lag and trailing-window features using history available now."""
    features = [
        history[-1],
        history[-2],
        history[-6],
    ]
    for window in (3, 6, 12):
        trailing = history[-window:]
        features.extend((trailing.mean(), trailing.std(ddof=0)))
    return np.asarray(features, dtype=float)


def ridge_forecast(
    train: pd.DataFrame,
    template: pd.DataFrame,
    target: str,
    alpha: float = 1.0,
) -> pd.Series:
    """Fit Ridge on past-only lag features and forecast recursively.

    Supervised training rows begin after 12 observations so every row has
    complete lag and trailing-window statistics. During multi-step prediction,
    each predicted value is appended to history before the next feature row is
    created; no target values from ``template`` are read.
    """
    if target not in train.columns:
        raise ValueError(f"train is missing target column: {target}")
    if alpha < 0:
        raise ValueError("alpha must be non-negative")
    values = pd.to_numeric(train[target], errors="coerce").to_numpy(dtype=float)
    if len(values) <= _MIN_HISTORY:
        raise ValueError(
            f"Ridge forecasting requires more than {_MIN_HISTORY} training observations"
        )
    if not np.isfinite(values).all():
        raise ValueError(f"training target {target} must contain only finite values")

    features = np.vstack(
        [_past_value_features(values[:index]) for index in range(_MIN_HISTORY, len(values))]
    )
    labels = values[_MIN_HISTORY:]
    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    model.fit(features, labels)

    history = values.tolist()
    predictions: list[float] = []
    for _ in range(len(template)):
        next_prediction = float(model.predict(_past_value_features(np.asarray(history)).reshape(1, -1))[0])
        predictions.append(next_prediction)
        history.append(next_prediction)

    return pd.Series(predictions, index=template.index, dtype=float)


def damped_exponential_smoothing_forecast(
    train: pd.DataFrame,
    template: pd.DataFrame,
    target: str,
) -> pd.Series:
    """Forecast with additive damped-trend exponential smoothing.

    The model is fitted only to the training target series. Seasonality is not
    enabled because the available trace is too short to establish a recurring
    seasonal period.
    """
    if target not in train.columns:
        raise ValueError(f"train is missing target column: {target}")
    values = pd.to_numeric(train[target], errors="coerce").to_numpy(dtype=float)
    if len(values) < 3:
        raise ValueError(
            "damped exponential smoothing requires at least 3 training observations"
        )
    if not np.isfinite(values).all():
        raise ValueError(f"training target {target} must contain only finite values")
    if template.empty:
        return pd.Series(index=template.index, dtype=float)
    if np.all(values == values[0]):
        return pd.Series(values[0], index=template.index, dtype=float)

    fitted = ExponentialSmoothing(
        values,
        trend="add",
        damped_trend=True,
        seasonal=None,
        initialization_method="estimated",
    ).fit(optimized=True)
    predictions = np.asarray(fitted.forecast(len(template)), dtype=float)
    if predictions.shape != (len(template),) or not np.isfinite(predictions).all():
        raise ValueError("damped exponential smoothing produced invalid forecasts")
    return pd.Series(predictions, index=template.index, dtype=float)
