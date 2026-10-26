from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_SEASON_LENGTH = 12


def _training_values(train: pd.DataFrame, target: str, minimum_size: int = 1) -> np.ndarray:
    """Return finite numeric target values, validating required history."""
    if target not in train.columns:
        raise ValueError(f"train is missing target column: {target}")
    values = pd.to_numeric(train[target], errors="coerce").to_numpy(dtype=float)
    if len(values) < minimum_size:
        raise ValueError(f"not enough training observations; need at least {minimum_size}")
    if not np.isfinite(values).all():
        raise ValueError(f"training target {target} must contain only finite numeric values")
    return values


def persistence_forecast(train: pd.DataFrame, template: pd.DataFrame, target: str) -> pd.Series:
    """Repeat the most recent training value across the forecast horizon."""
    values = _training_values(train, target)
    last_value = float(values[-1])
    return pd.Series(last_value, index=template.index, dtype=float)


def last_n_mean_forecast(
    train: pd.DataFrame,
    template: pd.DataFrame,
    target: str,
    window: int,
) -> pd.Series:
    """Forecast the mean of the latest ``window`` training observations."""
    if window <= 0:
        raise ValueError("window must be positive")
    values = _training_values(train, target, minimum_size=window)
    return pd.Series(float(values[-window:].mean()), index=template.index, dtype=float)


def last_6_mean_forecast(
    train: pd.DataFrame, template: pd.DataFrame, target: str
) -> pd.Series:
    """Repeat the mean of the most recent six training observations."""
    return last_n_mean_forecast(train, template, target, window=6)


def last_12_mean_forecast(
    train: pd.DataFrame, template: pd.DataFrame, target: str
) -> pd.Series:
    """Repeat the mean of the most recent twelve training observations."""
    return last_n_mean_forecast(train, template, target, window=12)


def training_mean_forecast(
    train: pd.DataFrame, template: pd.DataFrame, target: str
) -> pd.Series:
    """Repeat the mean of all available training observations."""
    values = _training_values(train, target)
    return pd.Series(float(values.mean()), index=template.index, dtype=float)


def seasonal_naive_forecast(
    train: pd.DataFrame,
    template: pd.DataFrame,
    target: str,
    season_length: int = DEFAULT_SEASON_LENGTH,
) -> pd.Series:
    """Repeat the latest observed seasonal cycle across the forecast horizon.

    The default period is 12 five-minute observations (one hour). This is a
    candidate baseline based on the data cadence, not an inferred seasonality.
    """
    if season_length <= 0:
        raise ValueError("season_length must be positive")
    values = _training_values(train, target, minimum_size=season_length)
    forecast = [float(values[-season_length + (i % season_length)]) for i in range(len(template))]
    return pd.Series(forecast, index=template.index, dtype=float)
