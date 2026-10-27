"""Pooled-residual forecast intervals for the required workload targets."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from cwfc.baselines import last_12_mean_forecast
from cwfc.evaluate import ForecastFunction, evaluate_forecasters
from cwfc.model_select import select_best
from cwfc.models import damped_exponential_smoothing_forecast, ridge_forecast
from cwfc.validation import rolling_origin_splits


_FORECASTERS: Mapping[str, ForecastFunction] = {
    "Last12Mean": last_12_mean_forecast,
    "Ridge": ridge_forecast,
    "DampedETS": damped_exponential_smoothing_forecast,
}
_TARGET_BOUNDS: Mapping[str, tuple[float, float | None]] = {
    "active_vm_count": (0.0, None),
    "aggregate_cpu_utilization": (0.0, 100.0),
}


def _pooled_cv_residuals(
    train_frame: pd.DataFrame,
    target: str,
    model: str,
    *,
    datetime_col: str,
    horizon: int,
    min_train_size: int,
    step: int,
) -> np.ndarray:
    """Pool actual-minus-forecast residuals from every CV fold and step."""
    ordered = train_frame.copy()
    ordered[datetime_col] = pd.to_datetime(ordered[datetime_col], utc=True)
    if ordered[datetime_col].isna().any():
        raise ValueError(f"{datetime_col} must not contain missing values")
    ordered = ordered.sort_values(datetime_col, kind="stable").reset_index(drop=True)
    values = pd.to_numeric(ordered[target], errors="coerce").to_numpy(dtype=float)
    if np.isinf(values).any():
        raise ValueError(f"{target} must not contain infinite values")
    ordered[target] = values

    splits = rolling_origin_splits(
        ordered,
        timestamp_col=datetime_col,
        horizon=horizon,
        min_train_size=min_train_size,
        step=step,
    )
    if not splits:
        raise ValueError("not enough training observations to pool CV residuals")

    forecaster = _FORECASTERS[model]
    template_columns = [datetime_col]
    if "timestamp" in ordered.columns:
        template_columns.insert(0, "timestamp")
    residuals: list[np.ndarray] = []
    for split in splits:
        train = ordered.loc[ordered[datetime_col] <= split.train_end].copy()
        validation = ordered.loc[
            ordered[datetime_col].between(split.validation_start, split.validation_end)
        ].copy()
        if not np.isfinite(train[target].to_numpy(dtype=float)).all():
            raise ValueError("CV training targets must be finite")
        template = validation.loc[:, template_columns]
        prediction = np.asarray(forecaster(train, template, target=target), dtype=float)
        actual = validation[target].to_numpy(dtype=float)
        if prediction.shape != actual.shape:
            raise ValueError(
                f"forecast for model {model} has shape {prediction.shape}; "
                f"expected {actual.shape}"
            )
        valid = np.isfinite(actual) & np.isfinite(prediction)
        if valid.any():
            residuals.append(actual[valid] - prediction[valid])

    if not residuals:
        raise ValueError("CV folds contain no valid residuals")
    return np.concatenate(residuals)


def forecast_targets_with_intervals(
    train_frame: pd.DataFrame,
    future_frame: pd.DataFrame,
    *,
    datetime_col: str = "datetime",
    horizon: int = 24,
    min_train_size: int = 50,
    step: int = 6,
) -> pd.DataFrame:
    """Select and forecast both targets with pooled-residual 80% intervals.

    Pass training observations separately from a future template. Any target
    columns in ``future_frame`` are deliberately excluded from forecast calls.
    The returned long-form interval table is separate from the required
    point-prediction CSV.
    """
    required_train = {datetime_col, *_TARGET_BOUNDS}
    missing_train = sorted(required_train - set(train_frame.columns))
    if missing_train:
        raise ValueError(f"train_frame is missing required columns: {missing_train}")
    if datetime_col not in future_frame.columns or "timestamp" not in future_frame.columns:
        raise ValueError(f"future_frame must contain timestamp and {datetime_col}")
    if train_frame.empty or future_frame.empty:
        raise ValueError("train_frame and future_frame must not be empty")
    if horizon <= 0 or min_train_size <= 0 or step <= 0:
        raise ValueError("horizon, min_train_size, and step must be positive")
    if len(future_frame) != horizon:
        raise ValueError(f"future_frame must contain exactly {horizon} timestamps")

    train_times = pd.to_datetime(train_frame[datetime_col], utc=True)
    future_times = pd.to_datetime(future_frame[datetime_col], utc=True)
    if train_times.isna().any() or future_times.isna().any():
        raise ValueError(f"{datetime_col} must not contain missing values")
    if train_times.duplicated().any() or future_times.duplicated().any():
        raise ValueError(f"{datetime_col} must contain unique timestamps")
    if future_times.min() <= train_times.max():
        raise ValueError("future timestamps must follow all training timestamps")

    ordered_train = train_frame.copy()
    ordered_train[datetime_col] = train_times
    ordered_train = ordered_train.sort_values(datetime_col, kind="stable").reset_index(
        drop=True
    )
    template_columns = ["timestamp", datetime_col]
    forecast_template = future_frame.loc[:, template_columns].copy()
    forecast_template[datetime_col] = future_times
    forecast_template = forecast_template.sort_values(
        datetime_col, kind="stable"
    ).reset_index(drop=True)

    output: list[pd.DataFrame] = []
    for target, (lower_bound, upper_bound) in _TARGET_BOUNDS.items():
        cv_results = evaluate_forecasters(
            ordered_train,
            target,
            datetime_col=datetime_col,
            horizon=horizon,
            min_train_size=min_train_size,
            step=step,
        )
        winner = select_best(cv_results)
        residuals = _pooled_cv_residuals(
            ordered_train,
            target,
            winner,
            datetime_col=datetime_col,
            horizon=horizon,
            min_train_size=min_train_size,
            step=step,
        )
        lower_error, upper_error = np.quantile(residuals, [0.1, 0.9])
        prediction = np.asarray(
            _FORECASTERS[winner](ordered_train, forecast_template, target=target),
            dtype=float,
        )
        if prediction.shape != (horizon,) or not np.isfinite(prediction).all():
            raise ValueError(f"selected model {winner} produced invalid forecasts")

        prediction = np.clip(prediction, lower_bound, upper_bound)
        lower = np.clip(prediction + lower_error, lower_bound, upper_bound)
        upper = np.clip(prediction + upper_error, lower_bound, upper_bound)
        result = forecast_template.copy()
        result["target"] = target
        result["model"] = winner
        result["forecast"] = prediction
        result["lower_80"] = lower
        result["upper_80"] = upper
        output.append(result)

    return pd.concat(output, ignore_index=True)
