"""Rolling-origin evaluation utilities for time-series forecasts."""

from __future__ import annotations

from collections.abc import Callable, Mapping

import numpy as np
import pandas as pd

from cwfc.baselines import (
    last_6_mean_forecast,
    last_12_mean_forecast,
    persistence_forecast,
    seasonal_naive_forecast,
    training_mean_forecast,
)
from cwfc.metrics import mae, normalized_mae, rmse, smape
from cwfc.validation import rolling_origin_splits


ForecastFunction = Callable[..., pd.Series | np.ndarray | list[float]]


def evaluate_rolling_origin(
    frame: pd.DataFrame,
    target: str,
    forecast_fn: ForecastFunction,
    *,
    datetime_col: str = "datetime",
    horizon: int,
    min_train_size: int,
    step: int | None = None,
    forecast_kwargs: Mapping[str, object] | None = None,
    baseline_forecast_fn: ForecastFunction = persistence_forecast,
    baseline_forecast_kwargs: Mapping[str, object] | None = None,
) -> pd.DataFrame:
    """Evaluate a forecasting function over expanding chronological folds.

    The input is sorted by ``datetime_col`` before folds are formed. Each
    forecast function receives ``(train_frame, validation_frame, target=...)``.
    The returned table contains one row per fold and a final mean row.
    """
    required = {datetime_col, target}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"frame is missing required columns: {missing}")
    if frame.empty:
        raise ValueError("frame must not be empty")
    if horizon <= 0 or min_train_size <= 0:
        raise ValueError("horizon and min_train_size must be positive")
    if step is not None and step <= 0:
        raise ValueError("step must be positive")

    ordered = frame.copy()
    ordered[datetime_col] = pd.to_datetime(ordered[datetime_col], utc=True)
    if ordered[datetime_col].isna().any():
        raise ValueError(f"{datetime_col} must not contain missing values")
    ordered = ordered.sort_values(datetime_col, kind="stable").reset_index(drop=True)
    if ordered[datetime_col].duplicated().any():
        raise ValueError(f"{datetime_col} must contain unique timestamps")

    target_values = pd.to_numeric(ordered[target], errors="coerce").to_numpy(dtype=float)
    if np.isinf(target_values).any():
        raise ValueError(f"{target} must not contain infinite values")
    ordered[target] = target_values

    splits = rolling_origin_splits(
        ordered,
        timestamp_col=datetime_col,
        horizon=horizon,
        min_train_size=min_train_size,
        step=step,
    )
    if not splits:
        raise ValueError("not enough observations to create a rolling-origin fold")

    rows: list[dict[str, object]] = []
    kwargs = dict(forecast_kwargs or {})
    for fold_number, split in enumerate(splits, start=1):
        train = ordered.loc[ordered[datetime_col] <= split.train_end].copy()
        validation = ordered.loc[
            ordered[datetime_col].between(split.validation_start, split.validation_end)
        ].copy()
        if not np.isfinite(train[target].to_numpy(dtype=float)).all():
            raise ValueError(f"training target contains missing values in fold {fold_number}")
        prediction = np.asarray(
            forecast_fn(train, validation, target=target, **kwargs),
            dtype=float,
        )
        actual = validation[target].to_numpy(dtype=float)
        if prediction.shape != actual.shape:
            raise ValueError(
                f"forecast for fold {fold_number} has shape {prediction.shape}; "
                f"expected {actual.shape}"
            )
        if not np.isfinite(prediction).all():
            raise ValueError(f"forecast for fold {fold_number} contains non-finite values")
        baseline_prediction = np.asarray(
            baseline_forecast_fn(
                train,
                validation,
                target=target,
                **dict(baseline_forecast_kwargs or {}),
            ),
            dtype=float,
        )
        if baseline_prediction.shape != actual.shape:
            raise ValueError(
                f"baseline forecast for fold {fold_number} has shape "
                f"{baseline_prediction.shape}; expected {actual.shape}"
            )
        if not np.isfinite(baseline_prediction).all():
            raise ValueError(
                f"baseline forecast for fold {fold_number} contains non-finite values"
            )

        rows.append(
            {
                "fold": fold_number,
                "train_size": len(train),
                "validation_size": len(validation),
                "train_end": split.train_end,
                "validation_start": split.validation_start,
                "validation_end": split.validation_end,
                "mae": mae(actual, prediction),
                "rmse": rmse(actual, prediction),
                "normalized_mae": normalized_mae(
                    actual, prediction, baseline_prediction
                ),
                "smape": smape(actual, prediction),
                "scored_observations": int(np.isfinite(actual).sum()),
            }
        )

    results = pd.DataFrame(rows)
    mean_row: dict[str, object] = {
        "fold": "mean",
        "train_size": np.nan,
        "validation_size": np.nan,
        "train_end": pd.NaT,
        "validation_start": pd.NaT,
        "validation_end": pd.NaT,
        "scored_observations": np.nan,
    }
    mean_row.update(
        results[["mae", "rmse", "normalized_mae", "smape"]].mean().to_dict()
    )
    return pd.concat([results, pd.DataFrame([mean_row])], ignore_index=True)


def evaluate_baselines(
    frame: pd.DataFrame,
    target: str,
    *,
    datetime_col: str = "datetime",
    horizon: int = 24,
    min_train_size: int = 50,
    step: int = 6,
    season_length: int = 12,
) -> pd.DataFrame:
    """Return every rolling-origin score with its fold and baseline name."""
    baseline_specs: list[tuple[str, ForecastFunction, Mapping[str, object]]] = [
        ("Persistence", persistence_forecast, {}),
        ("Last6Mean", last_6_mean_forecast, {}),
        ("Last12Mean", last_12_mean_forecast, {}),
        ("TrainingMean", training_mean_forecast, {}),
        ("SeasonalNaive", seasonal_naive_forecast, {"season_length": season_length}),
    ]
    results: list[pd.DataFrame] = []
    for baseline_name, forecast_fn, forecast_kwargs in baseline_specs:
        fold_results = evaluate_rolling_origin(
            frame,
            target,
            forecast_fn,
            datetime_col=datetime_col,
            horizon=horizon,
            min_train_size=min_train_size,
            step=step,
            forecast_kwargs=forecast_kwargs,
            baseline_forecast_fn=persistence_forecast,
        )
        fold_results = fold_results.loc[fold_results["fold"] != "mean"].copy()
        fold_results.insert(1, "baseline", baseline_name)
        results.append(fold_results)
    return pd.concat(results, ignore_index=True)
