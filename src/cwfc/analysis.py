"""Training-only cross-validation error analysis for selected forecasts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd

from cwfc.baselines import last_12_mean_forecast
from cwfc.evaluate import ForecastFunction, evaluate_forecasters
from cwfc.metrics import mae, rmse
from cwfc.model_select import select_best
from cwfc.models import damped_exponential_smoothing_forecast, ridge_forecast
from cwfc.validation import rolling_origin_splits


_FORECASTERS: Final[dict[str, ForecastFunction]] = {
    "Last12Mean": last_12_mean_forecast,
    "Ridge": ridge_forecast,
    "DampedETS": damped_exponential_smoothing_forecast,
}


@dataclass(frozen=True)
class CVErrorAnalysis:
    """Selected-model CV forecasts and their requested error summaries."""

    selected_model: str
    peak_threshold: float
    cv_forecasts: pd.DataFrame
    peak_summary: pd.DataFrame
    fold_mae: pd.DataFrame
    horizon_mae: pd.DataFrame


def analyze_selected_model_cv(
    train_frame: pd.DataFrame,
    *,
    target: str = "aggregate_cpu_utilization",
    datetime_col: str = "datetime",
    horizon: int = 24,
    min_train_size: int = 50,
    step: int = 6,
) -> CVErrorAnalysis:
    """Analyze the selected model using rolling-origin folds in training only.

    The peak threshold is the 90th percentile of the supplied training target.
    Residuals are signed as actual minus forecast; no holdout frame is accepted.
    """
    required = {datetime_col, target}
    missing = sorted(required - set(train_frame.columns))
    if missing:
        raise ValueError(f"train_frame is missing required columns: {missing}")
    if train_frame.empty:
        raise ValueError("train_frame must not be empty")

    ordered = train_frame.copy()
    ordered[datetime_col] = pd.to_datetime(ordered[datetime_col], utc=True)
    if ordered[datetime_col].isna().any():
        raise ValueError(f"{datetime_col} must not contain missing values")
    ordered = ordered.sort_values(datetime_col, kind="stable").reset_index(drop=True)
    if ordered[datetime_col].duplicated().any():
        raise ValueError(f"{datetime_col} must contain unique timestamps")
    target_values = pd.to_numeric(ordered[target], errors="coerce").to_numpy(dtype=float)
    if np.isinf(target_values).any():
        raise ValueError(f"{target} must not contain infinite values")
    finite_training_targets = target_values[np.isfinite(target_values)]
    if finite_training_targets.size == 0:
        raise ValueError(f"{target} must contain finite training observations")
    peak_threshold = float(np.quantile(finite_training_targets, 0.9))

    cv_results = evaluate_forecasters(
        ordered,
        target,
        datetime_col=datetime_col,
        horizon=horizon,
        min_train_size=min_train_size,
        step=step,
    )
    selected_model = select_best(cv_results)
    splits = rolling_origin_splits(
        ordered,
        timestamp_col=datetime_col,
        horizon=horizon,
        min_train_size=min_train_size,
        step=step,
    )
    if not splits:
        raise ValueError("not enough training observations for CV error analysis")

    forecast_fn = _FORECASTERS[selected_model]
    forecast_rows: list[dict[str, object]] = []
    for fold_number, split in enumerate(splits, start=1):
        fold_train = ordered.loc[ordered[datetime_col] <= split.train_end].copy()
        validation = ordered.loc[
            ordered[datetime_col].between(split.validation_start, split.validation_end)
        ].copy()
        if not np.isfinite(fold_train[target].to_numpy(dtype=float)).all():
            raise ValueError(f"training target contains missing values in fold {fold_number}")
        template = validation.drop(columns=[target])
        prediction = np.asarray(
            forecast_fn(fold_train, template, target=target),
            dtype=float,
        )
        actual = validation[target].to_numpy(dtype=float)
        if prediction.shape != actual.shape:
            raise ValueError(
                f"forecast for fold {fold_number} has shape {prediction.shape}; "
                f"expected {actual.shape}"
            )
        valid = np.isfinite(actual) & np.isfinite(prediction)
        for horizon_step, row_index in enumerate(validation.index, start=1):
            if not valid[horizon_step - 1]:
                continue
            actual_value = float(actual[horizon_step - 1])
            forecast_value = float(prediction[horizon_step - 1])
            row: dict[str, object] = {
                "model": selected_model,
                "fold": fold_number,
                "horizon_step": horizon_step,
                datetime_col: validation.loc[row_index, datetime_col],
                "actual": actual_value,
                "forecast": forecast_value,
                "residual": actual_value - forecast_value,
                "is_peak": actual_value > peak_threshold,
            }
            if "timestamp" in validation.columns:
                row["timestamp"] = validation.loc[row_index, "timestamp"]
            forecast_rows.append(row)

    cv_forecasts = pd.DataFrame(forecast_rows)
    if cv_forecasts.empty:
        raise ValueError("CV folds contain no valid actual/forecast pairs")

    peak_rows: list[dict[str, object]] = []
    for is_peak, label in ((True, "peak"), (False, "non_peak")):
        group = cv_forecasts.loc[cv_forecasts["is_peak"] == is_peak]
        if group.empty:
            peak_rows.append(
                {
                    "period": label,
                    "observations": 0,
                    "mae": np.nan,
                    "rmse": np.nan,
                    "bias": np.nan,
                }
            )
            continue
        peak_rows.append(
            {
                "period": label,
                "observations": len(group),
                "mae": mae(group["actual"], group["forecast"]),
                "rmse": rmse(group["actual"], group["forecast"]),
                "bias": float(group["residual"].mean()),
            }
        )
    peak_summary = pd.DataFrame(peak_rows)

    fold_rows = []
    for fold_number, group in cv_forecasts.groupby("fold", sort=True):
        fold_rows.append(
            {
                "fold": fold_number,
                "mae": mae(group["actual"], group["forecast"]),
                "observations": len(group),
            }
        )
    fold_mae = pd.DataFrame(fold_rows).sort_values(
        ["mae", "fold"], ascending=[False, True], kind="stable"
    ).reset_index(drop=True)

    horizon_rows = []
    for horizon_step, group in cv_forecasts.groupby("horizon_step", sort=True):
        horizon_rows.append(
            {
                "horizon_step": horizon_step,
                "mae": mae(group["actual"], group["forecast"]),
                "observations": len(group),
            }
        )
    horizon_mae = pd.DataFrame(horizon_rows)

    return CVErrorAnalysis(
        selected_model=selected_model,
        peak_threshold=peak_threshold,
        cv_forecasts=cv_forecasts,
        peak_summary=peak_summary,
        fold_mae=fold_mae,
        horizon_mae=horizon_mae,
    )
