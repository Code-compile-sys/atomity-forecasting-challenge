"""Generate challenge holdout forecasts from the command line."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from cwfc.baselines import last_12_mean_forecast
from cwfc.data import load_data
from cwfc.evaluate import ForecastFunction, evaluate_forecasters
from cwfc.models import damped_exponential_smoothing_forecast, ridge_forecast
from cwfc.model_select import select_best
from cwfc.targets import build_targets


_TARGETS = ("active_vm_count", "aggregate_cpu_utilization")
_OUTPUT_COLUMNS = ("timestamp", *_TARGETS)
_DATETIME_COLUMN = "datetime"
_CV_MIN_TRAIN_SIZE = 50
_CV_HORIZON = 24
_CV_STEP = 6
_MIN_CV_ROWS = _CV_MIN_TRAIN_SIZE + _CV_HORIZON
_FORECASTERS: Mapping[str, ForecastFunction] = {
    "Last12Mean": last_12_mean_forecast,
    "Ridge": ridge_forecast,
    "DampedETS": damped_exponential_smoothing_forecast,
}


def _forecast_one_target(
    train: pd.DataFrame,
    future: pd.DataFrame,
    target: str,
) -> np.ndarray:
    """Select from training-only CV, or use the last-12 baseline if CV cannot run."""
    if train.empty:
        raise ValueError("at least one training timestamp is required to forecast")
    if len(train) < _MIN_CV_ROWS:
        values = pd.to_numeric(train[target], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"training target {target} must contain finite values")
        window = values[-min(len(values), 12) :]
        return np.full(len(future), float(window.mean()), dtype=float)

    forecast_template = future.drop(columns=list(_TARGETS), errors="ignore")
    cv_results = evaluate_forecasters(
        train,
        target,
        datetime_col=_DATETIME_COLUMN,
        min_train_size=_CV_MIN_TRAIN_SIZE,
        horizon=_CV_HORIZON,
        step=_CV_STEP,
    )
    selected_model = select_best(cv_results)
    forecast = np.asarray(
        _FORECASTERS[selected_model](train, forecast_template, target=target),
        dtype=float,
    )
    if forecast.shape != (len(future),):
        raise ValueError(
            f"selected model {selected_model} returned an invalid shape for {target}"
        )
    if not np.isfinite(forecast).all():
        raise ValueError(f"selected model {selected_model} returned non-finite {target} values")
    if target == "active_vm_count":
        return np.maximum(forecast, 0)
    if target == "aggregate_cpu_utilization":
        return np.clip(forecast, 0, 100)
    return forecast


def generate_predictions(targets: pd.DataFrame, holdout: int = 24) -> pd.DataFrame:
    """Return validated point forecasts and raw timestamps for the final horizon."""
    if holdout <= 0:
        raise ValueError("holdout must be a positive integer")
    required = {_DATETIME_COLUMN, "timestamp", *_TARGETS}
    missing = sorted(required - set(targets.columns))
    if missing:
        raise ValueError(f"targets are missing required columns: {missing}")
    if len(targets) < 2:
        raise ValueError("at least two timestamps are required for a train/holdout split")

    ordered = targets.copy()
    ordered[_DATETIME_COLUMN] = pd.to_datetime(ordered[_DATETIME_COLUMN], utc=True)
    if ordered[_DATETIME_COLUMN].isna().any():
        raise ValueError(f"{_DATETIME_COLUMN} must not contain missing values")
    ordered = ordered.sort_values(_DATETIME_COLUMN, kind="stable").reset_index(drop=True)
    if ordered[_DATETIME_COLUMN].duplicated().any():
        raise ValueError(f"{_DATETIME_COLUMN} must contain unique timestamps")

    effective_holdout = min(holdout, len(ordered) - 1)
    train = ordered.iloc[:-effective_holdout].copy()
    future = ordered.iloc[-effective_holdout:].copy()
    prediction = future.loc[:, ["timestamp"]].copy()
    for target in _TARGETS:
        prediction[target] = _forecast_one_target(train, future, target)

    _validate_predictions(prediction, future)
    return prediction.loc[:, _OUTPUT_COLUMNS]


def _validate_predictions(predictions: pd.DataFrame, future: pd.DataFrame) -> None:
    """Fail with clear errors when the generated submission violates its contract."""
    if tuple(predictions.columns) != _OUTPUT_COLUMNS:
        raise ValueError(f"prediction columns must be exactly: {list(_OUTPUT_COLUMNS)}")
    if len(predictions) != len(future):
        raise ValueError("prediction row count does not match the holdout horizon")
    timestamps = pd.to_numeric(predictions["timestamp"], errors="coerce")
    expected_timestamps = pd.to_numeric(future["timestamp"], errors="coerce")
    if timestamps.isna().any() or not np.isfinite(timestamps.to_numpy(dtype=float)).all():
        raise ValueError("prediction timestamps must be finite numeric raw timestamps")
    if timestamps.duplicated().any():
        raise ValueError("prediction timestamps must not contain duplicates")
    if not timestamps.is_monotonic_increasing:
        raise ValueError("prediction timestamps must be in chronological order")
    if not timestamps.reset_index(drop=True).equals(
        expected_timestamps.reset_index(drop=True)
    ):
        raise ValueError("prediction timestamps do not match the holdout timestamps")

    for target in _TARGETS:
        values = pd.to_numeric(predictions[target], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"{target} predictions must all be finite numeric values")
    if (predictions["active_vm_count"] < 0).any():
        raise ValueError("active_vm_count predictions must be non-negative")
    cpu = predictions["aggregate_cpu_utilization"]
    if ((cpu < 0) | (cpu > 100)).any():
        raise ValueError("aggregate_cpu_utilization predictions must be in [0, 100]")


def main() -> None:
    """Parse CLI arguments and write the validated prediction CSV."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="directory with challenge CSVs")
    parser.add_argument("--output", type=Path, required=True, help="output predictions CSV path")
    parser.add_argument("--holdout", type=int, default=24, help="number of final timestamps to forecast")
    args = parser.parse_args()
    if args.holdout <= 0:
        parser.error("--holdout must be a positive integer")

    try:
        readings, _ = load_data(args.data)
        targets = build_targets(readings)
        predictions = generate_predictions(targets, holdout=args.holdout)
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))

    predictions.to_csv(args.output, index=False)
    print(f"Wrote {len(predictions)} holdout predictions to {args.output}")


if __name__ == "__main__":
    main()
