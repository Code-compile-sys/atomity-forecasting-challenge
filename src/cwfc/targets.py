"""Target construction and time-series split utilities for the challenge."""

from __future__ import annotations

import pandas as pd


def build_targets(readings: pd.DataFrame) -> pd.DataFrame:
    """Build aggregate time-series targets from VM-level CPU readings.

    The function groups the raw Azure VM-level readings by timestamp, counts the
    number of distinct active VMs with valid average-CPU observations, and
    aggregates their average CPU utilization into a single time series.
    """
    if readings.empty:
        raise ValueError("readings must not be empty")

    required_cols = {"timestamp", "vm_id", "avg_cpu"}
    missing = sorted(required_cols - set(readings.columns))
    if missing:
        raise ValueError(f"readings is missing required columns: {missing}")

    cleaned = readings.copy()
    cleaned["timestamp"] = pd.to_numeric(cleaned["timestamp"], errors="coerce")
    cleaned["avg_cpu"] = pd.to_numeric(cleaned["avg_cpu"], errors="coerce")
    cleaned = cleaned.dropna(subset=["timestamp", "vm_id", "avg_cpu"]).copy()

    grouped = (
        cleaned.groupby("timestamp", as_index=False)
        .agg(
            active_vm_count=("vm_id", lambda s: s.nunique()),
            aggregate_cpu_utilization=("avg_cpu", "mean"),
        )
        .rename(columns={"timestamp": "timestamp"})
    )

    grouped["datetime"] = pd.to_datetime(grouped["timestamp"], unit="s", utc=True)
    grouped = grouped[["timestamp", "datetime", "active_vm_count", "aggregate_cpu_utilization"]]
    grouped = grouped.sort_values("timestamp", ascending=True).reset_index(drop=True)
    return grouped


def split_train_holdout(df: pd.DataFrame, holdout: int = 24) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a time series into train and final holdout segments.

    The last ``holdout`` rows are treated as the holdout window, and the earlier
    rows are kept for model fitting.
    """
    if holdout < 0:
        raise ValueError("holdout must be non-negative")
    if holdout > len(df):
        raise ValueError("holdout cannot exceed the length of the dataframe")

    train_df = df.iloc[:-holdout] if holdout else df.copy()
    holdout_df = df.iloc[-holdout:] if holdout else df.iloc[0:0].copy()
    return train_df, holdout_df
