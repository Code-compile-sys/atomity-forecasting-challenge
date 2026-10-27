"""CV-based capacity policy comparison for aggregate CPU utilization."""

from __future__ import annotations

import numpy as np
import pandas as pd


_POLICY_FRACTIONS = (
    ("A: forecast", 0.0),
    ("B: half upper-interval margin", 0.5),
    ("C: full upper-interval margin", 1.0),
)


def compare_capacity_policies(cv_forecasts: pd.DataFrame) -> pd.DataFrame:
    """Compare three provisioning policies using selected-model CV forecasts.

    ``cv_forecasts`` must contain only rolling-origin CV cases for
    ``aggregate_cpu_utilization`` with ``actual`` and ``forecast`` columns.
    The pooled 90th-percentile actual-minus-forecast residual is the upper
    80% interval margin, matching the interval construction used elsewhere.
    """
    required = {"actual", "forecast"}
    missing = sorted(required - set(cv_forecasts.columns))
    if missing:
        raise ValueError(f"cv_forecasts is missing required columns: {missing}")
    if cv_forecasts.empty:
        raise ValueError("cv_forecasts must not be empty")
    if "target" in cv_forecasts.columns:
        targets = set(cv_forecasts["target"].dropna())
        if targets != {"aggregate_cpu_utilization"}:
            raise ValueError("cv_forecasts must contain aggregate CPU cases only")

    actual = pd.to_numeric(cv_forecasts["actual"], errors="coerce").to_numpy(dtype=float)
    forecast = pd.to_numeric(
        cv_forecasts["forecast"], errors="coerce"
    ).to_numpy(dtype=float)
    valid = np.isfinite(actual) & np.isfinite(forecast)
    if not valid.all():
        raise ValueError("CV actual and forecast values must be finite")

    residuals = actual - forecast
    upper_interval_margin = float(np.quantile(residuals, 0.9))
    rows: list[dict[str, float | str]] = []
    for policy, fraction in _POLICY_FRACTIONS:
        provision = forecast + fraction * upper_interval_margin
        under_provisioned = actual > provision
        adequately_provisioned = ~under_provisioned
        shortfalls = actual[under_provisioned] - provision[under_provisioned]
        headroom = provision[adequately_provisioned] - actual[adequately_provisioned]
        rows.append(
            {
                "policy": policy,
                "under_provisioning_rate": float(under_provisioned.mean()),
                "average_shortfall": (
                    float(shortfalls.mean()) if shortfalls.size else 0.0
                ),
                "wasted_headroom": float(headroom.mean()) if headroom.size else 0.0,
            }
        )
    return pd.DataFrame(rows)
