"""Model selection based on training-only cross-validation results."""

from __future__ import annotations

import numpy as np
import pandas as pd


_BASELINE = "Last12Mean"
_SIMPLICITY_ORDER = ("DampedETS", "Ridge")
_CANDIDATES = {_BASELINE, *_SIMPLICITY_ORDER}
_MINIMUM_IMPROVEMENT = 0.03


def select_best(cv_results: pd.DataFrame) -> str:
    """Choose a candidate using CV MAE and a 3% baseline improvement rule.

    The input must contain scores for all three evaluated candidates. A
    ``fold == "mean"`` summary row is ignored when fold-level scores exist.
    """
    required = {"model", "fold", "mae"}
    missing = sorted(required - set(cv_results.columns))
    if missing:
        raise ValueError(f"cv_results is missing required columns: {missing}")
    if cv_results.empty:
        raise ValueError("cv_results must not be empty")

    scored = cv_results.loc[cv_results["fold"] != "mean"].copy()
    if scored.empty:
        scored = cv_results.copy()

    candidates = set(scored["model"].dropna())
    if candidates != _CANDIDATES:
        raise ValueError(
            "cv_results must contain exactly these candidates: "
            f"{sorted(_CANDIDATES)}"
        )

    scored["mae"] = pd.to_numeric(scored["mae"], errors="coerce")
    if scored["mae"].isna().any() or not np.isfinite(scored["mae"]).all():
        raise ValueError("cv_results MAE values must be finite numbers")
    if (scored["mae"] < 0).any():
        raise ValueError("cv_results MAE values must be non-negative")

    mean_mae = scored.groupby("model")["mae"].mean()
    baseline_mae = float(mean_mae[_BASELINE])
    if baseline_mae == 0:
        return _BASELINE

    required_mae = baseline_mae * (1 - _MINIMUM_IMPROVEMENT)
    eligible = [
        model for model in _SIMPLICITY_ORDER
        if float(mean_mae[model]) <= required_mae
    ]
    if not eligible:
        return _BASELINE

    best_mae = min(float(mean_mae[model]) for model in eligible)
    tied = [
        model for model in eligible
        if np.isclose(float(mean_mae[model]), best_mae, rtol=1e-12, atol=1e-12)
    ]
    return next(model for model in _SIMPLICITY_ORDER if model in tied)
