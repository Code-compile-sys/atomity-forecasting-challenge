import numpy as np
import pandas as pd

from cwfc.evaluate import evaluate_forecasters
from cwfc.model_select import select_best


def test_select_best_chooses_baseline_for_constant_series():
    frame = pd.DataFrame(
        {
            "datetime": pd.date_range("2026-01-01", periods=86, freq="5min", tz="UTC"),
            "active_vm_count": np.full(86, 10_000.0),
        }
    )
    cv_results = evaluate_forecasters(frame, target="active_vm_count")

    assert select_best(cv_results) == "Last12Mean"


def test_select_best_requires_at_least_three_percent_gain_and_breaks_ties_by_simplicity():
    cv_results = pd.DataFrame(
        {
            "model": [
                "Last12Mean", "Last12Mean",
                "DampedETS", "DampedETS",
                "Ridge", "Ridge",
            ],
            "fold": [1, 2, 1, 2, 1, 2],
            "mae": [10.0, 10.0, 9.0, 9.0, 9.0, 9.0],
        }
    )

    assert select_best(cv_results) == "DampedETS"

    cv_results.loc[cv_results["model"] == "DampedETS", "mae"] = 9.8
    cv_results.loc[cv_results["model"] == "Ridge", "mae"] = 9.8
    assert select_best(cv_results) == "Last12Mean"
