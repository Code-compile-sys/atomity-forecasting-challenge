import numpy as np
import pandas as pd
import pytest

from cwfc.policy import compare_capacity_policies


def test_compare_capacity_policies_uses_pooled_upper_residual_margin():
    cv_forecasts = pd.DataFrame(
        {
            "target": ["aggregate_cpu_utilization"] * 4,
            "actual": [10.0, 10.0, 10.0, 10.0],
            "forecast": [8.0, 9.0, 10.0, 12.0],
        }
    )

    result = compare_capacity_policies(cv_forecasts)

    assert result["policy"].tolist() == [
        "A: forecast",
        "B: half upper-interval margin",
        "C: full upper-interval margin",
    ]
    assert len(result) == 3
    assert result.loc[0, "under_provisioning_rate"] == pytest.approx(0.5)
    assert result.loc[0, "average_shortfall"] == pytest.approx(1.5)
    assert result.loc[0, "wasted_headroom"] == pytest.approx(1.0)


def test_compare_capacity_policies_handles_no_under_or_over_provisioning():
    cv_forecasts = pd.DataFrame(
        {
            "actual": [5.0, 5.0, 5.0],
            "forecast": [5.0, 5.0, 5.0],
        }
    )

    result = compare_capacity_policies(cv_forecasts)

    assert np.isfinite(result.select_dtypes("number").to_numpy()).all()
    assert result["under_provisioning_rate"].eq(0.0).all()
    assert result["average_shortfall"].eq(0.0).all()
    assert result["wasted_headroom"].eq(0.0).all()


def test_compare_capacity_policies_rejects_non_cpu_data():
    cv_forecasts = pd.DataFrame(
        {
            "target": ["active_vm_count"],
            "actual": [10.0],
            "forecast": [10.0],
        }
    )

    with pytest.raises(ValueError, match="aggregate CPU"):
        compare_capacity_policies(cv_forecasts)
