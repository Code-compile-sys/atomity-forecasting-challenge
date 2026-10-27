import numpy as np
import pandas as pd

from cwfc.intervals import forecast_targets_with_intervals


def test_constant_targets_select_baseline_and_have_zero_width_intervals():
    train_frame = pd.DataFrame(
        {
            "timestamp": np.arange(86) * 300,
            "datetime": pd.date_range(
                "2026-01-01", periods=86, freq="5min", tz="UTC"
            ),
            "active_vm_count": np.full(86, 10_000.0),
            "aggregate_cpu_utilization": np.full(86, 40.0),
        }
    )
    future_frame = pd.DataFrame(
        {
            "timestamp": np.arange(86, 110) * 300,
            "datetime": pd.date_range(
                "2026-01-01 07:10", periods=24, freq="5min", tz="UTC"
            ),
            "active_vm_count": np.full(24, -999.0),
            "aggregate_cpu_utilization": np.full(24, 999.0),
        }
    )

    intervals = forecast_targets_with_intervals(train_frame, future_frame)

    assert len(intervals) == 48
    assert set(intervals["target"]) == {
        "active_vm_count",
        "aggregate_cpu_utilization",
    }
    assert intervals.groupby("target")["model"].unique().apply(list).to_dict() == {
        "active_vm_count": ["Last12Mean"],
        "aggregate_cpu_utilization": ["Last12Mean"],
    }
    assert intervals.groupby("target")["timestamp"].apply(
        lambda values: values.is_monotonic_increasing
    ).all()
    assert intervals.loc[
        intervals["target"] == "active_vm_count", "lower_80"
    ].eq(10_000.0).all()
    assert intervals.loc[
        intervals["target"] == "aggregate_cpu_utilization", "lower_80"
    ].eq(40.0).all()
    np.testing.assert_array_equal(intervals["lower_80"], intervals["upper_80"])
    assert intervals.loc[
        intervals["target"] == "active_vm_count", "lower_80"
    ].ge(0).all()
    cpu_interval = intervals.loc[
        intervals["target"] == "aggregate_cpu_utilization", ["lower_80", "upper_80"]
    ]
    assert cpu_interval.ge(0).all().all()
    assert cpu_interval.le(100).all().all()
    assert intervals.loc[
        intervals["target"] == "active_vm_count", "forecast"
    ].ge(0).all()
    cpu_forecast = intervals.loc[
        intervals["target"] == "aggregate_cpu_utilization", "forecast"
    ]
    assert cpu_forecast.ge(0).all()
    assert cpu_forecast.le(100).all()


def test_interval_forecasts_are_clipped_to_target_ranges():
    train_frame = pd.DataFrame(
        {
            "timestamp": np.arange(74) * 300,
            "datetime": pd.date_range(
                "2026-01-01", periods=74, freq="5min", tz="UTC"
            ),
            "active_vm_count": np.full(74, -5.0),
            "aggregate_cpu_utilization": np.full(74, 120.0),
        }
    )
    future_frame = pd.DataFrame(
        {
            "timestamp": np.arange(74, 98) * 300,
            "datetime": pd.date_range(
                "2026-01-01 06:10", periods=24, freq="5min", tz="UTC"
            ),
        }
    )

    intervals = forecast_targets_with_intervals(train_frame, future_frame)

    active_count = intervals.loc[intervals["target"] == "active_vm_count"]
    cpu = intervals.loc[intervals["target"] == "aggregate_cpu_utilization"]
    assert active_count[["forecast", "lower_80", "upper_80"]].eq(0).all().all()
    assert cpu[["forecast", "lower_80", "upper_80"]].eq(100).all().all()
