from __future__ import annotations

import pandas as pd

from cwfc.targets import build_targets, split_train_holdout


def test_build_targets_aggregates_valid_vm_observations():
    readings = pd.DataFrame(
        {
            "timestamp": [100, 100, 100, 200, 200, 200],
            "vm_id": ["vm1", "vm1", "vm2", "vm1", "vm2", "vm3"],
            "avg_cpu": [10.0, 20.0, 30.0, None, 50.0, 60.0],
        }
    )

    result = build_targets(readings)

    assert list(result.columns) == [
        "timestamp",
        "datetime",
        "active_vm_count",
        "aggregate_cpu_utilization",
    ]
    assert result["timestamp"].tolist() == [100, 200]
    assert result["active_vm_count"].tolist() == [2, 2]
    assert result["aggregate_cpu_utilization"].tolist() == [20.0, 55.0]
    assert isinstance(result["datetime"].dtype, pd.DatetimeTZDtype)


def test_split_train_holdout_keeps_order_and_last_rows_as_holdout():
    df = pd.DataFrame({"timestamp": list(range(10)), "value": list(range(10))})

    train_df, holdout_df = split_train_holdout(df, holdout=3)

    pd.testing.assert_frame_equal(train_df, df.iloc[:-3])
    pd.testing.assert_frame_equal(holdout_df, df.iloc[-3:])


def test_split_train_holdout_without_holdout_returns_full_and_empty_frames():
    df = pd.DataFrame({"timestamp": [1, 2], "value": [3, 4]})

    train_df, holdout_df = split_train_holdout(df, holdout=0)

    pd.testing.assert_frame_equal(train_df, df)
    assert holdout_df.empty
