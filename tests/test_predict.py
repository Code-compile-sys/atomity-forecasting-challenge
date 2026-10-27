import sys

import numpy as np
import pandas as pd

from cwfc.predict import generate_predictions, main


def _targets(count: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": np.arange(count, dtype=np.int64) * 300,
            "datetime": pd.date_range(
                "2026-01-01", periods=count, freq="5min", tz="UTC"
            ),
            "active_vm_count": np.full(count, 7.0),
            "aggregate_cpu_utilization": np.full(count, 20.0),
        }
    )


def _write_synthetic_dataset(directory, timestamp_count: int) -> list[int]:
    timestamps = (np.arange(timestamp_count, dtype=np.int64) + 1) * 300
    readings = []
    for index, timestamp in enumerate(timestamps):
        for vm_index in range(index % 3 + 1):
            readings.append(
                {
                    "timestamp": timestamp,
                    "vm_id": f"vm-{vm_index}",
                    "min_cpu": 5.0,
                    "max_cpu": 15.0,
                    "avg_cpu": 10.0 + index % 5,
                }
            )
    pd.DataFrame(readings).to_csv(
        directory / "vm_cpu_readings.csv.gz", index=False, compression="gzip"
    )
    pd.DataFrame(
        {
            "vm_id": ["vm-0", "vm-1", "vm-2"],
            "avg_cpu": [99.0, 99.0, 99.0],
        }
    ).to_csv(directory / "vm_metadata.csv.gz", index=False, compression="gzip")
    return timestamps.tolist()


def _run_cli(monkeypatch, data_dir, output_path):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cwfc.predict",
            "--data",
            str(data_dir),
            "--output",
            str(output_path),
        ],
    )
    main()


def test_generate_predictions_clamps_holdout_and_uses_baseline_on_tiny_data():
    targets = _targets(4)

    predictions = generate_predictions(targets)

    assert predictions.columns.tolist() == [
        "timestamp",
        "active_vm_count",
        "aggregate_cpu_utilization",
    ]
    assert predictions["timestamp"].tolist() == [300, 600, 900]
    assert predictions["active_vm_count"].tolist() == [7.0] * 3
    assert predictions["aggregate_cpu_utilization"].tolist() == [20.0] * 3
    assert np.isfinite(predictions.select_dtypes("number").to_numpy()).all()


def test_cli_output_format_has_exactly_24_valid_rows_and_no_index(tmp_path, monkeypatch):
    data_dir = tmp_path / "challenge_data"
    data_dir.mkdir()
    timestamps = _write_synthetic_dataset(data_dir, timestamp_count=30)
    output_path = tmp_path / "predictions.csv"

    _run_cli(monkeypatch, data_dir, output_path)

    predictions = pd.read_csv(output_path)
    assert predictions.columns.tolist() == [
        "timestamp",
        "active_vm_count",
        "aggregate_cpu_utilization",
    ]
    assert len(predictions) == 24
    assert predictions["timestamp"].tolist() == timestamps[-24:]
    assert predictions["timestamp"].is_unique
    assert predictions["timestamp"].is_monotonic_increasing
    assert np.isfinite(predictions.to_numpy(dtype=float)).all()
    assert predictions["active_vm_count"].ge(0).all()
    assert predictions["aggregate_cpu_utilization"].between(0, 100).all()


def test_cli_smoke_test_runs_end_to_end_on_tiny_varying_vm_dataset(
    tmp_path, monkeypatch
):
    data_dir = tmp_path / "tiny_data"
    data_dir.mkdir()
    timestamps = _write_synthetic_dataset(data_dir, timestamp_count=6)
    output_path = tmp_path / "tiny_predictions.csv"

    readings = pd.read_csv(data_dir / "vm_cpu_readings.csv.gz")
    assert readings.groupby("timestamp")["vm_id"].nunique().nunique() > 1

    _run_cli(monkeypatch, data_dir, output_path)

    predictions = pd.read_csv(output_path)
    assert predictions["timestamp"].tolist() == timestamps[-5:]
    assert len(predictions) == 5
    assert np.isfinite(predictions.to_numpy(dtype=float)).all()
    assert predictions["active_vm_count"].ge(0).all()
    assert predictions["aggregate_cpu_utilization"].between(0, 100).all()


def test_predictions_are_independent_of_last_24_target_values():
    targets = _targets(110)
    altered_holdout = targets.copy()
    altered_holdout.loc[targets.index[-24:], "active_vm_count"] = 9999
    altered_holdout.loc[targets.index[-24:], "aggregate_cpu_utilization"] = -500

    original_predictions = generate_predictions(targets)
    altered_predictions = generate_predictions(altered_holdout)

    pd.testing.assert_frame_equal(original_predictions, altered_predictions)


def test_generate_predictions_uses_cv_and_keeps_raw_holdout_timestamps():
    targets = _targets(110)

    predictions = generate_predictions(targets, holdout=24)

    assert len(predictions) == 24
    assert predictions["timestamp"].tolist() == targets["timestamp"].iloc[-24:].tolist()
    assert predictions["timestamp"].is_unique
    assert predictions["timestamp"].is_monotonic_increasing
    assert predictions["active_vm_count"].ge(0).all()
    assert predictions["aggregate_cpu_utilization"].between(0, 100).all()
