from __future__ import annotations

from pathlib import Path

import pandas as pd

from cwfc.data import load_data


def test_load_data_reads_expected_files(tmp_path: Path):
    readings_path = tmp_path / "vm_cpu_readings.csv.gz"
    metadata_path = tmp_path / "vm_metadata.csv.gz"

    readings_df = pd.DataFrame(
        {
            "timestamp": [1, 2],
            "vm_id": ["vm_1", "vm_2"],
            "min_cpu": [10.0, 20.0],
            "max_cpu": [30.0, 40.0],
            "avg_cpu": [20.0, 30.0],
        }
    )
    metadata_df = pd.DataFrame(
        {
            "vm_id": ["vm_1", "vm_2"],
            "subscription_id": ["sub_1", "sub_2"],
            "deployment_id": ["dep_1", "dep_2"],
            "timestamp_vm_created": [100, 200],
            "timestamp_vm_deleted": [300, 400],
            "max_cpu": [50.0, 60.0],
            "avg_cpu": [40.0, 50.0],
            "p95_max_cpu": [45.0, 55.0],
            "vm_category": ["A", "B"],
            "virtual_core_bucket": [1, 2],
            "memory_gb_bucket": [2, 4],
        }
    )

    readings_df.to_csv(readings_path, index=False, compression="gzip")
    metadata_df.to_csv(metadata_path, index=False, compression="gzip")

    result_readings, result_metadata = load_data(tmp_path)

    pd.testing.assert_frame_equal(result_readings, readings_df)
    pd.testing.assert_frame_equal(result_metadata, metadata_df)


def test_load_data_raises_if_file_missing(tmp_path: Path):
    try:
        load_data(tmp_path)
    except FileNotFoundError:
        return
    raise AssertionError("Expected FileNotFoundError when required challenge files are missing")
