"""Data loading utilities for the Azure VM workload forecasting challenge."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_data(path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the challenge CPU and VM metadata tables from a directory.

    Parameters
    ----------
    path:
        Directory containing ``vm_cpu_readings.csv.gz`` and
        ``vm_metadata.csv.gz``.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame]
        The CPU readings DataFrame followed by the metadata DataFrame.
    """
    base_path = Path(path)

    readings_path = base_path / "vm_cpu_readings.csv.gz"
    metadata_path = base_path / "vm_metadata.csv.gz"

    if not readings_path.exists():
        raise FileNotFoundError(f"CPU readings file not found: {readings_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    readings_df = pd.read_csv(readings_path)
    metadata_df = pd.read_csv(metadata_path)

    return readings_df, metadata_df
