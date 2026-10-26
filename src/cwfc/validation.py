from __future__ import annotations

from dataclasses import dataclass
import pandas as pd


@dataclass(frozen=True)
class TimeSplit:
    train_end: pd.Timestamp
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp


def rolling_origin_splits(
    frame: pd.DataFrame,
    timestamp_col: str = "timestamp",
    horizon: int = 288,
    min_train_size: int = 7 * 24 * 12,
    step: int | None = None,
) -> list[TimeSplit]:
    """Return expanding-window splits for evenly sampled time series.

    `horizon=288` is one day at 5-minute cadence.
    """
    if horizon <= 0 or min_train_size <= 0:
        raise ValueError("horizon and min_train_size must be positive")
    ts = pd.to_datetime(frame[timestamp_col], utc=True)
    if not ts.is_monotonic_increasing:
        raise ValueError("timestamps must be sorted")
    n = len(frame)
    step = horizon if step is None else step
    out: list[TimeSplit] = []
    train_end_idx = min_train_size
    while train_end_idx + horizon <= n:
        train_end = ts.iloc[train_end_idx - 1]
        val_start = ts.iloc[train_end_idx]
        val_end = ts.iloc[train_end_idx + horizon - 1]
        out.append(TimeSplit(train_end, val_start, val_end))
        train_end_idx += step
    return out


def rolling_origin_fold_metadata(
    frame: pd.DataFrame,
    datetime_col: str = "datetime",
    horizon: int = 24,
    min_train_size: int = 50,
    step: int = 6,
) -> pd.DataFrame:
    """Describe expanding-window folds by row bounds and timestamp boundaries.

    Row end indices are exclusive, matching normal Python slicing. The input
    must already be sorted by the datetime column.
    """
    if datetime_col not in frame.columns:
        raise ValueError(f"frame is missing required column: {datetime_col}")
    if horizon <= 0 or min_train_size <= 0 or step <= 0:
        raise ValueError("horizon, min_train_size, and step must be positive")

    splits = rolling_origin_splits(
        frame,
        timestamp_col=datetime_col,
        horizon=horizon,
        min_train_size=min_train_size,
        step=step,
    )
    rows: list[dict[str, object]] = []
    for fold_number, split in enumerate(splits):
        train_start_idx = 0
        train_end_idx = min_train_size + fold_number * step
        validation_start_idx = train_end_idx
        validation_end_idx = validation_start_idx + horizon
        rows.append(
            {
                "fold": fold_number + 1,
                "train_start_index": train_start_idx,
                "train_end_index_exclusive": train_end_idx,
                "validation_start_index": validation_start_idx,
                "validation_end_index_exclusive": validation_end_idx,
                "train_size": train_end_idx - train_start_idx,
                "validation_size": validation_end_idx - validation_start_idx,
                "train_start": pd.to_datetime(frame[datetime_col].iloc[train_start_idx], utc=True),
                "train_end": split.train_end,
                "validation_start": split.validation_start,
                "validation_end": split.validation_end,
            }
        )
    return pd.DataFrame(rows)
