import pandas as pd
import pytest
from cwfc.validation import rolling_origin_fold_metadata, rolling_origin_splits


def test_rolling_origin_splits_are_temporal():
    frame = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=20, freq="h", tz="UTC")
    })
    splits = rolling_origin_splits(frame, horizon=4, min_train_size=8, step=4)
    assert len(splits) == 3
    assert splits[0].train_end < splits[0].validation_start
    assert splits[0].validation_end < splits[1].validation_start


def test_rolling_origin_fold_metadata_has_expected_expanding_windows():
    frame = pd.DataFrame(
        {
            "datetime": pd.date_range(
                "2026-01-01", periods=110, freq="5min", tz="UTC"
            )
        }
    )

    folds = rolling_origin_fold_metadata(
        frame,
        datetime_col="datetime",
        min_train_size=50,
        horizon=24,
        step=6,
    )

    assert len(folds) == 7
    assert folds["train_size"].tolist() == [50, 56, 62, 68, 74, 80, 86]
    assert folds["validation_size"].tolist() == [24] * 7
    assert folds["validation_start_index"].tolist() == [50, 56, 62, 68, 74, 80, 86]
    assert folds["validation_end_index_exclusive"].tolist() == [74, 80, 86, 92, 98, 104, 110]
    assert folds["train_end"].iloc[0] < folds["validation_start"].iloc[0]
    assert folds["validation_start"].iloc[1] - folds["validation_start"].iloc[0] == pd.Timedelta(minutes=30)


def test_rolling_origin_fold_metadata_rejects_unsorted_datetime():
    frame = pd.DataFrame(
        {
            "datetime": pd.to_datetime(
                ["2026-01-01T01:00:00Z", "2026-01-01T00:00:00Z"]
            )
        }
    )

    with pytest.raises(ValueError, match="timestamps must be sorted"):
        rolling_origin_fold_metadata(
            frame, min_train_size=1, horizon=1, step=1
        )
