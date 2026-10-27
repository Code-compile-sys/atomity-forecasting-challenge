import numpy as np
import pandas as pd
import pytest

from cwfc.baselines import persistence_forecast
from cwfc.evaluate import evaluate_forecasters, evaluate_rolling_origin


def test_evaluate_rolling_origin_orders_by_datetime_and_reports_metrics():
    frame = pd.DataFrame(
        {
            "datetime": pd.date_range("2026-01-01", periods=30, freq="5min", tz="UTC"),
            "target": np.arange(30, dtype=float),
        }
    ).sample(frac=1, random_state=5)

    results = evaluate_rolling_origin(
        frame,
        target="target",
        forecast_fn=persistence_forecast,
        horizon=6,
        min_train_size=12,
        step=6,
    )

    assert results["fold"].tolist() == [1, 2, 3, "mean"]
    assert results.loc[0, "train_size"] == 12
    assert results.loc[0, "validation_size"] == 6
    assert results.loc[0, "train_end"] < results.loc[0, "validation_start"]
    assert results.loc[0, "mae"] == pytest.approx(3.5)
    assert results.loc[0, "rmse"] == pytest.approx(np.sqrt(91 / 6))
    assert results.loc[3, "mae"] == pytest.approx(results.loc[:2, "mae"].mean())
    assert {"mae", "rmse"}.issubset(results.columns)
    assert not {"normalized_mae", "smape"}.intersection(results.columns)


def test_evaluate_rolling_origin_rejects_duplicate_timestamps():
    frame = pd.DataFrame(
        {
            "datetime": pd.date_range("2026-01-01", periods=10, freq="h", tz="UTC"),
            "target": np.arange(10, dtype=float),
        }
    )
    frame.loc[1, "datetime"] = frame.loc[0, "datetime"]

    with pytest.raises(ValueError, match="unique timestamps"):
        evaluate_rolling_origin(
            frame,
            target="target",
            forecast_fn=persistence_forecast,
            horizon=2,
            min_train_size=4,
        )


def test_evaluate_rolling_origin_requires_enough_rows_for_a_fold():
    frame = pd.DataFrame(
        {
            "datetime": pd.date_range("2026-01-01", periods=5, freq="h", tz="UTC"),
            "target": np.arange(5, dtype=float),
        }
    )

    with pytest.raises(ValueError, match="not enough observations"):
        evaluate_rolling_origin(
            frame,
            target="target",
            forecast_fn=persistence_forecast,
            horizon=3,
            min_train_size=4,
        )


def test_evaluator_reports_mae_and_rmse_with_missing_validation_targets():
    frame = pd.DataFrame(
        {
            "datetime": pd.date_range("2026-01-01", periods=7, freq="h", tz="UTC"),
            "target": [1.0, 2.0, 3.0, 4.0, 5.0, np.nan, 7.0],
        }
    )

    def zero_forecast(train, template, target):
        return pd.Series(0.0, index=template.index)

    result = evaluate_rolling_origin(
        frame,
        target="target",
        forecast_fn=zero_forecast,
        horizon=3,
        min_train_size=4,
    )

    assert result.loc[0, "scored_observations"] == 2
    assert result.loc[0, "mae"] == pytest.approx(6.0)
    assert result.loc[0, "rmse"] == pytest.approx(np.sqrt(37.0))


def test_evaluate_forecasters_uses_three_models_and_same_folds():
    frame = pd.DataFrame(
        {
            "datetime": pd.date_range("2026-01-01", periods=110, freq="5min", tz="UTC"),
            "arbitrary_target": np.sin(np.arange(110) / 8),
        }
    )

    results = evaluate_forecasters(frame, target="arbitrary_target")

    assert len(results) == 24
    assert results["model"].nunique() == 3
    assert set(results["model"]) == {
        "Last12Mean",
        "Ridge",
        "DampedETS",
    }
    assert set(results["fold"]) == {1, 2, 3, 4, 5, 6, 7, "mean"}
    assert {"mae", "rmse"}.issubset(results.columns)
    assert not {"normalized_mae", "smape"}.intersection(results.columns)
    folds_by_model = results.groupby("model")["fold"].apply(tuple)
    assert folds_by_model.nunique() == 1
    assert (results.loc[results["fold"] != "mean", "validation_size"] == 24).all()
    for model, model_results in results.groupby("model"):
        fold_results = model_results.loc[model_results["fold"] != "mean"]
        mean_result = model_results.loc[model_results["fold"] == "mean"].iloc[0]
        assert mean_result["mae"] == pytest.approx(fold_results["mae"].mean())
        assert mean_result["rmse"] == pytest.approx(fold_results["rmse"].mean())


def test_evaluate_forecasters_returns_zero_error_for_constant_target():
    frame = pd.DataFrame(
        {
            "datetime": pd.date_range("2026-01-01", periods=86, freq="5min", tz="UTC"),
            "active_vm_count": np.full(86, 10_000.0),
        }
    )

    results = evaluate_forecasters(frame, target="active_vm_count")

    assert len(results) == 12
    assert results["model"].nunique() == 3
    assert results[["mae", "rmse"]].to_numpy().tolist() == [[0.0, 0.0]] * 12
