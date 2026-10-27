import numpy as np
import pandas as pd

from cwfc.analysis import analyze_selected_model_cv


def test_selected_model_error_analysis_uses_training_cv_and_reports_requested_summaries():
    values = 10 + np.sin(np.arange(110) / 8) + np.arange(110) * 0.002
    train_frame = pd.DataFrame(
        {
            "timestamp": np.arange(110) * 300,
            "datetime": pd.date_range(
                "2026-01-01", periods=110, freq="5min", tz="UTC"
            ),
            "aggregate_cpu_utilization": values,
        }
    )

    analysis = analyze_selected_model_cv(train_frame)
    forecasts = analysis.cv_forecasts

    assert analysis.selected_model in {"Last12Mean", "Ridge", "DampedETS"}
    assert len(forecasts) == 7 * 24
    assert forecasts["fold"].nunique() == 7
    assert forecasts["horizon_step"].min() == 1
    assert forecasts["horizon_step"].max() == 24
    assert forecasts["residual"].equals(forecasts["actual"] - forecasts["forecast"])
    assert (analysis.peak_threshold == np.quantile(values, 0.9))
    assert forecasts["is_peak"].equals(forecasts["actual"] > analysis.peak_threshold)
    assert set(analysis.peak_summary["period"]) == {"peak", "non_peak"}
    assert {"mae", "rmse", "bias"}.issubset(analysis.peak_summary.columns)
    assert len(analysis.fold_mae) == 7
    assert analysis.fold_mae["mae"].is_monotonic_decreasing
    assert len(analysis.horizon_mae) == 24
    assert analysis.horizon_mae["horizon_step"].tolist() == list(range(1, 25))
