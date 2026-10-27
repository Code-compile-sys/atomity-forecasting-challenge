import numpy as np
import pandas as pd
import pytest

from cwfc.models import damped_exponential_smoothing_forecast, ridge_forecast


def test_ridge_forecast_is_recursive_and_uses_template_index_only():
    train = pd.DataFrame(
        {"target": 10 + np.sin(np.arange(60) / 5)}
    )
    template = pd.DataFrame(
        {"target": [1000.0] * 24},
        index=pd.RangeIndex(start=60, stop=84),
    )
    other_template = template.assign(target=-1000.0)

    prediction = ridge_forecast(train, template, "target")
    other_prediction = ridge_forecast(train, other_template, "target")

    assert prediction.shape == (24,)
    assert prediction.index.equals(template.index)
    assert np.isfinite(prediction).all()
    np.testing.assert_allclose(prediction, other_prediction)


def test_ridge_forecast_rejects_insufficient_or_invalid_training_data():
    template = pd.DataFrame(index=range(3))

    with pytest.raises(ValueError, match="requires more than 12"):
        ridge_forecast(pd.DataFrame({"target": range(12)}), template, "target")

    train = pd.DataFrame({"target": [*range(20), np.nan]})
    with pytest.raises(ValueError, match="finite"):
        ridge_forecast(train, template, "target")


def test_ridge_forecast_rejects_negative_regularization():
    train = pd.DataFrame({"target": range(20)})
    template = pd.DataFrame(index=range(2))

    with pytest.raises(ValueError, match="alpha must be non-negative"):
        ridge_forecast(train, template, "target", alpha=-1)


def test_damped_exponential_smoothing_forecasts_full_horizon_without_template_targets():
    train = pd.DataFrame({"target": 20 + np.arange(60) * 0.1})
    template = pd.DataFrame(
        {"target": [10000.0] * 24},
        index=pd.RangeIndex(start=60, stop=84),
    )
    other_template = template.assign(target=-10000.0)

    prediction = damped_exponential_smoothing_forecast(train, template, "target")
    other_prediction = damped_exponential_smoothing_forecast(
        train, other_template, "target"
    )

    assert len(prediction) == 24
    assert prediction.index.equals(template.index)
    assert np.isfinite(prediction).all()
    np.testing.assert_allclose(prediction, other_prediction)


def test_damped_exponential_smoothing_preserves_constant_series_exactly():
    train = pd.DataFrame({"target": [10_000.0] * 50})
    template = pd.DataFrame({"target": [np.nan] * 24})

    prediction = damped_exponential_smoothing_forecast(train, template, "target")

    np.testing.assert_array_equal(prediction.to_numpy(), np.full(24, 10_000.0))


def test_damped_exponential_smoothing_validates_training_data():
    template = pd.DataFrame(index=range(3))

    with pytest.raises(ValueError, match="at least 3"):
        damped_exponential_smoothing_forecast(
            pd.DataFrame({"target": [1.0, 2.0]}), template, "target"
        )

    with pytest.raises(ValueError, match="finite"):
        damped_exponential_smoothing_forecast(
            pd.DataFrame({"target": [1.0, np.nan, 3.0]}), template, "target"
        )
