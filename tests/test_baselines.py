import pandas as pd
import pytest

from cwfc.baselines import (
    DEFAULT_SEASON_LENGTH,
    last_6_mean_forecast,
    last_12_mean_forecast,
    last_n_mean_forecast,
    persistence_forecast,
    seasonal_naive_forecast,
    training_mean_forecast,
)


@pytest.fixture
def train_and_template():
    train = pd.DataFrame({"target": list(range(1, 25))})
    template = pd.DataFrame(index=[30, 31, 32, 33])
    return train, template


def test_persistence_forecast_repeats_last_observation(train_and_template):
    train, template = train_and_template
    forecast = persistence_forecast(train, template, "target")

    assert forecast.tolist() == [24.0] * 4
    assert forecast.index.equals(template.index)


def test_last_n_mean_forecasts_use_only_recent_values(train_and_template):
    train, template = train_and_template

    assert last_6_mean_forecast(train, template, "target").tolist() == [21.5] * 4
    assert last_12_mean_forecast(train, template, "target").tolist() == [18.5] * 4
    assert last_n_mean_forecast(train, template, "target", window=3).tolist() == [23.0] * 4


def test_training_mean_forecast_uses_all_training_values(train_and_template):
    train, template = train_and_template
    assert training_mean_forecast(train, template, "target").tolist() == [12.5] * 4


def test_seasonal_naive_repeats_latest_cycle(train_and_template):
    train, template = train_and_template
    forecast = seasonal_naive_forecast(
        train, template, "target", season_length=3
    )

    assert forecast.tolist() == [22.0, 23.0, 24.0, 22.0]
    assert DEFAULT_SEASON_LENGTH == 12
    assert seasonal_naive_forecast(train, template, "target").tolist() == [13.0, 14.0, 15.0, 16.0]


@pytest.mark.parametrize(
    ("forecast_fn", "kwargs", "history_size"),
    [
        (last_6_mean_forecast, {}, 5),
        (last_12_mean_forecast, {}, 11),
        (seasonal_naive_forecast, {"season_length": 12}, 11),
    ],
)
def test_windowed_baselines_require_sufficient_history(
    forecast_fn, kwargs, history_size
):
    train = pd.DataFrame({"target": range(history_size)})
    template = pd.DataFrame(index=[0])

    with pytest.raises(ValueError, match="not enough training observations"):
        forecast_fn(train, template, "target", **kwargs)
