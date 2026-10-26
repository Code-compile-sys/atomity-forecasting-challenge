import numpy as np
import pytest

from cwfc.metrics import mae, normalized_mae, rmse, smape


def test_metrics_basic():
    y = [1, 2, 3]
    p = [1, 3, 1]
    assert mae(y, p) == 1.0
    assert round(rmse(y, p), 8) == round((5 / 3) ** 0.5, 8)
    assert smape(y, p) > 0
    assert normalized_mae(y, p, [1, 1, 1]) > 0


def test_metrics_ignore_missing_paired_observations():
    y = [1.0, np.nan, 3.0]
    prediction = [2.0, 100.0, 5.0]
    baseline = [0.0, 0.0, 0.0]

    assert mae(y, prediction) == 1.5
    assert rmse(y, prediction) == pytest.approx(np.sqrt(2.5))
    assert normalized_mae(y, prediction, baseline) == pytest.approx(0.75)
    assert smape(y, prediction) == pytest.approx((2 / 3 + 2 / 4) * 50)


def test_smape_protects_near_zero_values_with_epsilon():
    assert smape([0.0, 1e-12], [0.0, 2e-12], eps=1e-8) == pytest.approx(0.01)
    with pytest.raises(ValueError, match="eps must be positive"):
        smape([1.0], [1.0], eps=0)


def test_metrics_reject_empty_valid_pairs_and_infinities():
    with pytest.raises(ValueError, match="no valid paired observations"):
        mae([np.nan], [1.0])
    with pytest.raises(ValueError, match="infinite"):
        rmse([1.0], [np.inf])
