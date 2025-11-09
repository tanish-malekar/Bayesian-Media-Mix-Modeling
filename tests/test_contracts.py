import numpy as np
import pandas as pd
import pytest
from mmm.allocate import project_shares
from mmm.data import CHANNELS, CONTROLS, Preprocessor
from mmm.evaluate import metrics
from mmm.transforms import adstock, hill


def test_adstock_preserves_history_across_a_holdout_boundary():
    values = np.array([[10.], [0.], [4.], [0.]])
    whole = adstock(values, [.5])
    held = adstock(values[2:], [.5], initial=whole[1])
    np.testing.assert_allclose(whole[2:], held)
    np.testing.assert_allclose(whole[:, 0], [10, 5, 6.5, 3.25])


def test_hill_has_zero_response_half_saturation_and_diminishing_response():
    x = np.array([0., 1., 2., 4.])
    response = hill(x, 1, 1)
    assert response[0] == 0 and response[1] == .5
    assert np.all(np.diff(response) > 0) and response[-1] < 1


def test_fixed_budget_projection_respects_bounds():
    result = project_shares([1., -.1, .2], [.1, .2, .1], [.5, .6, .4])
    assert result.sum() == pytest.approx(1)
    assert np.all(result >= [.1, .2, .1]) and np.all(result <= [.5, .6, .4])
    with pytest.raises(ValueError):
        project_shares([1, 1], [.7, .7], [1, 1])


def test_holdout_revenue_is_not_a_feature_or_scaler():
    frame = pd.DataFrame({"DATE": pd.date_range("2020-01-06", periods=8, freq="7D"),
                          "revenue": np.arange(8) + 10, "events": ["none"] * 8,
                          **{name: np.arange(8) + 1 for name in CHANNELS + CONTROLS}})
    processor = Preprocessor.fit(frame.iloc[:6])
    changed = frame.copy(); changed.loc[6:, "revenue"] = 1e12
    original_x = processor.transform(frame)
    changed_x = processor.transform(changed)
    np.testing.assert_array_equal(original_x[0], changed_x[0])
    np.testing.assert_array_equal(original_x[1], changed_x[1])
    assert processor.revenue_scale == frame.iloc[:6].revenue.mean()


def test_mape_is_reported_in_percent_and_rejects_zero_actuals():
    assert metrics([100, 200], [90, 220])["mape_percent"] == pytest.approx(10)
    with pytest.raises(ValueError):
        metrics([0], [1])
