import numpy as np
import pandas as pd
import pytest

from demand_forecast.evaluate import daily_peak_mape, interval_metrics, mae, mape, rmse, score_table


def test_basic_metrics():
    a, p = np.array([100.0, 200.0]), np.array([110.0, 180.0])
    assert mae(a, p) == 15.0
    assert rmse(a, p) == pytest.approx(np.sqrt(250.0))
    assert mape(a, p) == pytest.approx(10.0)  # (10% + 10%) / 2


def test_daily_peak_mape():
    idx = pd.date_range("2024-01-01", periods=48, freq="h")
    actual = pd.Series(100.0, index=idx)
    actual.iloc[10] = 150.0   # day-1 peak
    actual.iloc[30] = 200.0   # day-2 peak
    forecast = actual * 1.1
    assert daily_peak_mape(actual, forecast) == pytest.approx(10.0)


def test_score_table_sorted_by_mape():
    idx = pd.date_range("2024-01-01", periods=24, freq="h")
    actual = pd.Series(100.0, index=idx)
    preds = pd.DataFrame({"bad": actual * 1.2, "good": actual * 1.01})
    table = score_table(actual, preds)
    assert list(table.index) == ["good", "bad"]
    assert table.loc["good", "MAPE_%"] == pytest.approx(1.0)


def test_interval_metrics():
    idx = pd.date_range("2024-01-01", periods=4, freq="h")
    actual = pd.Series([100.0, 100.0, 100.0, 100.0], index=idx)
    lower = pd.Series([90.0, 90.0, 101.0, 90.0], index=idx)   # third value falls outside
    upper = pd.Series([110.0, 110.0, 110.0, 110.0], index=idx)
    m = interval_metrics(actual, lower, upper)
    assert m["coverage_%"] == 75.0
    assert m["mean_width_MW"] == pytest.approx((20 + 20 + 9 + 20) / 4)
