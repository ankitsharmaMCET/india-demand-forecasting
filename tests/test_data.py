import pandas as pd
import pytest

from demand_forecast.data import load_hourly


def _write(tmp_path, index, values):
    p = tmp_path / "d.csv"
    pd.DataFrame({"datetime": index, "National Hourly Demand": values}).to_csv(p, index=False)
    return p


def test_load_ok(tmp_path):
    idx = pd.date_range("2024-01-01", periods=48, freq="h")
    s = load_hourly(_write(tmp_path, idx, range(48)))
    assert len(s) == 48 and s.index.freq is not None


def test_gap_is_rejected(tmp_path):
    idx = pd.date_range("2024-01-01", periods=48, freq="h").delete(10)
    with pytest.raises(ValueError, match="missing hourly"):
        load_hourly(_write(tmp_path, idx, range(47)))


def test_duplicates_rejected(tmp_path):
    idx = pd.date_range("2024-01-01", periods=5, freq="h").append(pd.DatetimeIndex(["2024-01-01 02:00"]))
    with pytest.raises(ValueError, match="duplicate"):
        load_hourly(_write(tmp_path, idx, range(6)))
