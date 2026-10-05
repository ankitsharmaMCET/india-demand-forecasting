import pandas as pd
import pytest

from demand_forecast.data import load_hourly


def _write(tmp_path, index, values):
    p = tmp_path / "d.csv"
    pd.DataFrame({"datetime": index, "demand_mw": values}).to_csv(p, index=False)
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


def _book(path, years, bump=0.0):
    """Write a fake ICED download: two days of January per year, plus a footer row."""
    rows = []
    for yr in years:
        for ts in pd.date_range(f"{yr}-01-01", f"{yr}-01-02 23:00", freq="h"):
            hour12 = ts.hour % 12 or 12
            label = f"{ts.day:02d}-{ts:%b} {hour12}{'am' if ts.hour < 12 else 'pm'}"
            rows.append((yr, label, 100.0 + (yr - 2000) + ts.hour + bump))
    df = pd.DataFrame(rows, columns=["Year", "Date", "Hourly Demand Met (in MW)"])
    footer = pd.DataFrame([["footer text", None, None]], columns=df.columns)
    pd.concat([df, footer]).to_excel(path, index=False)


def test_prepare_iced_merges_overlapping_years(tmp_path):
    from demand_forecast.data import prepare_iced

    _book(tmp_path / "Yearly Demand Profile.xlsx", [2020, 2021])
    _book(tmp_path / "Yearly Demand Profile (1).xlsx", [2021, 2022])
    out = prepare_iced(tmp_path, tmp_path / "merged.csv")
    merged = pd.read_csv(out, parse_dates=["datetime"])
    assert len(merged) == 3 * 48  # 2021 appears in both files but is kept once
    assert merged["datetime"].is_monotonic_increasing and merged["datetime"].is_unique
    first = merged.iloc[0]
    assert first["datetime"] == pd.Timestamp("2020-01-01 00:00") and first["demand_mw"] == 120.0
    assert merged.loc[merged["datetime"] == "2020-01-01 13:00", "demand_mw"].item() == 133.0


def test_prepare_iced_rejects_conflicting_downloads(tmp_path):
    from demand_forecast.data import prepare_iced

    _book(tmp_path / "Yearly Demand Profile.xlsx", [2020, 2021])
    _book(tmp_path / "Yearly Demand Profile (1).xlsx", [2021, 2022], bump=5.0)
    with pytest.raises(ValueError, match="different values"):
        prepare_iced(tmp_path, tmp_path / "merged.csv")
