import numpy as np
import pandas as pd

from demand_forecast.features import FEATURES, build_features, training_frame
from demand_forecast.models import naive_forecast

CHECKED = FEATURES + ["lag_24", "lag_48", "lag_72", "lag_168", "lag_336", "baseline"]


def test_lags_are_exact_shifts(synthetic):
    df = build_features(synthetic)
    t = synthetic.index[500]
    assert df.loc[t, "lag_24"] == synthetic.loc[t - pd.Timedelta(hours=24)]
    assert df.loc[t, "lag_168"] == synthetic.loc[t - pd.Timedelta(hours=168)]
    assert naive_forecast(df, 168).loc[t] == df.loc[t, "baseline"]


def test_no_short_lags_in_features():
    # lags below 24 h are not known at the 00:00 forecast origin
    assert not any(f.startswith("lag_") and int(f.split("_")[1]) < 24 for f in FEATURES)


def test_no_leakage_from_forecast_day(synthetic):
    """Changing every value from day D onward must not change day D's features."""
    day = pd.Timestamp("2021-03-10")
    corrupted = synthetic.copy()
    corrupted.loc[day:] = corrupted.loc[day:] * 10
    a = build_features(synthetic).loc[day: day + pd.Timedelta(hours=23), CHECKED]
    b = build_features(corrupted).loc[day: day + pd.Timedelta(hours=23), CHECKED]
    pd.testing.assert_frame_equal(a, b)


def test_features_do_use_the_past(synthetic):
    """Sanity check for the test above: changing yesterday must change features."""
    day = pd.Timestamp("2021-03-10")
    changed = synthetic.copy()
    changed.loc[day - pd.Timedelta(days=1): day - pd.Timedelta(hours=1)] *= 2
    a = build_features(synthetic).loc[day, "prev_day_mean"]
    b = build_features(changed).loc[day, "prev_day_mean"]
    assert b > a * 1.5


def test_training_frame_has_no_nans(synthetic):
    tf = training_frame(build_features(synthetic))
    assert len(tf) > 0
    assert not tf[FEATURES + ["y", "baseline"]].isna().any().any()
    assert tf.index.min() >= synthetic.index.min() + pd.Timedelta(days=14)


def test_covid_window_is_dropped():
    idx = pd.date_range("2020-01-01", periods=200 * 24, freq="h")
    y = pd.Series(np.linspace(100_000, 120_000, len(idx)), index=idx)
    tf = training_frame(build_features(y))
    assert not ((tf.index >= "2020-03-25") & (tf.index <= "2020-06-30 23:00")).any()
    assert len(training_frame(build_features(y), drop_covid=False)) > len(tf)
