"""Leakage-safe feature engineering for day-ahead hourly forecasting.

Forecast origin: 00:00 of day D. Everything known at that moment is data up to
23:00 of day D-1. To forecast any hour of day D we may therefore use:

* the same hour 1, 2, 3, 7 and 14 days earlier (lags >= 24 h), and
* daily summaries of day D-1 and earlier (never of day D itself).

Short lags (1-23 h) are deliberately NOT used: for hour 05:00 of day D, the
value at 04:00 of day D is not known at midnight.
"""
from __future__ import annotations

import holidays
import numpy as np
import pandas as pd

LAGS = (24, 48, 72, 168, 336)
BASE_LAG = 168  # same hour last week: seasonal-naive baseline and ratio denominator

RATIO_FEATURES = [
    "r_lag_24", "r_lag_48", "r_lag_72", "r_lag_336",
    "r_prev_day_mean", "r_prev_day_max", "r_prev_day_min", "r_lag_168_vs_week",
]
CALENDAR_FEATURES = [
    "hour", "dayofweek", "month", "doy_sin", "doy_cos",
    "is_holiday", "is_holiday_lag_7d", "is_holiday_lag_1d",
]
FEATURES = RATIO_FEATURES + CALENDAR_FEATURES

# National lockdown began 25 Mar 2020; demand did not behave normally until later.
COVID_START, COVID_END = "2020-03-25", "2020-06-30"


def _holiday_flags(index: pd.DatetimeIndex) -> pd.Series:
    years = range(index.year.min() - 1, index.year.max() + 2)
    cal = holidays.country_holidays("IN", years=years)
    dates = pd.Series(index.normalize(), index=index)
    return dates.map(lambda d: d.date() in cal).astype(int)


def build_features(y: pd.Series) -> pd.DataFrame:
    """Build the feature table for every timestamp of ``y`` (hourly, MW).

    The returned frame holds the raw lags, the unit-free ratio features, the
    calendar features, the baseline (``lag_168``) and the target ``y``.
    Rows without a full history (first 14 days) contain NaN and are dropped by
    :func:`training_frame`.
    """
    df = pd.DataFrame({"y": y})
    for lag in LAGS:
        df[f"lag_{lag}"] = y.shift(lag)

    # Daily summaries of the *previous* day (available at 00:00 of day D).
    daily = y.resample("D")
    prev = pd.DataFrame(
        {
            "prev_day_mean": daily.mean(),
            "prev_day_max": daily.max(),
            "prev_day_min": daily.min(),
            "prev_7d_mean": daily.mean().rolling(7).mean(),
        }
    ).shift(1)
    day_key = df.index.normalize()
    for col in prev.columns:
        df[col] = prev[col].reindex(day_key).to_numpy()

    # Unit-free ratios so tree models need not extrapolate a growing demand level.
    base = df["lag_168"]
    for lag in (24, 48, 72, 336):
        df[f"r_lag_{lag}"] = df[f"lag_{lag}"] / base
    df["r_prev_day_mean"] = df["prev_day_mean"] / df["prev_7d_mean"]
    df["r_prev_day_max"] = df["prev_day_max"] / df["prev_7d_mean"]
    df["r_prev_day_min"] = df["prev_day_min"] / df["prev_7d_mean"]
    df["r_lag_168_vs_week"] = base / df["prev_7d_mean"]

    idx = df.index
    df["hour"] = idx.hour
    df["dayofweek"] = idx.dayofweek
    df["month"] = idx.month
    doy = idx.dayofyear.to_numpy()
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    hol = _holiday_flags(idx)
    df["is_holiday"] = hol
    df["is_holiday_lag_7d"] = hol.shift(168)
    df["is_holiday_lag_1d"] = hol.shift(24)
    df["baseline"] = base
    return df


def training_frame(df: pd.DataFrame, drop_covid: bool = True) -> pd.DataFrame:
    """Drop rows with incomplete history and (optionally) the 2020 lockdown."""
    out = df.dropna(subset=FEATURES + ["y", "baseline"])
    if drop_covid:
        day = out.index.normalize()
        out = out[(day < pd.Timestamp(COVID_START)) | (day > pd.Timestamp(COVID_END))]
    return out
