"""Forecasters. Every model predicts one full day (24 hours) ahead."""
from __future__ import annotations

import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from .features import BASE_LAG, FEATURES, FEATURES_LAST, RATIO_FEATURES


# ---------------------------------------------------------------- baselines
def naive_forecast(df: pd.DataFrame, lag: int) -> pd.Series:
    """Same hour ``lag`` hours earlier (24 = yesterday, 168 = last week)."""
    return df[f"lag_{lag}"].rename(f"naive_{lag}")


# ---------------------------------------------------- ratio-target ML models
class RatioModel:
    """Learns ``y / lag_168`` and multiplies back, so the growing demand level
    is handled by the baseline and the model only learns the *relative* change.
    """

    name = "ratio_model"

    def __init__(self, estimator, features=None):
        self.estimator = estimator
        self.features = list(features or FEATURES)

    def fit(self, frame: pd.DataFrame) -> "RatioModel":
        self.estimator.fit(frame[self.features], frame["y"] / frame["baseline"])
        return self

    def predict(self, frame: pd.DataFrame) -> pd.Series:
        ratio = self.estimator.predict(frame[self.features])
        return pd.Series(ratio * frame["baseline"].to_numpy(), index=frame.index)


def make_ridge(alpha: float = 1.0) -> RatioModel:
    pre = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["hour", "dayofweek", "month"]),
            ("num", StandardScaler(),
             RATIO_FEATURES + ["doy_sin", "doy_cos", "is_holiday",
                               "is_holiday_lag_7d", "is_holiday_lag_1d"]),
        ]
    )
    m = RatioModel(Pipeline([("pre", pre), ("reg", Ridge(alpha=alpha))]))
    m.name = "ridge"
    return m


DEFAULT_LGB_PARAMS = dict(
    n_estimators=500, learning_rate=0.03, num_leaves=31, min_child_samples=50,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
)


def make_lightgbm(features=None, params: dict | None = None, seed: int = 0,
                  alpha: float | None = None, name: str = "lightgbm") -> RatioModel:
    """LightGBM on the ratio target. ``alpha`` switches to quantile regression."""
    p = {**DEFAULT_LGB_PARAMS, **(params or {})}
    if alpha is not None:
        p.update(objective="quantile", alpha=alpha)
    reg = lgb.LGBMRegressor(random_state=seed, n_jobs=2, verbose=-1, **p)
    m = RatioModel(reg, features or FEATURES_LAST)
    m.name = name
    return m


# --------------------------------------------------------- statistical model
def ets_day_ahead(y: pd.Series, origin: pd.Timestamp, window_days: int = 42,
                  season: int = BASE_LAG) -> pd.Series:
    """Holt-Winters (additive damped trend, additive weekly-hourly season).

    Fitted on the ``window_days`` before ``origin`` (00:00 of the forecast day)
    and used to forecast the next 24 hours. ``season=168`` captures both the
    daily and the weekly cycle in a single seasonal pattern.
    """
    hist = y.loc[origin - pd.Timedelta(days=window_days): origin - pd.Timedelta(hours=1)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = ExponentialSmoothing(
            hist.to_numpy(), trend="add", damped_trend=True,
            seasonal="add", seasonal_periods=season,
            initialization_method="heuristic",
        ).fit()
    fc = fit.forecast(24)
    return pd.Series(np.asarray(fc), index=pd.date_range(origin, periods=24, freq="h"))
