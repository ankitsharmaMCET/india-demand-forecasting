"""Forecast accuracy metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd


def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def mape(y_true, y_pred) -> float:
    """Mean absolute percentage error, in percent. ``y_true`` must be non-zero."""
    y_true = np.asarray(y_true, dtype=float)
    return float(np.mean(np.abs(y_true - np.asarray(y_pred)) / np.abs(y_true)) * 100)


def daily_peak_mape(y_true: pd.Series, y_pred: pd.Series) -> float:
    """MAPE of the forecast daily peak against the actual daily peak."""
    actual = y_true.groupby(y_true.index.normalize()).max()
    forecast = y_pred.groupby(y_pred.index.normalize()).max()
    return mape(actual, forecast.reindex(actual.index))


def score_table(actual: pd.Series, preds: pd.DataFrame) -> pd.DataFrame:
    """One row per model: MAE, RMSE, MAPE (all hours) and daily-peak MAPE."""
    rows = {}
    for name in preds.columns:
        p = preds[name].dropna()
        a = actual.loc[p.index]
        rows[name] = {
            "MAE_MW": mae(a, p),
            "RMSE_MW": rmse(a, p),
            "MAPE_%": mape(a, p),
            "daily_peak_MAPE_%": daily_peak_mape(a, p),
            "n_hours": len(p),
        }
    return pd.DataFrame(rows).T.sort_values("MAPE_%")


def mape_by(actual: pd.Series, preds: pd.DataFrame, key: pd.Index) -> pd.DataFrame:
    """MAPE of every model grouped by ``key`` (e.g. hour of day or month)."""
    out = {}
    for name in preds.columns:
        err = (preds[name] - actual).abs() / actual.abs() * 100
        out[name] = err.groupby(key).mean()
    return pd.DataFrame(out)


def interval_metrics(actual: pd.Series, lower: pd.Series, upper: pd.Series) -> dict:
    """Empirical coverage and average width of a prediction interval."""
    inside = (actual >= lower) & (actual <= upper)
    return {
        "coverage_%": float(inside.mean() * 100),
        "mean_width_MW": float((upper - lower).mean()),
        "mean_width_%_of_demand": float(((upper - lower) / actual).mean() * 100),
    }
