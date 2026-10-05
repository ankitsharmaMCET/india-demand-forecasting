"""Walk-forward (rolling-origin) backtest of all models."""
from __future__ import annotations

import pandas as pd

from .features import build_features, training_frame
from .models import ets_day_ahead, make_lightgbm, make_ridge, naive_forecast

TEST_START = "2023-05-01"
TEST_END = "2024-04-30 23:00"


def run_backtest(y: pd.Series, test_start: str = TEST_START, test_end: str = TEST_END,
                 include_ets: bool = True, verbose: bool = True) -> pd.DataFrame:
    """Forecast every day of the test window, one day ahead.

    ML models are re-trained at the start of every calendar month using only
    data strictly before that month (expanding window). Baselines need no
    training. ETS is re-fitted every day on the previous six weeks.
    Returns a frame with the actual values and one column per model.
    """
    df = build_features(y)
    test_idx = df.loc[test_start:test_end].index
    preds = pd.DataFrame(index=test_idx)
    preds["actual"] = df.loc[test_idx, "y"]
    preds["naive_24"] = naive_forecast(df, 24).loc[test_idx]
    preds["naive_168"] = naive_forecast(df, 168).loc[test_idx]

    months = pd.period_range(test_idx.min(), test_idx.max(), freq="M")
    ridge_parts, lgb_parts = [], []
    for period in months:
        month_start = period.start_time
        month_rows = df.loc[(df.index >= month_start) & (df.index <= period.end_time)]
        month_rows = month_rows.loc[month_rows.index.isin(test_idx)]
        train = training_frame(df.loc[df.index < month_start])
        month_rows = month_rows.dropna(subset=["baseline"])
        for maker, parts in ((make_ridge, ridge_parts), (make_lightgbm, lgb_parts)):
            model = maker().fit(train)
            parts.append(model.predict(month_rows))
        if verbose:
            print(f"  {period}: trained on {len(train):,} hours", flush=True)
    preds["ridge"] = pd.concat(ridge_parts)
    preds["lightgbm"] = pd.concat(lgb_parts)

    if include_ets:
        days = pd.date_range(test_idx.min().normalize(), test_idx.max().normalize(), freq="D")
        parts = []
        for i, day in enumerate(days):
            parts.append(ets_day_ahead(y, day))
            if verbose and i % 60 == 0:
                print(f"  ETS day {i + 1}/{len(days)}", flush=True)
        preds["ets"] = pd.concat(parts)
    return preds
