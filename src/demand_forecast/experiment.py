"""Walk-forward (rolling-origin) backtest of all models."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features import FEATURES, FEATURES_LAST, build_features, training_frame
from .models import ets_day_ahead, make_lightgbm, make_ridge, naive_forecast

# Test folds. The hyper-parameters are tuned on VALIDATION (a year *before* both folds).
VALIDATION = ("2021-05-01", "2022-04-30 23:00")
FOLDS = {
    "2022-23": ("2022-05-01", "2023-04-30 23:00"),
    "2023-24": ("2023-05-01", "2024-04-30 23:00"),
}
TEST_START, TEST_END = FOLDS["2023-24"]
INTERVAL = (0.1, 0.9)  # 80 % prediction interval

MODEL_COLUMNS = ["naive_24", "naive_168", "ets", "ridge", "lightgbm_basic", "lightgbm"]


def run_backtest(y: pd.Series, test_start: str = TEST_START, test_end: str = TEST_END,
                 include_ets: bool = True, verbose: bool = True,
                 lgb_params: dict | None = None, intervals: bool = True) -> pd.DataFrame:
    """Forecast every day of the test window, one day ahead.

    ML models are re-trained at the start of every calendar month using only
    data strictly before that month (expanding window). Baselines need no
    training. ETS is re-fitted every day on the previous six weeks.

    Columns: ``actual`` plus one per model; ``lightgbm_q10`` / ``lightgbm_q90``
    bound the 80 % prediction interval of the main LightGBM model.
    ``lightgbm_basic`` is the same model without the last-observed-hours features.
    """
    df = build_features(y)
    test_idx = df.loc[test_start:test_end].index
    preds = pd.DataFrame(index=test_idx)
    preds["actual"] = df.loc[test_idx, "y"]
    preds["naive_24"] = naive_forecast(df, 24).loc[test_idx]
    preds["naive_168"] = naive_forecast(df, 168).loc[test_idx]

    makers = {
        "ridge": lambda: make_ridge(),
        "lightgbm_basic": lambda: make_lightgbm(FEATURES, lgb_params, name="lightgbm_basic"),
        "lightgbm": lambda: make_lightgbm(FEATURES_LAST, lgb_params),
    }
    if intervals:
        makers["lightgbm_q10"] = lambda: make_lightgbm(FEATURES_LAST, lgb_params, alpha=INTERVAL[0])
        makers["lightgbm_q90"] = lambda: make_lightgbm(FEATURES_LAST, lgb_params, alpha=INTERVAL[1])
    parts = {k: [] for k in makers}

    for period in pd.period_range(test_idx.min(), test_idx.max(), freq="M"):
        month_start = period.start_time
        rows = df.loc[(df.index >= month_start) & (df.index <= period.end_time)]
        rows = rows.loc[rows.index.isin(test_idx)].dropna(subset=["baseline"])
        train = training_frame(df.loc[df.index < month_start])
        for key, maker in makers.items():
            parts[key].append(maker().fit(train).predict(rows))
        if verbose:
            print(f"  {period}: trained on {len(train):,} hours", flush=True)
    for key, chunks in parts.items():
        preds[key] = pd.concat(chunks)
    if intervals:
        lo = np.minimum(preds["lightgbm_q10"], preds["lightgbm_q90"])
        hi = np.maximum(preds["lightgbm_q10"], preds["lightgbm_q90"])
        preds["lightgbm_q10"], preds["lightgbm_q90"] = lo, hi

    if include_ets:
        days = pd.date_range(test_idx.min().normalize(), test_idx.max().normalize(), freq="D")
        chunks = []
        for i, day in enumerate(days):
            chunks.append(ets_day_ahead(y, day))
            if verbose and i % 90 == 0:
                print(f"  ETS day {i + 1}/{len(days)}", flush=True)
        preds["ets"] = pd.concat(chunks)
    return preds
