"""Hyper-parameter search for LightGBM on a validation year.

The search only ever sees data from the validation window and earlier, so the
later test folds stay untouched. Each candidate is scored by walk-forward MAE:
it is retrained at the start of every quarter of the validation year.
"""
from __future__ import annotations

import itertools

import pandas as pd

from .evaluate import mae, mape
from .features import FEATURES_LAST, build_features, training_frame
from .models import make_lightgbm

GRID = {
    "num_leaves": [15, 31, 63],
    "rate_trees": [(0.03, 500), (0.06, 250)],  # learning_rate, n_estimators
    "min_child_samples": [50, 200],
}


def candidates(grid: dict = GRID) -> list[dict]:
    keys = list(grid)
    out = []
    for values in itertools.product(*(grid[k] for k in keys)):
        c = dict(zip(keys, values))
        lr, n = c.pop("rate_trees")
        out.append({**c, "learning_rate": lr, "n_estimators": n})
    return out


def tune_lightgbm(y: pd.Series, val_start: str, val_end: str, grid: dict = GRID,
                  verbose: bool = True) -> pd.DataFrame:
    """Return one row per candidate with its validation MAE and MAPE, best first."""
    df = build_features(y)
    val_idx = df.loc[val_start:val_end].index
    quarters = pd.period_range(val_idx.min(), val_idx.max(), freq="Q")
    rows = []
    for params in candidates(grid):
        preds = []
        for q in quarters:
            train = training_frame(df.loc[df.index < q.start_time])
            test = df.loc[(df.index >= q.start_time) & (df.index <= q.end_time)]
            test = test.loc[test.index.isin(val_idx)].dropna(subset=FEATURES_LAST + ["baseline"])
            preds.append(make_lightgbm(FEATURES_LAST, params).fit(train).predict(test))
        p = pd.concat(preds)
        a = df.loc[p.index, "y"]
        rows.append({**params, "val_MAE_MW": mae(a, p), "val_MAPE_%": mape(a, p)})
        if verbose:
            print(f"  {params} -> MAE {rows[-1]['val_MAE_MW']:.0f}", flush=True)
    return pd.DataFrame(rows).sort_values("val_MAE_MW").reset_index(drop=True)
