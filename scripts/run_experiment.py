"""Run the backtest on all test folds and write metrics + predictions to results/."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from demand_forecast.data import load_hourly  # noqa: E402
from demand_forecast.evaluate import interval_metrics, mape_by, score_table  # noqa: E402
from demand_forecast.experiment import FOLDS, HEADLINE, MODEL_COLUMNS, run_backtest  # noqa: E402


def main() -> None:
    results = ROOT / "results"
    results.mkdir(exist_ok=True)
    y = load_hourly(ROOT / "data/raw/iced_hourly.csv")
    print(f"Loaded {len(y):,} hourly values, {y.index.min()} -> {y.index.max()}")

    params_file = results / "best_params.json"
    params = json.loads(params_file.read_text()) if params_file.exists() else None
    print("LightGBM parameters:", params or "defaults (run scripts/tune.py to tune)")

    all_preds, all_metrics, all_intervals = [], [], []
    for label, (start, end) in FOLDS.items():
        print(f"\nFold {label}: {start} -> {end}")
        t0 = time.time()
        preds = run_backtest(y, start, end, lgb_params=params)
        print(f"  finished in {time.time() - t0:.0f}s")
        actual = preds["actual"]
        table = score_table(actual, preds[MODEL_COLUMNS])
        table.insert(0, "fold", label)
        all_metrics.append(table.reset_index(names="model"))
        iv = interval_metrics(actual, preds["lightgbm_q10"], preds["lightgbm_q90"])
        all_intervals.append({"fold": label, "nominal_%": 80, **iv})
        all_preds.append(preds.assign(fold=label))
        if label == HEADLINE:  # headline fold = the latest complete test year
            preds.to_csv(results / "predictions.csv")
            table.drop(columns="fold").round(3).to_csv(results / "metrics.csv")
            models = preds[MODEL_COLUMNS]
            mape_by(actual, models, models.index.hour).round(3).to_csv(results / "mape_by_hour.csv")
            mape_by(actual, models, models.index.to_period("M")).round(3).to_csv(results / "mape_by_month.csv")

    pd.concat(all_metrics).round(3).to_csv(results / "metrics_all_folds.csv", index=False)
    pd.DataFrame(all_intervals).round(2).to_csv(results / "interval_metrics.csv", index=False)
    pd.concat(all_preds).to_csv(results / "predictions_all_folds.csv")
    print("\n" + pd.concat(all_metrics).round(2).to_string(index=False))
    print("\n" + pd.DataFrame(all_intervals).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
