"""Run the full backtest and write metrics + predictions to results/."""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from demand_forecast.data import download, load_hourly  # noqa: E402
from demand_forecast.evaluate import mape_by, score_table  # noqa: E402
from demand_forecast.experiment import run_backtest  # noqa: E402


def main() -> None:
    results = ROOT / "results"
    results.mkdir(exist_ok=True)
    path = download(ROOT / "data/raw/study1_hourly.csv")
    y = load_hourly(path)
    print(f"Loaded {len(y):,} hourly values, {y.index.min()} -> {y.index.max()}")

    t0 = time.time()
    preds = run_backtest(y)
    print(f"Backtest finished in {time.time() - t0:.0f}s")
    preds.to_csv(results / "predictions.csv")

    actual, models = preds["actual"], preds.drop(columns="actual")
    table = score_table(actual, models)
    table.round(3).to_csv(results / "metrics.csv")
    mape_by(actual, models, models.index.hour).round(3).to_csv(results / "mape_by_hour.csv")
    mape_by(actual, models, models.index.to_period("M")).round(3).to_csv(results / "mape_by_month.csv")
    print(table.round(2).to_string())


if __name__ == "__main__":
    main()
