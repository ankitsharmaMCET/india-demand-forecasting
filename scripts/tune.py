"""Tune LightGBM on the validation year (2021-05 to 2022-04), before all test folds."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from demand_forecast.data import load_hourly  # noqa: E402
from demand_forecast.experiment import VALIDATION  # noqa: E402
from demand_forecast.tuning import tune_lightgbm  # noqa: E402


def main() -> None:
    y = load_hourly(ROOT / "data/raw/iced_hourly.csv")
    table = tune_lightgbm(y, *VALIDATION)
    (ROOT / "results").mkdir(exist_ok=True)
    table.round(3).to_csv(ROOT / "results/tuning.csv", index=False)
    best = table.iloc[0]
    params = {
        "num_leaves": int(best["num_leaves"]), "min_child_samples": int(best["min_child_samples"]),
        "learning_rate": float(best["learning_rate"]), "n_estimators": int(best["n_estimators"]),
    }
    (ROOT / "results/best_params.json").write_text(json.dumps(params, indent=2) + "\n")
    print("\nBest on validation year:", params)
    print(table.head(5).round(1).to_string())


if __name__ == "__main__":
    main()
