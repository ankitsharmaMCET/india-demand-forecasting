"""Merge the ICED 'Yearly Demand Profile' XLS downloads in data/raw/ into data/raw/iced_hourly.csv."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from demand_forecast.data import load_hourly, prepare_iced  # noqa: E402


def main() -> None:
    out = prepare_iced(ROOT / "data/raw", ROOT / "data/raw/iced_hourly.csv")
    y = load_hourly(out)  # raises if there are gaps, duplicates or missing values
    print(f"{out}: {len(y):,} hourly values, {y.index.min()} -> {y.index.max()}, no gaps")
    print(y.groupby(y.index.year).agg(["count", "mean"]).round(0).to_string())


if __name__ == "__main__":
    main()
