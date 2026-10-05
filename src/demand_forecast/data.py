"""Download and load the hourly national demand series."""
from __future__ import annotations

import urllib.request
from pathlib import Path

import pandas as pd

# Public mirror of hourly national demand (MW), Jan 2019 - Apr 2024.
# Published by HalcyonVector/Grid-Sentinel (CC BY-SA 4.0); see README for provenance.
DATA_URL = (
    "https://raw.githubusercontent.com/HalcyonVector/Grid-Sentinel/main/"
    "Dataset/study1_hourly.csv"
)
DEFAULT_PATH = Path("data/raw/study1_hourly.csv")
DEMAND_COLUMN = "National Hourly Demand"


def download(dest: Path = DEFAULT_PATH, url: str = DATA_URL) -> Path:
    """Download the CSV to ``dest`` (skipped if it already exists)."""
    dest = Path(dest)
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)
    return dest


def load_hourly(path: Path = DEFAULT_PATH, column: str = DEMAND_COLUMN) -> pd.Series:
    """Return the demand series (MW) on a strict hourly index.

    Raises ``ValueError`` if the index has duplicates, gaps or missing values,
    because every later step assumes a complete hourly grid.
    """
    df = pd.read_csv(path, usecols=["datetime", column], parse_dates=["datetime"])
    s = df.set_index("datetime")[column].sort_index().astype(float)
    s.index.name = "datetime"
    s.name = "demand_mw"
    if s.index.duplicated().any():
        raise ValueError("duplicate timestamps in demand series")
    full = pd.date_range(s.index.min(), s.index.max(), freq="h")
    if len(full) != len(s):
        raise ValueError(f"{len(full) - len(s)} missing hourly timestamps")
    if s.isna().any():
        raise ValueError("missing demand values")
    return s.asfreq("h")
