"""Build and load the hourly national demand series.

Source: NITI Aayog's India Climate & Energy Dashboard (ICED), "Yearly Demand Profile"
(hourly demand met in MW, credited on the dashboard to Grid-India). Each XLS download
holds two years, so several downloads are merged by :func:`prepare_iced`.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

RAW_DIR = Path("data/raw")
DEFAULT_PATH = RAW_DIR / "iced_hourly.csv"
DEMAND_COLUMN = "demand_mw"
XLSX_PATTERN = "Yearly Demand Profile*.xlsx"


def _read_one(path: Path) -> pd.DataFrame:
    d = pd.read_excel(path).iloc[:, :3]
    d.columns = ["year", "date", "mw"]
    d = d[pd.to_numeric(d["year"], errors="coerce").notna()].copy()  # drops the footer text
    d["year"] = d["year"].astype(int)
    d["datetime"] = pd.to_datetime(d["year"].astype(str) + " " + d["date"].astype(str),
                                   format="%Y %d-%b %I%p")
    d["mw"] = pd.to_numeric(d["mw"], errors="raise")
    return d[["datetime", "mw"]]


def prepare_iced(raw_dir: Path = RAW_DIR, dest: Path = DEFAULT_PATH) -> Path:
    """Merge the ICED XLS downloads in ``raw_dir`` into one hourly CSV at ``dest``.

    Years that appear in two downloads must agree exactly, otherwise ``ValueError``.
    """
    files = sorted(Path(raw_dir).glob(XLSX_PATTERN))
    if not files:
        raise FileNotFoundError(f"no '{XLSX_PATTERN}' files in {raw_dir}; see README, 'Data'")
    parts = pd.concat([_read_one(f) for f in files])
    if (parts.groupby("datetime")["mw"].nunique() > 1).any():
        raise ValueError("the same hour has different values in different downloads")
    merged = parts.drop_duplicates("datetime").sort_values("datetime")
    merged.rename(columns={"mw": DEMAND_COLUMN}).to_csv(dest, index=False)
    return Path(dest)


def load_hourly(path: Path = DEFAULT_PATH, column: str = DEMAND_COLUMN) -> pd.Series:
    """Return the demand series (MW) on a strict hourly index.

    Builds the CSV from the XLS downloads on first use. Raises ``ValueError`` if the
    index has duplicates, gaps or missing values, because every later step assumes a
    complete hourly grid.
    """
    path = Path(path)
    if not path.exists() and path == DEFAULT_PATH:
        prepare_iced()
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
