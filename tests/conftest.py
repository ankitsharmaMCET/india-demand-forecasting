import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic() -> pd.Series:
    """120 days of hourly demand with daily + weekly cycles, growth and noise."""
    idx = pd.date_range("2021-01-01", periods=120 * 24, freq="h")
    rng = np.random.default_rng(0)
    hour = idx.hour.to_numpy()
    dow = idx.dayofweek.to_numpy()
    level = 150_000 * (1 + 0.0002 * np.arange(len(idx)) / 24)
    y = level * (1 + 0.15 * np.sin(2 * np.pi * (hour - 6) / 24) - 0.05 * (dow >= 5))
    y = y * (1 + rng.normal(0, 0.01, len(idx)))
    return pd.Series(y, index=idx, name="demand_mw")
