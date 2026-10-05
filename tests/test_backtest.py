import pandas as pd

from demand_forecast.evaluate import mape
from demand_forecast.experiment import run_backtest


def test_backtest_runs_and_beats_a_bad_baseline(synthetic):
    preds = run_backtest(synthetic, test_start="2021-04-10", test_end="2021-04-20 23:00",
                         include_ets=False, verbose=False)
    assert {"actual", "naive_24", "naive_168", "ridge", "lightgbm"} <= set(preds.columns)
    assert len(preds) == 11 * 24
    assert not preds.isna().any().any()
    # the synthetic series is highly regular, so the models must be accurate
    assert mape(preds["actual"], preds["lightgbm"]) < 3.0
    assert mape(preds["actual"], preds["ridge"]) < 3.0


def test_models_only_see_the_past(synthetic):
    """Corrupting the test window must not change forecasts for earlier hours."""
    kw = dict(test_start="2021-04-10", test_end="2021-04-20 23:00", include_ets=False, verbose=False)
    base = run_backtest(synthetic, **kw)
    corrupted = synthetic.copy()
    corrupted.loc["2021-04-15":] *= 5
    other = run_backtest(corrupted, **kw)
    early = base.index < "2021-04-15"
    pd.testing.assert_frame_equal(
        base.loc[early].drop(columns="actual"), other.loc[early].drop(columns="actual"))
