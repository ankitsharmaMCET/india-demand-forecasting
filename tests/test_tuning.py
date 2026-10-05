import pandas as pd

from demand_forecast.tuning import candidates, tune_lightgbm


def test_candidates_expand_the_grid():
    grid = {"num_leaves": [15, 31], "rate_trees": [(0.05, 50)], "min_child_samples": [20, 50]}
    cands = candidates(grid)
    assert len(cands) == 4
    assert all(c["learning_rate"] == 0.05 and c["n_estimators"] == 50 for c in cands)


def test_tuning_returns_sorted_scores(synthetic):
    grid = {"num_leaves": [7, 15], "rate_trees": [(0.1, 40)], "min_child_samples": [20]}
    table = tune_lightgbm(synthetic, "2021-04-01", "2021-04-30 23:00", grid, verbose=False)
    assert len(table) == 2
    assert table["val_MAE_MW"].is_monotonic_increasing
    assert {"num_leaves", "learning_rate", "n_estimators", "min_child_samples"} <= set(table.columns)
