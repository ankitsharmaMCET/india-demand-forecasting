# India day-ahead electricity demand forecasting

Day-ahead **hourly** forecasts of India's national electricity demand. Two naive baselines, a classical statistical model (Holt-Winters), Ridge regression and LightGBM are compared under a strict walk-forward evaluation on four held-out periods, with prediction intervals, a validation-year hyper-parameter search, leakage tests and a walkthrough notebook.

**Result.** On the latest complete test year (1 May 2024 to 30 Apr 2025, 8,760 hours) LightGBM reaches **1.62 % MAPE** (MAE 3.2 GW). That is 39 % lower MAE than repeating yesterday's value and 59 % lower than repeating last week's. The LightGBM MAPE is 1.63, 1.65 and 1.62 on the three complete test years from May 2022 and 1.63 on the 11 months to March 2026.

![Accuracy by model](results/figures/04_accuracy.png)

Test year 2024-25 (MAPE is the mean absolute percentage error over all hours; "daily peak" scores the forecast of each day's maximum):

| Model | MAE (MW) | RMSE (MW) | MAPE (%) | Daily-peak MAPE (%) |
|---|---:|---:|---:|---:|
| **LightGBM** (lag, calendar and last-hours features) | **3,152** | **4,512** | **1.62** | 1.71 |
| LightGBM without the last-hours features | 3,323 | 4,690 | 1.72 | **1.65** |
| Ridge regression (basic features) | 3,709 | 5,071 | 1.94 | 1.79 |
| Naive: same hour yesterday | 5,136 | 7,001 | 2.67 | 2.44 |
| Naive: same hour last week | 7,732 | 10,192 | 4.03 | 3.77 |
| Holt-Winters (ETS), refitted daily | 9,664 | 14,280 | 4.94 | 6.23 |

MAPE (%) on every test fold (`results/metrics_all_folds.csv` has every metric):

| Model | 2022-23 | 2023-24 | 2024-25 | 2025-26 (May to Mar) |
|---|---:|---:|---:|---:|
| LightGBM | 1.63 | 1.65 | 1.62 | 1.63 |
| LightGBM without last-hours features | 1.70 | 1.83 | 1.72 | 1.77 |
| Ridge | 1.97 | 2.06 | 1.94 | 2.08 |
| Naive: yesterday | 2.60 | 2.67 | 2.67 | 2.69 |
| Holt-Winters | 3.96 | 3.98 | 4.94 | 3.92 |
| Naive: last week | 4.30 | 4.69 | 4.03 | 3.88 |

In 2024-25 LightGBM beats the yesterday baseline in all 12 months and beats Ridge in 11 of 12; it beats the version without last-hours features in 9 of 12 months and is slightly worse on the daily-peak measure (1.71 against 1.65). Its error is largest in October 2024 (2.09 %) and September 2024 (2.07 %). Public holidays are harder than ordinary days (3.18 % against 1.54 % MAPE). Average bias is small, about +0.09 GW. The causes of the worst months have not been investigated, and the model has no weather input.

![One week of forecasts](results/figures/02_week_forecast.png)

## The forecasting task

A forecast is issued at **00:00 of day D** and must predict all 24 hours of day D, using only data up to 23:00 of day D-1. This matches how a day-ahead schedule is produced, and it constrains the features:

* Same hour 1, 2, 3, 7 and 14 days earlier (lags of 24 hours or more).
* Daily summaries (mean, max, min) of day D-1 and a 7-day mean.
* The **last observed hours**: the 23:00 value and the 18:00 to 23:00 mean of day D-1, each relative to the same hours a week earlier. These carry the most recent momentum, and the model combines them with the hour of day to know how far ahead it is forecasting.
* Calendar features: hour, weekday, month, day of year (sine and cosine), and Indian public holidays (today, yesterday, and the same day last week).
* Lags of 1 to 23 hours are **not** used, because at midnight the value one hour before 05:00 is not yet known.

Demand grew about 5 % a year on average from 2017 to 2024 (about 2 % in 2025) and tree models cannot extrapolate beyond the range they were trained on. So the ML models predict the **ratio** `demand / same hour last week` and multiply back, and every demand-level feature is expressed as a ratio too.

![What the model relies on](results/figures/06_feature_importance.png)

## Evaluation protocol

* **Four test folds:** May 2022 to Apr 2023, May 2023 to Apr 2024, May 2024 to Apr 2025 (the headline year) and May 2025 to Mar 2026 (11 months, where the data ends). Nothing in them is used for training or for choosing settings.
* **Walk-forward:** Ridge and LightGBM are retrained at the start of every month on all data before that month (expanding window, from January 2017), then forecast each day of that month.
* **Hyper-parameter search** (`scripts/tune.py`): 12 LightGBM settings are scored on a separate **validation year, May 2021 to Apr 2022**, which lies entirely before all test folds. Candidates are retrained quarterly and ranked by MAE. The selected setting (63 leaves, minimum 50 samples per leaf, learning rate 0.06, 250 trees) is only slightly better than the others: best and worst differ by 2.9 % in MAE, so tuning matters little here (`results/tuning.csv`). The best setting has the largest number of leaves in the grid, so a larger value was not tried.
* **Holt-Winters:** refitted every day on the previous 42 days, additive damped trend and one 168-hour (daily plus weekly) seasonal pattern. Its window length was not tuned.
* **COVID-19:** rows from 25 Mar to 30 Jun 2020 are excluded from training, because demand then did not behave normally. This is a judgement call, and its effect has not been measured.
* **Leakage tests** (`tests/`): changing every value from day D onward must not change day D's features, and corrupting the later part of a test window must not change forecasts for earlier hours.

## Error by hour of day

![Error by hour](results/figures/03_error_by_hour.png)

Holt-Winters beats LightGBM only at 00:00 (0.78 % against 0.90 %), the hour nearest the last observation. The last-hours features were added for this reason: they cut LightGBM's error at midnight from 1.45 % to 0.90 %.

## Prediction intervals

![80% prediction interval](results/figures/05_prediction_interval.png)

Quantile LightGBM at the 10th and 90th percentiles gives an 80 % interval about 6 to 7 GW wide (roughly 3.4 to 3.7 % of demand). It is **too narrow**: it contains only 62 % to 66 % of the actual values across the four folds (64.5 % in 2024-25) instead of 80 %. Calibrating the interval, for example with conformal prediction, is the obvious fix and is not done here (`results/interval_metrics.csv`).

## The data

![Demand series](results/figures/01_demand_series.png)

Hourly national demand met in MW, 1 Jan 2017 to 31 Mar 2026 (81,048 hours, no gaps or duplicates, verified on load). It comes from the **Yearly Demand Profile** chart of NITI Aayog's [India Climate & Energy Dashboard](https://iced.niti.gov.in/energy/electricity/distribution/national-level-consumption/load-curve), which credits Grid-India as the source and offers the series as XLS downloads (two years per file). The dashboard notes that some of its data are derived or assumed, so this is an official aggregator and not a primary Grid-India file.

An earlier version of this project used a copy of the same series from the Grid-Sentinel repository (Jan 2019 to Apr 2024). On the 46,728 hours the two cover in common, the values are identical.

The XLS files are **not committed** (size, and the dashboard's own terms apply). Download them yourself, see Reproduce below.

## Reproduce

1. Download the data. Open the dashboard page above, pick two years in "Yearly Demand Profile", click **Apply Filters**, then the **XLS** icon. Repeat so that every year from 2017 to 2026 is covered (2017+2018, 2019+2020, 2021+2022, 2023+2024, 2025+2026). Put the files in `data/raw/` without renaming them (`Yearly Demand Profile.xlsx`, `Yearly Demand Profile (1).xlsx`, ...). Overlapping years are merged and must agree.
2. Run:

```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt && pip install -e .
python scripts/prepare_data.py       # merges the XLS files into data/raw/iced_hourly.csv and checks them
python scripts/tune.py               # validation-year search (about 1 minute), writes results/best_params.json
python scripts/run_experiment.py     # all four test folds (about 10 minutes), writes results/
python scripts/make_figures.py       # redraws the figures
pytest                               # 20 tests
```

`notebooks/walkthrough.ipynb` explains the data, the setup and the results, and needs the `results/` files above. Install `jupyter` to rerun it.

Outputs in `results/`: `metrics.csv` (headline fold), `metrics_all_folds.csv`, `interval_metrics.csv`, `tuning.csv`, `best_params.json`, `predictions.csv` (headline fold) and `predictions_all_folds.csv` (actual and every model's forecast per hour), `mape_by_hour.csv`, `mape_by_month.csv`.

## Layout

```
src/demand_forecast/   data.py  features.py  models.py  evaluate.py  tuning.py  experiment.py
scripts/               prepare_data.py  tune.py  run_experiment.py  make_figures.py
notebooks/             walkthrough.ipynb
tests/                 leakage, metric, data-validation, tuning and backtest tests
results/               metrics, predictions, figures
.github/workflows/     tests.yml (runs pytest on every push)
```

## Limitations

* Four test periods of national data, so the ranking could differ in another region or grid.
* No weather, temperature or economic indicators. Demand at national level is strongly weather-driven.
* Prediction intervals are too narrow (see above).
* The best LightGBM setting sits at the edge of a small search grid.
* National totals only. A regional or state-level model would be more useful for grid operators.
* The hourly data ends in March 2026.
* Holt-Winters uses one untuned window length, so its result is a modest baseline and not a tuned statistical benchmark.
* The hourly series comes from a government dashboard that says some of its data are derived or assumed, not from a primary Grid-India file.

## Next steps

Add temperature as a feature, calibrate the prediction intervals, extend to regional demand, and backtest on more years as new data becomes available.

## Licence

Code: MIT (see `LICENSE`). The data keeps its own CC BY-SA 4.0 licence.
