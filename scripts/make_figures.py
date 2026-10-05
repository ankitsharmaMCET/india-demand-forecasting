"""Create the figures in results/figures/ from results/*.csv and the raw data."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from demand_forecast.data import load_hourly  # noqa: E402
from demand_forecast.experiment import TEST_END, TEST_START  # noqa: E402

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
# One colour per entity, used identically in every figure. Baselines are neutral.
COLOR = {"lightgbm": "#2a78d6", "ridge": "#eb6834", "ets": "#1baf7a",
         "naive_24": "#52514e", "naive_168": "#8f8e87", "actual": INK}
STYLE = {"naive_24": (0, (5, 3)), "naive_168": (0, (1, 2))}
LABEL = {"lightgbm": "LightGBM", "ridge": "Ridge", "ets": "Holt-Winters (ETS)",
         "naive_24": "Naive: same hour yesterday", "naive_168": "Naive: same hour last week",
         "actual": "Actual"}


def style_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def new_fig(w=9, h=4.2):
    fig, ax = plt.subplots(figsize=(w, h), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    style_axes(ax)
    return fig, ax


def title(ax, text, sub=None):
    ax.set_title(text, loc="left", color=INK, fontsize=12, fontweight="bold", pad=22 if sub else 10)
    if sub:
        ax.text(0, 1.03, sub, transform=ax.transAxes, color=INK2, fontsize=9, va="bottom")


def legend(ax, **kw):
    leg = ax.legend(frameon=False, fontsize=8.5, labelcolor=INK2, **kw)
    return leg


def main() -> None:
    out = ROOT / "results/figures"
    out.mkdir(parents=True, exist_ok=True)
    y = load_hourly(ROOT / "data/raw/study1_hourly.csv")
    preds = pd.read_csv(ROOT / "results/predictions.csv", index_col=0, parse_dates=True)
    metrics = pd.read_csv(ROOT / "results/metrics.csv", index_col=0)
    by_hour = pd.read_csv(ROOT / "results/mape_by_hour.csv", index_col=0)

    # 1 - the series
    fig, ax = new_fig()
    daily = y.resample("D").mean() / 1000
    ax.plot(daily.index, daily.values, color=COLOR["lightgbm"], linewidth=0.9)
    ax.axvspan(pd.Timestamp(TEST_START), pd.Timestamp(TEST_END), color=GRID, alpha=0.7, lw=0)
    ax.text(pd.Timestamp(TEST_START), ax.get_ylim()[0] + 4, " test year", color=INK2, fontsize=8.5)
    ax.annotate("2020 lockdown", xy=(pd.Timestamp("2020-04-20"), daily["2020-04-20"]),
                xytext=(pd.Timestamp("2020-07-15"), 105), color=INK2, fontsize=8.5,
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    ax.set_ylabel("Mean daily demand (GW)", color=INK2, fontsize=9)
    title(ax, "India's national electricity demand, 2019 to 2024",
          "Daily mean of hourly demand. Shaded: the year held out for testing.")
    fig.tight_layout()
    fig.savefig(out / "01_demand_series.png", facecolor=SURFACE)
    plt.close(fig)

    # 2 - one week, forecast vs actual
    week = preds.loc["2023-08-21":"2023-08-27 23:00"]
    fig, ax = new_fig()
    for name in ("actual", "naive_168", "lightgbm"):
        ax.plot(week.index, week[name] / 1000, color=COLOR[name], linewidth=2 if name != "naive_168" else 1.6,
                linestyle=STYLE.get(name, "-"), label=LABEL[name])
    ax.set_ylabel("Demand (GW)", color=INK2, fontsize=9)
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%a %d %b"))
    title(ax, "Day-ahead forecast against actual demand, 21 to 27 Aug 2023",
          "Each day is forecast at 00:00 using only data up to the previous midnight.")
    legend(ax, loc="lower right", ncol=1)
    fig.tight_layout()
    fig.savefig(out / "02_week_forecast.png", facecolor=SURFACE)
    plt.close(fig)

    # 3 - error by hour of day
    fig, ax = new_fig()
    order = ["naive_168", "naive_24", "ets", "ridge", "lightgbm"]
    for name in order:
        ax.plot(by_hour.index, by_hour[name], color=COLOR[name], linewidth=2 if name == "lightgbm" else 1.6,
                linestyle=STYLE.get(name, "-"), label=LABEL[name])
    ax.set_xlabel("Hour of day (forecast target)", color=INK2, fontsize=9)
    ax.set_ylabel("MAPE (%)", color=INK2, fontsize=9)
    ax.set_xticks(range(0, 24, 3))
    title(ax, "Forecast error by hour of day, test year",
          "Holt-Winters beats the ML models only in the first hours after midnight, nearest the last observation.")
    legend(ax, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
    fig.tight_layout()
    fig.savefig(out / "03_error_by_hour.png", facecolor=SURFACE)
    plt.close(fig)

    # 4 - headline accuracy
    m = metrics.sort_values("MAPE_%", ascending=False)
    fig, ax = new_fig(9, 3.4)
    bars = ax.barh([LABEL[n] for n in m.index], m["MAPE_%"], color=[COLOR[n] for n in m.index], height=0.55)
    for b, v in zip(bars, m["MAPE_%"]):
        ax.text(v + 0.06, b.get_y() + b.get_height() / 2, f"{v:.2f}%", va="center", color=INK, fontsize=9)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_xlabel("MAPE (%), all hours of the test year", color=INK2, fontsize=9)
    ax.set_xlim(0, m["MAPE_%"].max() * 1.15)
    title(ax, "Day-ahead accuracy, 1 May 2023 to 30 Apr 2024", "Lower is better. Baselines in grey.")
    fig.tight_layout()
    fig.savefig(out / "04_accuracy.png", facecolor=SURFACE)
    plt.close(fig)
    print("figures written to", out)


if __name__ == "__main__":
    main()
