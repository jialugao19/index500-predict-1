import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def plot_daily_timeseries(
    daily: pd.DataFrame,
    value_col: str,
    roll_20_col: str,
    roll_60_col: str,
    title: str,
    out_path: str,
) -> None:
    """Plot daily metric timeseries with rolling averages."""

    # Build x-axis as datetime for better rendering.
    dates = pd.to_datetime(daily["date"].astype(str), format="%Y%m%d")

    # Create and save the plot.
    plt.figure(figsize=(12, 4))
    plt.plot(dates, daily[value_col], label=value_col, linewidth=0.8)
    if roll_20_col in daily.columns:
        plt.plot(dates, daily[roll_20_col], label=roll_20_col, linewidth=1.2)
    if roll_60_col in daily.columns:
        plt.plot(dates, daily[roll_60_col], label=roll_60_col, linewidth=1.2)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_baseline_comparison(metrics: dict, split: str, out_path: str) -> None:
    """Plot a baseline comparison chart on a given split."""

    # Collect metrics for all models on the split.
    models = sorted(metrics["overall"].keys())
    ic_values = [metrics["overall"][model].get(split, {}).get("ic", np.nan) for model in models]
    rmse_values = [metrics["overall"][model].get(split, {}).get("rmse", np.nan) for model in models]

    # Create a simple two-panel bar chart.
    plt.figure(figsize=(12, 4))
    ax1 = plt.subplot(1, 2, 1)
    ax1.bar(models, ic_values)
    ax1.set_title(f"IC ({split})")
    ax1.tick_params(axis="x", rotation=45)

    ax2 = plt.subplot(1, 2, 2)
    ax2.bar(models, rmse_values)
    ax2.set_title(f"RMSE ({split})")
    ax2.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_prediction_scatter(pred_table: pd.DataFrame, model_name: str, out_path: str) -> None:
    """Plot scatter of predictions vs labels on test split."""

    # Filter to test split.
    part = pred_table.loc[(pred_table["model_name"] == model_name) & (pred_table["split"] == "test")].copy()

    # Create scatter plot with alpha for density.
    plt.figure(figsize=(5, 5))
    plt.scatter(part["pred"], part["label"], s=6, alpha=0.2)
    plt.title(f"Prediction Scatter ({model_name}, test)")
    plt.xlabel("pred")
    plt.ylabel("label")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_prediction_timeseries(pred_table: pd.DataFrame, model_name: str, out_path: str) -> None:
    """Plot prediction and label timeseries on test split (daily mean)."""

    # Aggregate to daily mean for readability.
    part = pred_table.loc[(pred_table["model_name"] == model_name) & (pred_table["split"] == "test")].copy()
    daily = part.groupby("date", sort=True).agg(pred=("pred", "mean"), label=("label", "mean")).reset_index()
    dates = pd.to_datetime(daily["date"].astype(str), format="%Y%m%d")

    # Plot daily mean pred and label.
    plt.figure(figsize=(12, 4))
    plt.plot(dates, daily["pred"], label="pred_mean", linewidth=1.0)
    plt.plot(dates, daily["label"], label="label_mean", linewidth=1.0)
    plt.title(f"Prediction Timeseries ({model_name}, test, daily mean)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_feature_importance(importances: list[dict], out_path: str, top_k: int) -> None:
    """Plot a horizontal bar chart of top-K feature importances."""

    # Convert importance rows into a sorted DataFrame.
    table = pd.DataFrame(importances).copy()
    table["importance"] = table["importance"].astype(float)
    table = table.sort_values("importance", ascending=False).head(int(top_k))

    # Draw a compact barh chart to show relative magnitudes.
    plt.figure(figsize=(8, 6))
    plt.barh(table["feature"][::-1], table["importance"][::-1])
    plt.title(f"XGB Feature Importance (Top {int(top_k)})")
    plt.xlabel("importance")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_cumulative_metric(daily: pd.DataFrame, value_col: str, title: str, out_path: str) -> None:
    """Plot cumulative sum of a daily metric series."""

    # Build x-axis as datetime and compute the cumulative series.
    dates = pd.to_datetime(daily["date"].astype(str), format="%Y%m%d")
    values = daily[value_col].astype(float).fillna(0.0).to_numpy()
    cumulative = np.cumsum(values)

    # Create and save a compact cumulative curve chart.
    plt.figure(figsize=(12, 4))
    plt.plot(dates, cumulative, label=f"cumsum({value_col})", linewidth=1.2)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_monthly_timeseries(
    monthly: pd.DataFrame,
    value_col: str,
    title: str,
    out_path: str,
) -> None:
    """Plot monthly metric timeseries."""

    # Convert YYYYMM into a month-start datetime index for plotting.
    months = monthly["month"].astype(int)
    dates = pd.to_datetime(months.astype(str) + "01", format="%Y%m%d")

    # Create and save a compact line plot.
    plt.figure(figsize=(12, 4))
    plt.plot(dates, monthly[value_col], label=value_col, linewidth=1.2, marker="o", markersize=3)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_raw_vs_basis_delta(
    overall_metrics: dict,
    split: str,
    pairs: list[tuple[str, str]],
    out_path: str,
) -> None:
    """Plot IC/RankIC deltas between raw and basis-corrected models."""

    # Build a small table of deltas for each (raw, basis) pair.
    rows: list[dict] = []
    for raw_name, basis_name in pairs:
        # Pull split metrics for both models.
        raw_row = overall_metrics.get(raw_name, {}).get(split, {})
        basis_row = overall_metrics.get(basis_name, {}).get(split, {})
        rows.append(
            {
                "pair": f"{raw_name} -> {basis_name}",
                "ic_delta": float(basis_row.get("ic", np.nan)) - float(raw_row.get("ic", np.nan)),
                "rank_ic_delta": float(basis_row.get("rank_ic", np.nan)) - float(raw_row.get("rank_ic", np.nan)),
            }
        )
    table = pd.DataFrame(rows).copy()

    # Plot two-panel bars for IC and RankIC deltas.
    plt.figure(figsize=(12, 4))
    ax1 = plt.subplot(1, 2, 1)
    ax1.bar(table["pair"], table["ic_delta"])
    ax1.axhline(0.0, color="black", linewidth=0.8)
    ax1.set_title(f"IC Delta (basis - raw, {split})")
    ax1.tick_params(axis="x", rotation=45)

    ax2 = plt.subplot(1, 2, 2)
    ax2.bar(table["pair"], table["rank_ic_delta"])
    ax2.axhline(0.0, color="black", linewidth=0.8)
    ax2.set_title(f"RankIC Delta (basis - raw, {split})")
    ax2.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_histogram(values: np.ndarray, bins: int, title: str, out_path: str) -> None:
    """Plot a simple histogram for a 1D numeric series."""

    # Keep only finite values for stable histogram rendering.
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]

    # Draw and save a compact histogram.
    plt.figure(figsize=(8, 4))
    plt.hist(vals, bins=int(bins), alpha=0.85)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
