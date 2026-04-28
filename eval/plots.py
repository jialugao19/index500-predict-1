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


def _build_nonoverlap_trade_table(part: pd.DataFrame, horizon_minutes: int) -> pd.DataFrame:
    """Build a non-overlap trade table from overlapping ETF predictions."""

    # Assign a stable within-day row index.
    trades = part.loc[:, ["date", "datetime", "pred", "label"]].copy()
    trades["row_in_day"] = trades.groupby("date", sort=True).cumcount()

    # Keep only non-overlap entry rows.
    trades = trades.loc[(trades["row_in_day"].astype(int) % int(horizon_minutes)) == 0].copy()

    # Compute trade-level returns from non-overlap holding windows.
    trades["position"] = np.sign(trades["pred"].astype(float))
    trades["strategy_ret"] = trades["position"].astype(float) * trades["label"].astype(float)
    return trades


def _attach_trade_exit_datetime(trades: pd.DataFrame, etf_price_table: pd.DataFrame, horizon_minutes: int) -> pd.DataFrame:
    """Attach exit timestamps to each non-overlap trade."""

    # Resolve each trade exit against the full ETF minute timeline.
    pieces: list[pd.DataFrame] = []
    for date, day_trades in trades.groupby("date", sort=True):
        # Load the full ETF minute timeline for the day.
        day_prices = etf_price_table.loc[etf_price_table["date"].astype(int) == int(date), ["datetime"]].copy()
        day_prices = day_prices.sort_values("datetime", ascending=True).reset_index(drop=True)

        # Map each entry row index to its exit timestamp.
        day_part = day_trades.copy()
        exit_idx = day_part["row_in_day"].astype(int).to_numpy() + int(horizon_minutes)
        day_part["exit_datetime"] = day_prices.iloc[exit_idx]["datetime"].to_numpy()
        pieces.append(day_part)
    out = pd.concat(pieces, axis=0, ignore_index=True)
    return out


def _expand_trade_curve_to_minute_grid(
    etf_price_table: pd.DataFrame,
    trades: pd.DataFrame,
    ret_col: str,
) -> pd.DataFrame:
    """Expand trade-level returns into a minute-level step curve."""

    # Build a minute grid from the ETF price table.
    minute_grid = etf_price_table.loc[:, ["datetime", "close"]].copy()
    minute_grid = minute_grid.sort_values("datetime", ascending=True).reset_index(drop=True)

    # Convert sequential trade returns into exit-time equity updates.
    equity = 1.0
    updates: list[dict] = []
    for row in trades.sort_values("exit_datetime", ascending=True).itertuples(index=False):
        # Compound the next trade return.
        equity = equity * (1.0 + float(getattr(row, ret_col)))
        updates.append({"datetime": pd.to_datetime(row.exit_datetime), "equity": float(equity)})

    # Forward-fill equity between trade exits.
    updates_df = pd.DataFrame(updates)
    if len(updates_df) == 0:
        minute_grid["equity"] = 1.0
        return minute_grid
    minute_grid = minute_grid.merge(updates_df, on="datetime", how="left")
    minute_grid["equity"] = minute_grid["equity"].ffill().fillna(1.0)
    return minute_grid


def compute_nonoverlap_backtest_summary(
    pred_table: pd.DataFrame,
    model_name: str,
    etf_price_table: pd.DataFrame,
    horizon_minutes: int,
) -> dict:
    """Compute summary stats for the ETF non-overlap backtest."""

    # Filter the selected model and align the ETF price panel.
    part = pred_table.loc[(pred_table["model_name"] == model_name) & (pred_table["split"] == "test")].copy()
    part = part.sort_values(["date", "datetime"], ascending=True).reset_index(drop=True)
    bench = etf_price_table.loc[:, ["date", "datetime", "close"]].copy()
    bench = bench.sort_values(["date", "datetime"], ascending=True).reset_index(drop=True)

    # Build the trade table and the realized strategy curve.
    trades = _build_nonoverlap_trade_table(part=part, horizon_minutes=horizon_minutes)
    trades = _attach_trade_exit_datetime(trades=trades, etf_price_table=bench, horizon_minutes=horizon_minutes)
    bench["benchmark_curve"] = bench["close"].astype(float) / float(bench["close"].astype(float).iloc[0])
    strategy = _expand_trade_curve_to_minute_grid(
        etf_price_table=bench.loc[:, ["datetime", "benchmark_curve"]].rename(columns={"benchmark_curve": "close"}),
        trades=trades,
        ret_col="strategy_ret",
    )

    # Summarize end value, excess return, and max drawdown.
    strategy_curve = strategy["equity"].astype(float).to_numpy()
    benchmark_curve = bench["benchmark_curve"].astype(float).to_numpy()
    excess_curve_bps = (strategy_curve / benchmark_curve - 1.0) * 1e4
    strategy_drawdown_pct = (strategy_curve / np.maximum.accumulate(strategy_curve) - 1.0) * 100.0
    benchmark_drawdown_pct = (benchmark_curve / np.maximum.accumulate(benchmark_curve) - 1.0) * 100.0
    return {
        "trade_count": int(len(trades)),
        "horizon_minutes": int(horizon_minutes),
        "strategy_end": float(strategy_curve[-1]),
        "strategy_total_return_pct": float((strategy_curve[-1] - 1.0) * 100.0),
        "benchmark_end": float(benchmark_curve[-1]),
        "benchmark_total_return_pct": float((benchmark_curve[-1] - 1.0) * 100.0),
        "excess_end_bps": float(excess_curve_bps[-1]),
        "strategy_max_drawdown_pct": float(np.min(strategy_drawdown_pct)),
        "benchmark_max_drawdown_pct": float(np.min(benchmark_drawdown_pct)),
    }


def plot_etf_backtest_compare(
    pred_table: pd.DataFrame,
    model_name: str,
    out_path: str,
    etf_price_table: pd.DataFrame,
    horizon_minutes: int,
) -> None:
    """Plot a 30m non-overlap ETF strategy against a true long ETF benchmark."""

    # Filter to the selected model and test split.
    part = pred_table.loc[(pred_table["model_name"] == model_name) & (pred_table["split"] == "test")].copy()
    part = part.sort_values(["date", "datetime"], ascending=True).reset_index(drop=True)
    etf_price_table = etf_price_table.loc[:, ["date", "datetime", "close"]].copy()
    etf_price_table = etf_price_table.sort_values(["date", "datetime"], ascending=True).reset_index(drop=True)

    # Build the non-overlap trade table.
    trades = _build_nonoverlap_trade_table(part=part, horizon_minutes=horizon_minutes)
    trades = _attach_trade_exit_datetime(trades=trades, etf_price_table=etf_price_table, horizon_minutes=horizon_minutes)

    # Build the minute-level benchmark and strategy curves.
    bench = etf_price_table.loc[:, ["date", "datetime", "close"]].copy()
    bench["benchmark_curve"] = bench["close"].astype(float) / float(bench["close"].astype(float).iloc[0])
    strategy = _expand_trade_curve_to_minute_grid(
        etf_price_table=bench.loc[:, ["datetime", "benchmark_curve"]].rename(columns={"benchmark_curve": "close"}),
        trades=trades,
        ret_col="strategy_ret",
    )
    strategy_curve = strategy["equity"].astype(float).to_numpy()
    benchmark_curve = bench["benchmark_curve"].astype(float).to_numpy()
    excess_curve_bps = (strategy_curve / benchmark_curve - 1.0) * 1e4
    strategy_drawdown_pct = (strategy_curve / np.maximum.accumulate(strategy_curve) - 1.0) * 100.0
    benchmark_drawdown_pct = (benchmark_curve / np.maximum.accumulate(benchmark_curve) - 1.0) * 100.0

    # Draw the cumulative curve panel.
    fig, (ax1, ax2, ax3) = plt.subplots(
        3,
        1,
        figsize=(12, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [1.8, 1.4, 1.2]},
    )
    ax1.plot(bench["datetime"], strategy_curve, label="Strategy: 30m non-overlap, sign(pred)", linewidth=1.2)
    ax1.plot(bench["datetime"], benchmark_curve, label="Benchmark: long ETF", linewidth=1.2)
    ax1.set_ylabel("Curve")
    ax1.set_title(f"ETF 30m Non-overlap Backtest vs Long ETF ({model_name}, test)")
    ax1.legend()

    # Draw the excess-return panel.
    ax2.plot(bench["datetime"], excess_curve_bps, label="Strategy excess vs long ETF", linewidth=1.1)
    ax2.axhline(0.0, color="black", linewidth=0.8, alpha=0.7)
    ax2.set_ylabel("Excess (bps)")
    ax2.legend()

    # Draw the drawdown panel.
    ax3.plot(bench["datetime"], strategy_drawdown_pct, label="Strategy drawdown", linewidth=1.0)
    ax3.plot(bench["datetime"], benchmark_drawdown_pct, label="Benchmark drawdown", linewidth=1.0, alpha=0.85)
    ax3.axhline(0.0, color="black", linewidth=0.8, alpha=0.7)
    ax3.set_ylabel("Drawdown (%)")
    ax3.legend()

    # Save the combined figure.
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)


def plot_rolling_ic_rankic(daily: pd.DataFrame, title: str, out_path: str) -> None:
    """Plot ETF rolling IC and RankIC curves."""

    # Build the datetime x-axis from the daily table.
    dates = pd.to_datetime(daily["date"].astype(str), format="%Y%m%d")

    # Draw the IC panel.
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    ax1.plot(dates, daily["ic"], label="daily_ic", linewidth=0.8, alpha=0.55)
    if "ic_roll_20" in daily.columns:
        ax1.plot(dates, daily["ic_roll_20"], label="ic_roll_20", linewidth=1.2)
    if "ic_roll_60" in daily.columns:
        ax1.plot(dates, daily["ic_roll_60"], label="ic_roll_60", linewidth=1.2)
    ax1.axhline(0.0, color="black", linewidth=0.8, alpha=0.7)
    ax1.set_ylabel("IC")
    ax1.set_title(title)
    ax1.legend()

    # Draw the RankIC panel.
    ax2.plot(dates, daily["rank_ic"], label="daily_rank_ic", linewidth=0.8, alpha=0.55)
    if "rank_ic_roll_20" in daily.columns:
        ax2.plot(dates, daily["rank_ic_roll_20"], label="rank_ic_roll_20", linewidth=1.2)
    if "rank_ic_roll_60" in daily.columns:
        ax2.plot(dates, daily["rank_ic_roll_60"], label="rank_ic_roll_60", linewidth=1.2)
    ax2.axhline(0.0, color="black", linewidth=0.8, alpha=0.7)
    ax2.set_ylabel("RankIC")
    ax2.legend()

    # Save the combined figure.
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)


def plot_prediction_bucket_calibration_spread(
    pred_table: pd.DataFrame,
    model_name: str,
    out_path: str,
    n_buckets: int,
) -> None:
    """Plot ETF prediction bucket calibration and spread on the test split."""

    # Filter to the selected model and test split.
    part = pred_table.loc[(pred_table["model_name"] == model_name) & (pred_table["split"] == "test"), ["pred", "label"]].copy()

    # Build prediction quantile buckets with stable tie handling.
    rank = part["pred"].rank(method="first")
    part["bucket"] = pd.qcut(rank, q=int(n_buckets), labels=False).astype(int) + 1

    # Summarize bucket-level calibration statistics.
    bucket_table = (
        part.groupby("bucket", sort=True)
        .agg(pred_mean=("pred", "mean"), label_mean=("label", "mean"), n=("pred", "size"))
        .reset_index()
    )
    bucket_table["pred_mean_bps"] = bucket_table["pred_mean"].astype(float) * 1e4
    bucket_table["label_mean_bps"] = bucket_table["label_mean"].astype(float) * 1e4
    top_bottom_spread_bps = float(bucket_table["label_mean_bps"].iloc[-1] - bucket_table["label_mean_bps"].iloc[0])

    # Draw the calibration panel.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))
    ax1.plot(bucket_table["bucket"], bucket_table["pred_mean_bps"], marker="o", linewidth=1.2, label="pred_mean")
    ax1.plot(bucket_table["bucket"], bucket_table["label_mean_bps"], marker="o", linewidth=1.2, label="label_mean")
    ax1.axhline(0.0, color="black", linewidth=0.8, alpha=0.7)
    ax1.set_xlabel("Prediction bucket")
    ax1.set_ylabel("Return (bps)")
    ax1.set_title("Prediction bucket calibration")
    ax1.legend()

    # Draw the realized spread panel.
    ax2.bar(bucket_table["bucket"].astype(str), bucket_table["label_mean_bps"])
    ax2.axhline(0.0, color="black", linewidth=0.8, alpha=0.7)
    ax2.set_xlabel("Prediction bucket")
    ax2.set_ylabel("Realized return (bps)")
    ax2.set_title(f"Realized spread, top-bottom={top_bottom_spread_bps:.2f} bps")

    # Save the combined figure.
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)


def plot_feature_importance(importances: list[dict], out_path: str, top_k: int, title: str) -> None:
    """Plot a horizontal bar chart of top-K feature importances."""

    # Convert importance rows into a sorted DataFrame.
    table = pd.DataFrame(importances).copy()
    table["importance"] = table["importance"].astype(float)
    table = table.sort_values("importance", ascending=False).head(int(top_k))

    # Draw a compact barh chart to show relative magnitudes.
    plt.figure(figsize=(8, 6))
    plt.barh(table["feature"][::-1], table["importance"][::-1])
    plt.title(f"{str(title)} (Top {int(top_k)})")
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
