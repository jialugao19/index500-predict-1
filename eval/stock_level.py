import numpy as np
import pandas as pd

from eval.metrics import safe_corr, safe_spearman


def compute_stock_sufficient_stats(
    day_panel: pd.DataFrame,
    pred: np.ndarray,
    label_col: str,
) -> pd.DataFrame:
    """Compute per-stock sufficient statistics for time-series metrics on one day."""

    # Build a minimal day frame with prediction and label.
    frame = day_panel.loc[:, ["stock_code", "weight", label_col]].copy()
    frame["pred"] = pred.astype(float)
    frame = frame.rename(columns={label_col: "label"})

    # Keep only finite pairs to match metric definitions.
    mask = np.isfinite(frame["pred"].to_numpy(dtype=float)) & np.isfinite(frame["label"].to_numpy(dtype=float))
    frame = frame.loc[mask].copy()

    # Precompute additive columns so groupby-sum yields global stats.
    frame["pred2"] = frame["pred"] * frame["pred"]
    frame["label2"] = frame["label"] * frame["label"]
    frame["pred_label"] = frame["pred"] * frame["label"]
    frame["abs_err"] = (frame["pred"] - frame["label"]).abs()
    frame["sq_err"] = (frame["pred"] - frame["label"]) * (frame["pred"] - frame["label"])
    frame["dir_correct"] = ((frame["pred"] > 0.0) == (frame["label"] > 0.0)).astype(float)

    # Aggregate sufficient stats by stock_code for later multi-day accumulation.
    grouped = frame.groupby("stock_code", sort=True).agg(
        n=("pred", "size"),
        weight_sum=("weight", "sum"),
        pred_sum=("pred", "sum"),
        label_sum=("label", "sum"),
        pred2_sum=("pred2", "sum"),
        label2_sum=("label2", "sum"),
        pred_label_sum=("pred_label", "sum"),
        abs_err_sum=("abs_err", "sum"),
        sq_err_sum=("sq_err", "sum"),
        dir_correct_sum=("dir_correct", "sum"),
    )
    out = grouped.reset_index()
    return out


def finalize_stock_metrics_from_sufficient_stats(stats: pd.DataFrame) -> pd.DataFrame:
    """Finalize per-stock metrics from accumulated sufficient statistics."""

    # Compute per-stock means and second-moment based variances.
    out = stats.copy()
    out["n"] = out["n"].astype(int)
    out["weight_mean"] = out["weight_sum"] / out["n"]
    out["pred_mean"] = out["pred_sum"] / out["n"]
    out["label_mean"] = out["label_sum"] / out["n"]
    out["pred_var"] = out["pred2_sum"] / out["n"] - out["pred_mean"] * out["pred_mean"]
    out["label_var"] = out["label2_sum"] / out["n"] - out["label_mean"] * out["label_mean"]

    # Compute Pearson IC using covariance and variance terms.
    cov = out["pred_label_sum"] / out["n"] - out["pred_mean"] * out["label_mean"]
    denom = np.sqrt(out["pred_var"] * out["label_var"])
    out["ic"] = cov / denom
    out.loc[(out["n"] < 2) | (~np.isfinite(denom)) | (denom == 0.0), "ic"] = float("nan")

    # Compute standard regression-style error metrics and direction hit rate.
    out["rmse"] = np.sqrt(out["sq_err_sum"] / out["n"])
    out["mae"] = out["abs_err_sum"] / out["n"]
    out["direction_acc"] = out["dir_correct_sum"] / out["n"]

    # Compute an approximate correlation t-statistic for quick good/bad tagging.
    out["ic_t"] = out["ic"] * np.sqrt((out["n"] - 2.0) / (1.0 - out["ic"] * out["ic"]))
    return out


def compute_stock_daily_ic_table(
    day_panel: pd.DataFrame,
    pred: np.ndarray,
    label_col: str,
) -> pd.DataFrame:
    """Compute per-stock within-day IC/RankIC table for one trading day."""

    # Build a minimal day frame with stock_code, label, and prediction.
    frame = day_panel.loc[:, ["stock_code", label_col]].copy()
    frame["pred"] = pred.astype(float)
    frame = frame.rename(columns={label_col: "label"})

    # Compute per-stock correlations across minutes within the day.
    rows: list[dict] = []
    for stock_code, part in frame.groupby("stock_code", sort=True):
        # Filter to finite pairs within the stock time series.
        pred_vec = part["pred"].to_numpy(dtype=float)
        label_vec = part["label"].to_numpy(dtype=float)
        mask = np.isfinite(pred_vec) & np.isfinite(label_vec)
        rows.append(
            {
                "stock_code": int(stock_code),
                "daily_ic": safe_corr(pred_vec[mask], label_vec[mask]) if int(np.sum(mask)) else float("nan"),
                "daily_rank_ic": safe_spearman(pred_vec[mask], label_vec[mask]) if int(np.sum(mask)) else float("nan"),
                "n": int(np.sum(mask)),
            }
        )

    # Return a stable table for downstream per-stock aggregation.
    out = pd.DataFrame(rows).sort_values("stock_code", ascending=True)
    return out


def compute_panel_ic_by_minute(
    day_panel: pd.DataFrame,
    pred: np.ndarray,
    label_col: str,
) -> pd.DataFrame:
    """Compute per-minute cross-sectional IC tables for a single trading day."""

    # Attach prediction into a minimal frame for grouping.
    frame = day_panel.loc[:, ["date", "datetime", "MinuteIndex", label_col]].copy()
    frame["pred"] = pred.astype(float)

    # Compute minute-level IC across stocks at the same timestamp.
    rows: list[dict] = []
    for (date, dt, minute_idx), part in frame.groupby(["date", "datetime", "MinuteIndex"], sort=True):
        # Compute IC on finite pairs within the minute cross-section.
        pred_vec = part["pred"].to_numpy(dtype=float)
        label_vec = part[label_col].to_numpy(dtype=float)
        mask = np.isfinite(pred_vec) & np.isfinite(label_vec)
        rows.append(
            {
                "date": int(date),
                "datetime": pd.to_datetime(dt),
                "minute_index": int(minute_idx),
                "ic": safe_corr(pred_vec[mask], label_vec[mask]) if int(np.sum(mask)) else float("nan"),
                "rank_ic": safe_spearman(pred_vec[mask], label_vec[mask]) if int(np.sum(mask)) else float("nan"),
                "n": int(np.sum(mask)),
            }
        )

    # Return a stable, sorted table for downstream bucket and daily summaries.
    out = pd.DataFrame(rows).sort_values(["date", "minute_index", "datetime"], ascending=[True, True, True])
    return out


def summarize_panel_ic_daily(minute_ic: pd.DataFrame) -> pd.DataFrame:
    """Summarize per-minute IC into per-day statistics."""

    # Aggregate minute-level IC into daily mean statistics.
    daily = (
        minute_ic.groupby("date", sort=True)
        .agg(
            ic_mean=("ic", "mean"),
            rank_ic_mean=("rank_ic", "mean"),
            n_sum=("n", "sum"),
            minutes=("minute_index", "nunique"),
        )
        .reset_index()
    )
    return daily


def summarize_panel_ic_by_minute_bucket(minute_ic: pd.DataFrame, bucket_size: int) -> pd.DataFrame:
    """Summarize per-minute IC by time-of-day buckets."""

    # Map minute indices into coarse buckets for intraday diagnostics.
    minute_ic = minute_ic.copy()
    minute_ic["minute_bucket"] = (minute_ic["minute_index"].astype(int) // int(bucket_size)).astype(int)

    # Aggregate IC by bucket across the test period.
    bucket = (
        minute_ic.groupby("minute_bucket", sort=True)
        .agg(
            ic_mean=("ic", "mean"),
            rank_ic_mean=("rank_ic", "mean"),
            n_sum=("n", "sum"),
            minutes=("minute_index", "nunique"),
        )
        .reset_index()
    )
    return bucket
