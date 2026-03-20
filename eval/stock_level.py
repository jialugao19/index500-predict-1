import numpy as np
import pandas as pd

from eval.metrics import safe_corr, safe_spearman


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

