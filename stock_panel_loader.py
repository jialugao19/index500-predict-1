import os
import pathlib
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from features.builders.stock import build_stock_feature_panel_day
from features.registry import load_registry


def ensure_dir(path: str) -> None:
    """Ensure a directory exists."""

    # Create directory recursively.
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)


def date_to_split(date: int) -> str:
    """Map date to split label using AGENTS.md time partition."""

    # Apply the required split rules.
    if 20210101 <= date <= 20231231:
        return "train"
    if date >= 20240201:
        return "test"
    return "ignore"


def load_index_weights(weight_path: str) -> pd.DataFrame:
    """Load and preprocess index weights for 000905."""

    # Load weights from feather.
    weights = pd.read_feather(weight_path)

    # Normalize columns for later joins and lookups.
    weights = weights.loc[:, ["con_code", "trade_date", "weight"]].copy()
    weights["trade_date"] = weights["trade_date"].astype(int)
    weights["con_int"] = weights["con_code"].str.slice(0, 6).astype(int)
    weights["weight_frac"] = weights["weight"].astype(float) / 100.0

    # Sort for fast latest-trade-date lookup.
    weights = weights.sort_values(["trade_date", "con_int"], ascending=[True, True])
    return weights


def get_constituent_weights_for_date(weights: pd.DataFrame, date: int) -> pd.DataFrame:
    """Get the latest constituent weights as-of a given date."""

    # Select the latest available trade_date not after the date.
    trade_dates = np.sort(weights["trade_date"].unique())
    chosen_trade_date = trade_dates[trade_dates <= date].max()

    # Slice weights for the chosen trade_date.
    day_weights = weights.loc[weights["trade_date"] == chosen_trade_date, ["con_int", "weight_frac"]].copy()
    return day_weights


def load_stock_minute_bars(stock1m_root: str, date: int, constituents: np.ndarray) -> pd.DataFrame:
    """Load one-day stock 1m bars filtered by constituents."""

    # Build the path and load the day file.
    year = str(date)[:4]
    file_path = os.path.join(stock1m_root, year, f"{date}.feather")
    day = pd.read_feather(file_path)

    # Filter to constituents and keep only required columns.
    mask = day["StockCode"].isin(constituents)
    day = day.loc[
        mask,
        ["StockCode", "DateTime", "Open", "High", "Low", "Close", "Vol", "Amount", "MinuteIndex", "Date"],
    ].copy()
    day = day.sort_values(["StockCode", "DateTime"], ascending=[True, True])

    # Filter non-trading minutes to match AGENTS.md.
    times = pd.to_datetime(day["DateTime"]).dt.time
    t0930 = pd.to_datetime("09:30:00").time()
    t1130 = pd.to_datetime("11:30:00").time()
    t1300 = pd.to_datetime("13:00:00").time()
    t1500 = pd.to_datetime("15:00:00").time()
    in_am = (times >= t0930) & (times <= t1130)
    in_pm = (times >= t1300) & (times <= t1500)
    day = day.loc[in_am | in_pm].copy()
    return day


def build_stock_panel_day(
    date: int,
    weights: pd.DataFrame,
    stock1m_root: str,
    horizon_minutes: int,
) -> pd.DataFrame:
    """Build one-day stock-level panel with features, weights, and label."""

    # Build the clean base panel first and then delegate factor computation to the feature builder.
    base = build_stock_base_panel_day(date=date, weights=weights, stock1m_root=stock1m_root, horizon_minutes=horizon_minutes)
    out = build_stock_feature_panel_day(
        base_panel=base,
        specs_root=os.path.join(os.path.dirname(os.path.abspath(__file__)), "features", "specs"),
        factor_set_name="stock_default",
    )

    # Attach split as a constant per day so downstream filters stay unchanged.
    out["split"] = date_to_split(date=int(date))
    return out


def build_stock_base_panel_day(
    date: int,
    weights: pd.DataFrame,
    stock1m_root: str,
    horizon_minutes: int,
) -> pd.DataFrame:
    """Build one-day clean base panel (bars + weights + invalid gating)."""

    # Load weights as-of date and constituent stock bars.
    day_weights = get_constituent_weights_for_date(weights=weights, date=date)
    constituents = day_weights["con_int"].to_numpy()
    stock_day = load_stock_minute_bars(stock1m_root=stock1m_root, date=date, constituents=constituents)

    # Join weights so each stock row has its index weight.
    merged = stock_day.merge(day_weights, left_on="StockCode", right_on="con_int", how="inner")
    merged = merged.drop(columns=["con_int"]).rename(columns={"weight_frac": "weight"})

    # Mark suspension-like abnormal minutes and replace base fields with NaN on invalid bars.
    grp = merged.groupby("StockCode", sort=False)
    raw_ret_1 = merged["Close"].astype(float) / grp["Close"].shift(1).astype(float) - 1.0
    limit_like = (merged["High"].astype(float) == merged["Low"].astype(float)) & np.isfinite(raw_ret_1.to_numpy(dtype=float)) & (
        np.abs(raw_ret_1.to_numpy(dtype=float)) >= 0.095
    )
    invalid_bar = (
        (merged["Vol"].astype(float) <= 0.0)
        | (merged["Amount"].astype(float) <= 0.0)
        | (~np.isfinite(merged["Open"].astype(float)))
        | (~np.isfinite(merged["High"].astype(float)))
        | (~np.isfinite(merged["Low"].astype(float)))
        | (~np.isfinite(merged["Close"].astype(float)))
        | limit_like
    )
    merged["invalid_bar"] = invalid_bar.astype(int)
    merged.loc[invalid_bar, ["Open", "High", "Low", "Close", "Vol", "Amount"]] = np.nan

    # Compute stock forward return label aligned to time t.
    close_future = grp["Close"].shift(-int(horizon_minutes))
    merged["label_stock_10m"] = close_future.astype(float) / merged["Close"].astype(float) - 1.0

    # Pack into a standardized base panel schema for downstream factor builders.
    out = merged.loc[
        :,
        [
            "Date",
            "DateTime",
            "StockCode",
            "MinuteIndex",
            "weight",
            "label_stock_10m",
            "Open",
            "High",
            "Low",
            "Close",
            "Vol",
            "Amount",
            "invalid_bar",
        ],
    ].copy()
    out = out.rename(columns={"Date": "date", "DateTime": "datetime", "StockCode": "stock_code"})
    out["datetime"] = pd.to_datetime(out["datetime"]).astype("datetime64[us]")
    out["split"] = date_to_split(date=int(date))
    return out


def load_or_build_stock_panel_day(
    date: int,
    weights: pd.DataFrame,
    stock1m_root: str,
    horizon_minutes: int,
    base_cache_root: str,
    feature_cache_root: str,
    specs_root: str,
    factor_set_name: str,
) -> pd.DataFrame:
    """Load cached stock panel day or build and cache it."""

    # Decide cache file path for the feature panel.
    ensure_dir(feature_cache_root)
    cache_path = os.path.join(feature_cache_root, f"{date}.parquet")

    # Load from cache if available.
    if os.path.exists(cache_path):
        day = pd.read_parquet(cache_path)
        day["datetime"] = pd.to_datetime(day["datetime"]).astype("datetime64[us]")
        return day

    # Load or build the base panel under its own cache.
    base = load_or_build_stock_base_panel_day(
        date=date,
        weights=weights,
        stock1m_root=stock1m_root,
        horizon_minutes=horizon_minutes,
        base_cache_root=base_cache_root,
    )

    # Build the feature panel and attach split for downstream filters.
    day = build_stock_feature_panel_day(
        base_panel=base,
        specs_root=specs_root,
        factor_set_name=factor_set_name,
    )
    day["split"] = date_to_split(date=int(date))

    # Cache the built panel as a single parquet file for repeated runs.
    day.to_parquet(cache_path, index=False)
    return day


def load_or_build_stock_base_panel_day(
    date: int,
    weights: pd.DataFrame,
    stock1m_root: str,
    horizon_minutes: int,
    base_cache_root: str,
) -> pd.DataFrame:
    """Load cached base panel day or build and cache it."""

    # Decide base cache file path.
    ensure_dir(base_cache_root)
    cache_path = os.path.join(base_cache_root, f"{date}.parquet")

    # Load from cache if available.
    if os.path.exists(cache_path):
        day = pd.read_parquet(cache_path)
        day["datetime"] = pd.to_datetime(day["datetime"]).astype("datetime64[us]")
        return day

    # Build base panel day and cache it.
    day = build_stock_base_panel_day(date=date, weights=weights, stock1m_root=stock1m_root, horizon_minutes=horizon_minutes)
    day.to_parquet(cache_path, index=False)
    return day


def write_stock_panel_parquet(
    dates: list[int],
    weights: pd.DataFrame,
    stock1m_root: str,
    horizon_minutes: int,
    base_cache_root: str,
    feature_cache_root: str,
    specs_root: str,
    factor_set_name: str,
    out_path: str,
) -> dict:
    """Write stock_panel.parquet by streaming day panels into one parquet file."""

    # Load factor set once so coverage diagnostics and feature list stay consistent.
    registry = load_registry(specs_root=specs_root)
    factor_set = registry.factor_sets[str(factor_set_name)]
    feature_cols = list(factor_set.factors) + ["minute_of_day", "is_open_30min", "is_close_30min"]

    # Create the output parent directory.
    ensure_dir(os.path.dirname(out_path))

    # Remove existing output to avoid mixing partial runs.
    if os.path.exists(out_path):
        os.remove(out_path)

    # Stream day panels to one parquet file to avoid holding the full dataset in memory.
    writer: pq.ParquetWriter | None = None
    rows_written = 0
    non_null_counts = {str(col): 0 for col in feature_cols}
    start_time = time.time()
    for idx, date in enumerate(dates):
        # Print periodic progress for long runs.
        if int(idx) % 25 == 0:
            elapsed = time.time() - start_time
            print(f"[INFO] stock_panel progress day={int(idx)}/{len(dates)} rows_written={int(rows_written)} elapsed={elapsed:.1f}s.")

        # Build or load one-day panel and keep only train/test splits.
        day = load_or_build_stock_panel_day(
            date=int(date),
            weights=weights,
            stock1m_root=stock1m_root,
            horizon_minutes=horizon_minutes,
            base_cache_root=base_cache_root,
            feature_cache_root=feature_cache_root,
            specs_root=specs_root,
            factor_set_name=factor_set_name,
        )
        day = day.loc[day["split"].isin(["train", "test"])].copy()

        # Drop rows with truncated labels due to horizon.
        day = day.dropna(subset=["label_stock_10m"])

        # Update per-feature coverage counters for basic missingness diagnostics.
        counts = day.loc[:, feature_cols].notna().sum().to_dict()
        for col in feature_cols:
            non_null_counts[str(col)] += int(counts[str(col)])

        # Initialize parquet writer with the first batch schema.
        table = pa.Table.from_pandas(day, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(out_path, table.schema, compression="zstd")

        # Append the batch as one row group.
        writer.write_table(table)
        rows_written += int(len(day))

    # Finalize the parquet file.
    assert writer is not None
    writer.close()

    # Return basic metadata for downstream training code.
    feature_coverage = [
        {"feature": str(col), "non_null_rate": float(non_null_counts[str(col)] / float(rows_written))} for col in feature_cols
    ]
    return {"rows_written": rows_written, "feature_cols": feature_cols, "feature_coverage": feature_coverage}
