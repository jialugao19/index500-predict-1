import os
import pathlib
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


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
    day = day.loc[mask, ["StockCode", "DateTime", "Close", "Vol", "Amount", "MinuteIndex", "Date"]].copy()
    day = day.sort_values(["StockCode", "DateTime"], ascending=[True, True])
    return day


def build_stock_panel_day(
    date: int,
    weights: pd.DataFrame,
    stock1m_root: str,
    horizon_minutes: int,
) -> pd.DataFrame:
    """Build one-day stock-level panel with features, weights, and label."""

    # Load weights as-of date and constituent stock bars.
    day_weights = get_constituent_weights_for_date(weights=weights, date=date)
    constituents = day_weights["con_int"].to_numpy()
    stock_day = load_stock_minute_bars(stock1m_root=stock1m_root, date=date, constituents=constituents)

    # Join weights so each stock row has its index weight.
    merged = stock_day.merge(day_weights, left_on="StockCode", right_on="con_int", how="inner")
    merged = merged.drop(columns=["con_int"]).rename(columns={"weight_frac": "weight"})

    # Compute stock self features using only t and past information.
    merged["ret_1m"] = merged["Close"] / merged.groupby("StockCode")["Close"].shift(1) - 1.0
    merged["ret_5m"] = merged["Close"] / merged.groupby("StockCode")["Close"].shift(5) - 1.0
    merged["ret_10m"] = merged["Close"] / merged.groupby("StockCode")["Close"].shift(10) - 1.0
    merged["ret_30m"] = merged["Close"] / merged.groupby("StockCode")["Close"].shift(30) - 1.0
    merged["vol_chg_1m"] = merged["Vol"] / merged.groupby("StockCode")["Vol"].shift(1) - 1.0
    merged["amt_chg_1m"] = merged["Amount"] / merged.groupby("StockCode")["Amount"].shift(1) - 1.0
    merged = merged.replace([np.inf, -np.inf], np.nan)

    # Compute realized volatility and rolling liquidity stats per stock.
    ret_sq = merged["ret_1m"] * merged["ret_1m"]
    merged["rv_10"] = np.sqrt(ret_sq.groupby(merged["StockCode"]).rolling(window=10, min_periods=10).sum().reset_index(level=0, drop=True))
    merged["rv_20"] = np.sqrt(ret_sq.groupby(merged["StockCode"]).rolling(window=20, min_periods=20).sum().reset_index(level=0, drop=True))
    merged["vol_roll_mean_20"] = merged.groupby("StockCode")["Vol"].rolling(window=20, min_periods=20).mean().reset_index(level=0, drop=True)
    merged["vol_roll_std_20"] = merged.groupby("StockCode")["Vol"].rolling(window=20, min_periods=20).std().reset_index(level=0, drop=True)
    merged["amt_roll_mean_20"] = merged.groupby("StockCode")["Amount"].rolling(window=20, min_periods=20).mean().reset_index(level=0, drop=True)
    merged["amt_roll_std_20"] = merged.groupby("StockCode")["Amount"].rolling(window=20, min_periods=20).std().reset_index(level=0, drop=True)

    # Compute index-relative factors using same-minute cross-section only.
    merged["w_ret_1m"] = merged["weight"] * merged["ret_1m"]
    basket = merged.groupby("DateTime", sort=True).agg(
        basket_ret_1m=("w_ret_1m", "sum"),
        basket_weight=("weight", "sum"),
    )
    basket["basket_ret_1m"] = basket["basket_ret_1m"] / basket["basket_weight"]
    merged = merged.merge(basket.loc[:, ["basket_ret_1m"]].reset_index(), on="DateTime", how="left")
    merged["ret_1m_rel_basket"] = merged["ret_1m"] - merged["basket_ret_1m"]

    # Compute cross-sectional rank factors within each minute.
    merged["rank_ret_1m"] = merged.groupby("DateTime", sort=False)["ret_1m"].rank(pct=True)
    merged["rank_vol_chg_1m"] = merged.groupby("DateTime", sort=False)["vol_chg_1m"].rank(pct=True)
    merged["rank_weight"] = merged.groupby("DateTime", sort=False)["weight"].rank(pct=True)

    # Compute stock forward return label aligned to time t.
    close_future = merged.groupby("StockCode")["Close"].shift(-horizon_minutes)
    merged["label_stock_10m"] = close_future / merged["Close"].astype(float) - 1.0

    # Pack into the standardized panel schema required by AGENTS.md.
    out = merged.loc[
        :,
        [
            "Date",
            "DateTime",
            "StockCode",
            "weight",
            "label_stock_10m",
            "MinuteIndex",
            "ret_1m",
            "ret_5m",
            "ret_10m",
            "ret_30m",
            "vol_chg_1m",
            "amt_chg_1m",
            "rv_10",
            "rv_20",
            "vol_roll_mean_20",
            "vol_roll_std_20",
            "amt_roll_mean_20",
            "amt_roll_std_20",
            "basket_ret_1m",
            "ret_1m_rel_basket",
            "rank_ret_1m",
            "rank_vol_chg_1m",
            "rank_weight",
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
    cache_root: str,
) -> pd.DataFrame:
    """Load cached stock panel day or build and cache it."""

    # Decide cache file path.
    ensure_dir(cache_root)
    cache_path = os.path.join(cache_root, f"{date}.parquet")

    # Load from cache if available.
    if os.path.exists(cache_path):
        day = pd.read_parquet(cache_path)
        day["datetime"] = pd.to_datetime(day["datetime"]).astype("datetime64[us]")
        return day

    # Build the day panel and cache it.
    day = build_stock_panel_day(
        date=date,
        weights=weights,
        stock1m_root=stock1m_root,
        horizon_minutes=horizon_minutes,
    )
    day.to_parquet(cache_path, index=False)
    return day


def write_stock_panel_parquet(
    dates: list[int],
    weights: pd.DataFrame,
    stock1m_root: str,
    horizon_minutes: int,
    cache_root: str,
    out_path: str,
) -> dict:
    """Write stock_panel.parquet by streaming day panels into one parquet file."""

    # Create the output parent directory.
    ensure_dir(os.path.dirname(out_path))

    # Remove existing output to avoid mixing partial runs.
    if os.path.exists(out_path):
        os.remove(out_path)

    # Stream day panels to one parquet file to avoid holding the full dataset in memory.
    writer: pq.ParquetWriter | None = None
    rows_written = 0
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
            cache_root=cache_root,
        )
        day = day.loc[day["split"].isin(["train", "test"])].copy()

        # Drop rows with truncated labels due to horizon.
        day = day.dropna(subset=["label_stock_10m"])

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
    feature_cols = [
        "MinuteIndex",
        "ret_1m",
        "ret_5m",
        "ret_10m",
        "ret_30m",
        "vol_chg_1m",
        "amt_chg_1m",
        "rv_10",
        "rv_20",
        "vol_roll_mean_20",
        "vol_roll_std_20",
        "amt_roll_mean_20",
        "amt_roll_std_20",
        "basket_ret_1m",
        "ret_1m_rel_basket",
        "rank_ret_1m",
        "rank_vol_chg_1m",
        "rank_weight",
        "weight",
    ]
    return {"rows_written": rows_written, "feature_cols": feature_cols}
