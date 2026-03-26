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

    # Load weights as-of date and constituent stock bars.
    day_weights = get_constituent_weights_for_date(weights=weights, date=date)
    constituents = day_weights["con_int"].to_numpy()
    stock_day = load_stock_minute_bars(stock1m_root=stock1m_root, date=date, constituents=constituents)

    # Join weights so each stock row has its index weight.
    merged = stock_day.merge(day_weights, left_on="StockCode", right_on="con_int", how="inner")
    merged = merged.drop(columns=["con_int"]).rename(columns={"weight_frac": "weight"})

    # Mark suspension-like abnormal minutes and enforce "window contains invalid -> NaN" rules.
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
    merged.loc[invalid_bar, ["Open", "High", "Low", "Close", "Vol", "Amount"]] = np.nan

    # Compute return-momentum features using only t and past information.
    merged["ret_1"] = merged["Close"] / grp["Close"].shift(1) - 1.0
    merged["ret_5"] = merged["Close"] / grp["Close"].shift(5) - 1.0
    merged["ret_10"] = merged["Close"] / grp["Close"].shift(10) - 1.0
    merged["ret_30"] = merged["Close"] / grp["Close"].shift(30) - 1.0
    merged["ret_60"] = merged["Close"] / grp["Close"].shift(60) - 1.0
    open_px = grp["Open"].transform("first").astype(float)
    merged["ret_open"] = merged["Close"].astype(float) / open_px - 1.0
    merged = merged.replace([np.inf, -np.inf], np.nan)

    # Compute volume features using rolling windows with invalid-bar gating.
    vol_roll_mean_5 = grp["Vol"].rolling(window=5, min_periods=5).mean().reset_index(level=0, drop=True)
    vol_roll_mean_30 = grp["Vol"].rolling(window=30, min_periods=30).mean().reset_index(level=0, drop=True)
    vol_roll_sum_5 = grp["Vol"].rolling(window=5, min_periods=5).sum().reset_index(level=0, drop=True)
    merged["vol_change_1"] = merged["Vol"].astype(float) / vol_roll_mean_5.astype(float)
    merged["vol_change_5"] = vol_roll_sum_5.astype(float) / (vol_roll_mean_30.astype(float) * 5.0)
    merged = merged.replace([np.inf, -np.inf], np.nan)

    # Compute price-volume relation features using only past information.
    cum_vol = grp["Vol"].cumsum()
    cum_amt = grp["Amount"].cumsum()
    vwap = cum_amt.astype(float) / cum_vol.astype(float)
    merged["vwap_dev"] = (merged["Close"].astype(float) - vwap.astype(float)) / vwap.astype(float)
    high_so_far = grp["High"].cummax().astype(float)
    low_so_far = grp["Low"].cummin().astype(float)
    merged["price_high_dev"] = (merged["Close"].astype(float) - high_so_far) / high_so_far
    merged["price_low_dev"] = (merged["Close"].astype(float) - low_so_far) / low_so_far
    corr_window = 30
    amount_ret_corr = grp.apply(
        lambda part: part["Amount"].rolling(window=corr_window, min_periods=corr_window).corr(part["ret_1"]),
        include_groups=False,
    )
    amount_ret_corr = amount_ret_corr.reset_index(level=0, drop=True)
    merged["amount_ret_corr"] = amount_ret_corr.astype(float)
    merged = merged.replace([np.inf, -np.inf], np.nan)

    # Compute volatility features from rolling return dispersion and OHLC range.
    merged["volatility_10"] = grp["ret_1"].rolling(window=10, min_periods=10).std().reset_index(level=0, drop=True).to_numpy(dtype=float)
    merged["volatility_30"] = grp["ret_1"].rolling(window=30, min_periods=30).std().reset_index(level=0, drop=True).to_numpy(dtype=float)
    range_window = 30
    win_high = grp["High"].rolling(window=range_window, min_periods=range_window).max().reset_index(level=0, drop=True)
    win_low = grp["Low"].rolling(window=range_window, min_periods=range_window).min().reset_index(level=0, drop=True)
    open_start = grp["Open"].shift(range_window - 1).astype(float)
    merged["high_low_range"] = (win_high.astype(float) - win_low.astype(float)) / open_start
    merged = merged.replace([np.inf, -np.inf], np.nan)

    # Enforce the "window contains suspension bar -> NaN" rule for window-based features.
    invalid = invalid_bar.astype(int)
    invalid_roll_2 = invalid.groupby(merged["StockCode"]).rolling(window=2, min_periods=2).sum().reset_index(level=0, drop=True)
    invalid_roll_6 = invalid.groupby(merged["StockCode"]).rolling(window=6, min_periods=6).sum().reset_index(level=0, drop=True)
    invalid_roll_11 = invalid.groupby(merged["StockCode"]).rolling(window=11, min_periods=11).sum().reset_index(level=0, drop=True)
    invalid_roll_31 = invalid.groupby(merged["StockCode"]).rolling(window=31, min_periods=31).sum().reset_index(level=0, drop=True)
    invalid_roll_61 = invalid.groupby(merged["StockCode"]).rolling(window=61, min_periods=61).sum().reset_index(level=0, drop=True)
    merged.loc[invalid_roll_2.to_numpy() > 0, ["ret_1", "vol_change_1"]] = np.nan
    merged.loc[invalid_roll_6.to_numpy() > 0, ["ret_5", "vol_change_5"]] = np.nan
    merged.loc[invalid_roll_11.to_numpy() > 0, ["ret_10", "volatility_10"]] = np.nan
    merged.loc[invalid_roll_31.to_numpy() > 0, ["ret_30", "volatility_30", "high_low_range", "amount_ret_corr"]] = np.nan
    merged.loc[invalid_roll_61.to_numpy() > 0, ["ret_60"]] = np.nan
    invalid_cum = invalid.groupby(merged["StockCode"]).cumsum()
    merged.loc[invalid_cum.to_numpy() > 0, ["ret_open", "vwap_dev", "price_high_dev", "price_low_dev"]] = np.nan

    # Compute time features without cross-sectional normalization.
    merged["minute_of_day"] = merged["MinuteIndex"].astype(int)
    merged["is_open_30min"] = (merged["MinuteIndex"].astype(int) < 30).astype(int)
    merged["is_close_30min"] = (merged["MinuteIndex"].astype(int) >= 211).astype(int)

    # Apply cross-sectional winsorize + rank normalization per minute for non-time features.
    xs_cols = [
        "ret_1",
        "ret_5",
        "ret_10",
        "ret_30",
        "ret_60",
        "ret_open",
        "vol_change_1",
        "vol_change_5",
        "vwap_dev",
        "price_high_dev",
        "price_low_dev",
        "amount_ret_corr",
        "volatility_10",
        "volatility_30",
        "high_low_range",
    ]
    bounds = merged.groupby("DateTime", sort=False)[xs_cols].quantile([0.01, 0.99]).reset_index()
    bounds = bounds.rename(columns={"level_1": "q"})
    lower = bounds.loc[bounds["q"] == 0.01].drop(columns=["q"]).set_index("DateTime")
    upper = bounds.loc[bounds["q"] == 0.99].drop(columns=["q"]).set_index("DateTime")
    lower = lower.rename(columns={c: f"{c}__lo" for c in xs_cols})
    upper = upper.rename(columns={c: f"{c}__hi" for c in xs_cols})
    merged = merged.merge(lower.reset_index(), on="DateTime", how="left")
    merged = merged.merge(upper.reset_index(), on="DateTime", how="left")
    for col in xs_cols:
        merged[col] = merged[col].astype(float).clip(lower=merged[f"{col}__lo"].astype(float), upper=merged[f"{col}__hi"].astype(float))
        merged[col] = merged.groupby("DateTime", sort=False)[col].rank(pct=True)
    merged = merged.drop(columns=[f"{c}__lo" for c in xs_cols] + [f"{c}__hi" for c in xs_cols])

    # Compute stock forward return label aligned to time t.
    close_future = grp["Close"].shift(-horizon_minutes)
    merged["label_stock_10m"] = close_future.astype(float) / merged["Close"].astype(float) - 1.0

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
            "ret_1",
            "ret_5",
            "ret_10",
            "ret_30",
            "ret_60",
            "ret_open",
            "vol_change_1",
            "vol_change_5",
            "vwap_dev",
            "price_high_dev",
            "price_low_dev",
            "amount_ret_corr",
            "volatility_10",
            "volatility_30",
            "high_low_range",
            "minute_of_day",
            "is_open_30min",
            "is_close_30min",
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
        "ret_1",
        "ret_5",
        "ret_10",
        "ret_30",
        "ret_60",
        "ret_open",
        "vol_change_1",
        "vol_change_5",
        "vwap_dev",
        "price_high_dev",
        "price_low_dev",
        "amount_ret_corr",
        "volatility_10",
        "volatility_30",
        "high_low_range",
        "minute_of_day",
        "is_open_30min",
        "is_close_30min",
    ]
    return {"rows_written": rows_written, "feature_cols": feature_cols}
