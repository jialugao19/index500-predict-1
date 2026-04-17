import concurrent.futures as cf
import os
import pathlib
import time
from functools import lru_cache

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from features.builders.stock import build_stock_feature_panel_day, get_stock_factor_history_plan
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


def _mark_invalid_stock_bars(day: pd.DataFrame, wipe_invalid_values: bool) -> pd.DataFrame:
    """Mark invalid stock minute bars and optionally replace unusable OHLCVA values with NaN."""

    # Build one-step returns so limit-like bars can be identified.
    out = day.copy()
    grp = out.groupby("StockCode", sort=False)
    raw_ret_1 = out["Close"].astype(float) / grp["Close"].shift(1).astype(float) - 1.0

    # Combine hard invalid conditions into one explicit mask.
    limit_like = (out["High"].astype(float) == out["Low"].astype(float)) & np.isfinite(raw_ret_1.to_numpy(dtype=float)) & (
        np.abs(raw_ret_1.to_numpy(dtype=float)) >= 0.095
    )
    invalid_bar = (
        (out["Vol"].astype(float) <= 0.0)
        | (out["Amount"].astype(float) <= 0.0)
        | (~np.isfinite(out["Open"].astype(float)))
        | (~np.isfinite(out["High"].astype(float)))
        | (~np.isfinite(out["Low"].astype(float)))
        | (~np.isfinite(out["Close"].astype(float)))
        | limit_like
    )

    # Persist the invalid flag and optionally wipe broken base values.
    out["invalid_bar"] = invalid_bar.astype(int)
    if bool(wipe_invalid_values):
        out.loc[invalid_bar, ["Open", "High", "Low", "Close", "Vol", "Amount"]] = np.nan
    return out


@lru_cache(maxsize=None)
def list_available_stock_dates(stock1m_root: str) -> list[int]:
    """List available stock minute dates from the raw day files."""

    # Walk year directories once and parse all feather filenames into ints.
    dates: list[int] = []
    root = pathlib.Path(stock1m_root)
    for year_dir in sorted(root.iterdir()):
        if not year_dir.is_dir():
            continue
        for feather_path in sorted(year_dir.glob("*.feather")):
            dates.append(int(feather_path.stem))
    return dates


def get_previous_stock_dates(stock1m_root: str, date: int, history_days: int) -> list[int]:
    """Get previous trading dates needed for stock-factor warmup."""

    # Resolve the current date position inside the full stock-date calendar.
    all_dates = list_available_stock_dates(stock1m_root=stock1m_root)
    date_array = np.asarray(all_dates, dtype=np.int64)
    current_pos = int(np.searchsorted(date_array, int(date)))
    if current_pos >= len(date_array) or int(date_array[current_pos]) != int(date):
        raise ValueError(f"date {date} not found under stock1m_root={stock1m_root}")

    # Slice the requested number of prior trading dates.
    start_pos = max(0, current_pos - int(history_days))
    return [int(x) for x in date_array[start_pos:current_pos].tolist()]


def build_stock_history_panel(
    date: int,
    constituents: np.ndarray,
    stock1m_root: str,
    history_days: int,
    history_bars: int,
) -> pd.DataFrame:
    """Build a compact cross-day history panel for long-window stock factors."""

    # Return an empty history frame when the factor set does not require warmup.
    out_cols = ["date", "datetime", "stock_code", "MinuteIndex", "Open", "High", "Low", "Close", "Vol", "Amount", "invalid_bar"]
    if int(history_days) <= 0 or int(history_bars) <= 0:
        return pd.DataFrame(columns=out_cols)

    # Load the required previous trading days for the current constituent universe.
    prev_dates = get_previous_stock_dates(stock1m_root=stock1m_root, date=int(date), history_days=int(history_days))
    pieces: list[pd.DataFrame] = []
    for prev_date in prev_dates:
        history_day = load_stock_minute_bars(stock1m_root=stock1m_root, date=int(prev_date), constituents=constituents)
        history_day = _mark_invalid_stock_bars(day=history_day, wipe_invalid_values=False)
        history_day = history_day.loc[
            :,
            ["Date", "DateTime", "StockCode", "MinuteIndex", "Open", "High", "Low", "Close", "Vol", "Amount", "invalid_bar"],
        ].copy()
        history_day = history_day.rename(columns={"Date": "date", "DateTime": "datetime", "StockCode": "stock_code"})
        history_day["datetime"] = pd.to_datetime(history_day["datetime"]).astype("datetime64[us]")
        pieces.append(history_day)

    # Keep only the tail rows per stock that are required by the rewrite plan.
    if len(pieces) == 0:
        return pd.DataFrame(columns=out_cols)
    history = pd.concat(pieces, axis=0, ignore_index=True)
    history = history.sort_values(["stock_code", "datetime"], ascending=[True, True])
    history = history.groupby("stock_code", sort=False, group_keys=False).tail(int(history_bars)).copy()
    history = history.sort_values(["datetime", "stock_code"], ascending=[True, True]).reset_index(drop=True)
    return history.loc[:, out_cols].copy()


def build_stock_panel_day(
    date: int,
    weights: pd.DataFrame,
    stock1m_root: str,
    horizon_minutes: int,
) -> pd.DataFrame:
    """Build one-day stock-level panel with features, weights, and label."""

    # Build the clean base panel first and then delegate factor computation to the feature builder.
    base = build_stock_base_panel_day(date=date, weights=weights, stock1m_root=stock1m_root, horizon_minutes=horizon_minutes)
    history_plan = get_stock_factor_history_plan(
        specs_root=os.path.join(os.path.dirname(os.path.abspath(__file__)), "features", "specs"),
        factor_set_name="stock_all",
    )
    history = build_stock_history_panel(
        date=int(date),
        constituents=base["stock_code"].astype(int).unique(),
        stock1m_root=stock1m_root,
        history_days=int(history_plan["history_days"]),
        history_bars=int(history_plan["history_bars"]),
    )
    out = build_stock_feature_panel_day(
        base_panel=base,
        specs_root=os.path.join(os.path.dirname(os.path.abspath(__file__)), "features", "specs"),
        factor_set_name="stock_all",
        history_panel=history,
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
    stock_day = _mark_invalid_stock_bars(day=stock_day, wipe_invalid_values=True)

    # Join weights so each stock row has its index weight.
    merged = stock_day.merge(day_weights, left_on="StockCode", right_on="con_int", how="inner")
    merged = merged.drop(columns=["con_int"]).rename(columns={"weight_frac": "weight"})

    # Compute stock forward return label aligned to time t.
    grp = merged.groupby("StockCode", sort=False)
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
        # Read the cached feature panel with stable datetime dtype.
        day = pd.read_parquet(cache_path)
        day["datetime"] = pd.to_datetime(day["datetime"]).astype("datetime64[us]")

        # Backfill Amount for legacy caches so amount-based basket weights can be computed.
        if "Amount" not in day.columns:
            base = load_or_build_stock_base_panel_day(
                date=int(date),
                weights=weights,
                stock1m_root=stock1m_root,
                horizon_minutes=int(horizon_minutes),
                base_cache_root=base_cache_root,
            )
            base = base.loc[:, ["date", "datetime", "stock_code", "MinuteIndex", "Amount"]].copy()
            base["datetime"] = pd.to_datetime(base["datetime"]).astype("datetime64[us]")
            day = day.merge(base, on=["date", "datetime", "stock_code", "MinuteIndex"], how="left")
        return day

    # Load or build the base panel under its own cache.
    base = load_or_build_stock_base_panel_day(
        date=date,
        weights=weights,
        stock1m_root=stock1m_root,
        horizon_minutes=horizon_minutes,
        base_cache_root=base_cache_root,
    )
    history_plan = get_stock_factor_history_plan(specs_root=specs_root, factor_set_name=factor_set_name)
    history = build_stock_history_panel(
        date=int(date),
        constituents=base["stock_code"].astype(int).unique(),
        stock1m_root=stock1m_root,
        history_days=int(history_plan["history_days"]),
        history_bars=int(history_plan["history_bars"]),
    )

    # Build the feature panel and attach split for downstream filters.
    day = build_stock_feature_panel_day(
        base_panel=base,
        specs_root=specs_root,
        factor_set_name=factor_set_name,
        history_panel=history,
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


def get_stock_feature_cols(specs_root: str, factor_set_name: str) -> list[str]:
    """Get the output feature columns for one stock factor set."""

    # Resolve the selected factor set from the registry.
    registry = load_registry(specs_root=specs_root)
    factor_set = registry.factor_sets[str(factor_set_name)]

    # Append the shared time features used by the training pipeline.
    return list(factor_set.factors) + ["minute_of_day", "is_open_30min", "is_close_30min"]


def _prebuild_stock_feature_cache_task(task: tuple) -> dict:
    """Build or load one stock feature-cache day for parallel prebuild."""

    # Unpack the task payload into explicit local names.
    (
        date,
        weight_path,
        stock1m_root,
        horizon_minutes,
        base_cache_root,
        feature_cache_root,
        specs_root,
        factor_set_name,
    ) = task

    # Check whether the day cache already exists before the build call.
    cache_path = os.path.join(feature_cache_root, f"{int(date)}.parquet")
    existed = os.path.exists(cache_path)

    # Load weights locally inside the worker and build or load the day cache.
    weights = load_index_weights(weight_path=weight_path)
    day = load_or_build_stock_panel_day(
        date=int(date),
        weights=weights,
        stock1m_root=stock1m_root,
        horizon_minutes=int(horizon_minutes),
        base_cache_root=base_cache_root,
        feature_cache_root=feature_cache_root,
        specs_root=specs_root,
        factor_set_name=factor_set_name,
    )

    # Return compact progress metadata to the caller.
    return {"date": int(date), "cache_hit": int(existed), "rows": int(len(day))}


def prebuild_stock_feature_cache(
    dates: list[int],
    weight_path: str,
    stock1m_root: str,
    horizon_minutes: int,
    base_cache_root: str,
    feature_cache_root: str,
    specs_root: str,
    factor_set_name: str,
    n_jobs: int,
) -> dict:
    """Prebuild stock feature-cache days before training starts."""

    # Materialize the task payloads once for either serial or parallel execution.
    ensure_dir(feature_cache_root)
    tasks = [
        (
            int(date),
            weight_path,
            stock1m_root,
            int(horizon_minutes),
            base_cache_root,
            feature_cache_root,
            specs_root,
            factor_set_name,
        )
        for date in dates
    ]

    # Run serially when the caller requests a single worker.
    rows: list[dict] = []
    start_time = time.time()
    if int(n_jobs) == 1:
        for idx, task in enumerate(tasks, start=1):
            row = _prebuild_stock_feature_cache_task(task=task)
            rows.append(row)
            if int(idx) % 25 == 0:
                elapsed = time.time() - start_time
                print(f"[INFO] prebuild_cache progress day={int(idx)}/{len(tasks)} elapsed={elapsed:.1f}s.")
    else:
        # Run with process-level parallelism because factor evaluation is CPU-heavy.
        with cf.ProcessPoolExecutor(max_workers=int(n_jobs)) as executor:
            futures = [executor.submit(_prebuild_stock_feature_cache_task, task) for task in tasks]
            for idx, future in enumerate(cf.as_completed(futures), start=1):
                row = future.result()
                rows.append(row)
                if int(idx) % 25 == 0:
                    elapsed = time.time() - start_time
                    print(f"[INFO] prebuild_cache progress day={int(idx)}/{len(tasks)} elapsed={elapsed:.1f}s.")

    # Summarize the prebuild pass for logging and manifests.
    rows = sorted(rows, key=lambda row: int(row["date"]))
    cache_hits = int(sum([int(row["cache_hit"]) for row in rows]))
    return {"days": int(len(rows)), "cache_hits": cache_hits, "cache_misses": int(len(rows) - cache_hits)}


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

    # Load factor-set output columns once so coverage diagnostics stay consistent.
    feature_cols = get_stock_feature_cols(specs_root=specs_root, factor_set_name=factor_set_name)

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
