import datetime as dt
import os
import pathlib
import random
import time

import numpy as np
import pandas as pd

from features.manifest import build_factor_set_manifest
from features.registry import cached_load_registry
from eval.metrics import (
    compute_error_summary,
    compute_metrics,
    compute_ols_summary,
    safe_corr,
    safe_spearman,
    summarize_error_by_minute_bucket,
)
from eval.feature_importance import compute_feature_ic_table
from eval.stock_level import (
    compute_stock_sufficient_stats,
    compute_stock_daily_ic_table,
    compute_panel_ic_by_minute,
    finalize_stock_metrics_from_sufficient_stats,
    summarize_panel_ic_by_minute_bucket,
    summarize_panel_ic_daily,
)
from eval.walk_forward import run_walk_forward_validation
from features.builders.etf import build_etf_features_day
from features.etf import compute_label_from_close
from models.lgbm import fit_lgbm_model
from models.zscore import (
    fit_frame_zscore_stats,
    fit_series_zscore_stats,
    inverse_series_zscore,
    transform_frame_zscore,
    transform_series_zscore,
)
from models.xgb import fit_xgb_model
from eval.plots import (
    plot_baseline_comparison,
    plot_cumulative_metric,
    plot_daily_timeseries,
    plot_feature_importance,
    plot_monthly_timeseries,
    plot_prediction_scatter,
    plot_prediction_timeseries,
    plot_histogram,
)
from eval.writers import (
    compute_coverage_audit_tables,
    write_data_audit_md,
    write_bottom_up_report_md,
    write_report_md,
    write_report_tex,
    write_summary_md,
    write_yaml,
)
from basket_aggregator import aggregate_day_to_basket
from stock_panel_loader import (
    get_stock_feature_cols,
    load_index_weights,
    load_or_build_stock_panel_day,
    prebuild_stock_feature_cache,
)


def set_global_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""

    # Set Python-level RNG seeds.
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path: str) -> None:
    """Ensure a directory exists."""

    # Create directory recursively.
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)


def list_available_dates_from_etf1m_dir(etf1m_dir: str) -> list[int]:
    """List all available dates from the etf1m directory."""

    # Collect all feather filenames as candidate dates.
    all_dates: list[int] = []
    for year_name in sorted(os.listdir(etf1m_dir)):
        # Skip non-year folders.
        if not year_name.isdigit():
            continue
        year_dir = os.path.join(etf1m_dir, year_name)
        for file_name in sorted(os.listdir(year_dir)):
            # Parse yyyymmdd from filenames like 20210104.feather.
            if not file_name.endswith(".feather"):
                continue
            date_str = file_name.replace(".feather", "")
            if not date_str.isdigit():
                continue
            all_dates.append(int(date_str))

    # Return sorted unique dates.
    all_dates_sorted = sorted(set(all_dates))
    return all_dates_sorted


def filter_dates_by_range(all_dates: list[int], start_date: int, end_date: int) -> list[int]:
    """Filter dates by inclusive range."""

    # Keep dates within the required range.
    kept_dates = [date for date in all_dates if start_date <= date <= end_date]
    return kept_dates


def load_etf_minute_bars(etf1m_root: str, date: int, etf_code_int: int) -> pd.DataFrame:
    """Load one-day ETF 1m bars for a specific ETF."""

    # Build the path and load the day file.
    year = str(date)[:4]
    file_path = os.path.join(etf1m_root, year, f"{date}.feather")
    day = pd.read_feather(file_path)

    # Filter to the target ETF and keep necessary columns.
    day = day.loc[day["StockCode"] == etf_code_int, ["DateTime", "Close", "Vol", "Amount", "Date", "MinuteIndex"]].copy()
    day = day.sort_values("DateTime", ascending=True)
    return day


def date_to_split(date: int) -> str:
    """Map date to split label using AGENTS.md time partition."""

    # Apply the required split rules.
    if 20210101 <= date <= 20231231:
        return "train"
    if date >= 20240201:
        return "test"
    return "ignore"


def choose_train_test_dates(
    all_dates: list[int],
    train_start: int,
    train_end: int,
    test_start: int,
    max_train_days: int,
    max_test_days: int,
) -> tuple[list[int], list[int]]:
    """Choose a bounded set of train/test dates to keep runs fast and reproducible."""

    # Slice all available dates into calendar train/test buckets.
    train_dates = [d for d in all_dates if int(train_start) <= int(d) <= int(train_end)]
    test_dates = [d for d in all_dates if int(d) >= int(test_start)]

    # Keep the most recent train days and the earliest test days.
    train_kept = train_dates[-int(max_train_days) :]
    test_kept = test_dates[: int(max_test_days)]
    return train_kept, test_kept


def pick_best_model_by_metric(etf_overall: dict, split: str, metric_name: str) -> str:
    """Pick the best model name by a specific overall metric on a split."""

    # Rank models by the metric and return the argmax.
    scored: list[tuple[str, float]] = []
    for model_name in sorted(etf_overall.keys()):
        row = etf_overall[model_name].get(str(split), {})
        scored.append((str(model_name), float(row.get(str(metric_name), float("nan")))))
    scored_sorted = sorted(scored, key=lambda x: (np.nan_to_num(x[1], nan=-1e9)), reverse=True)
    return str(scored_sorted[0][0])


def compute_raw_vs_basis_delta_test(etf_overall: dict, pairs: list[tuple[str, str]]) -> list[dict]:
    """Compute test IC/RankIC deltas between raw and basis-corrected models."""

    # Build a small list of per-pair deltas for reporting.
    rows: list[dict] = []
    for raw_name, basis_name in pairs:
        raw_row = etf_overall[str(raw_name)]["test"]
        basis_row = etf_overall[str(basis_name)]["test"]
        rows.append(
            {
                "raw_model": str(raw_name),
                "basis_model": str(basis_name),
                "ic_delta": float(basis_row["ic"]) - float(raw_row["ic"]),
                "rank_ic_delta": float(basis_row["rank_ic"]) - float(raw_row["rank_ic"]),
            }
        )
    return rows


def make_basket_prediction_table(
    basket: pd.DataFrame,
    model_name: str,
) -> pd.DataFrame:
    """Build a standardized basket-level prediction table for metrics."""

    # Standardize output columns required by compute_metrics.
    out = pd.DataFrame(
        {
            "datetime": pd.to_datetime(basket["datetime"]).astype("datetime64[ns]"),
            "date": basket["date"].astype(int),
            "pred": basket["basket_pred"].astype(float),
            "label": basket["basket_label"].astype(float),
            "split": basket["split"].astype(str),
            "model_name": model_name,
        }
    )
    return out


def make_etf_prediction_table(
    frame: pd.DataFrame,
    model_name: str,
    pred: np.ndarray,
    label_col: str,
) -> pd.DataFrame:
    """Build a standardized ETF-level prediction table for metrics."""

    # Standardize output columns required by compute_metrics.
    out = pd.DataFrame(
        {
            "datetime": pd.to_datetime(frame["datetime"]).astype("datetime64[ns]"),
            "date": frame["date"].astype(int),
            "pred": pred.astype(float),
            "label": frame[label_col].astype(float),
            "split": frame["split"].astype(str),
            "model_name": model_name,
        }
    )
    return out


def collect_stock_training_sample(
    dates: list[int],
    weights: pd.DataFrame,
    stock1m_root: str,
    horizon_minutes: int,
    base_cache_root: str,
    feature_cache_root: str,
    specs_root: str,
    factor_set_name: str,
    feature_cols: list[str],
    max_rows: int,
    sample_mod: int,
) -> pd.DataFrame:
    """Collect a deterministic sample of stock panel rows for model training."""

    # Iterate training dates and collect a stable hash-based subsample.
    pieces: list[pd.DataFrame] = []
    for date in dates:
        # Load cached day panel and keep train split only.
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
        day = day.loc[day["split"] == "train"].copy()
        day = day.dropna(subset=["label_stock_10m"])

        # Build a deterministic hash-based sample mask for reproducibility.
        key = (
            day["date"].astype(np.int64) * 1_000_003
            + day["stock_code"].astype(np.int64) * 10_007
            + day["MinuteIndex"].astype(np.int64) * 10_009
        )
        mask = (key % int(sample_mod)).astype(int) == 0
        sampled = day.loc[mask, ["date", "stock_code", "MinuteIndex", "label_stock_10m"] + feature_cols].copy()
        pieces.append(sampled)

    # Concatenate samples into a single training frame.
    sample = pd.concat(pieces, axis=0, ignore_index=True)
    assert int(len(sample)) <= int(max_rows)
    return sample


def build_etf_minute_dataset(
    dates: list[int],
    etf1m_root: str,
    etf_code_int: int,
    horizon_minutes: int,
    specs_root: str,
    factor_set_name: str,
) -> pd.DataFrame:
    """Build ETF minute feature/label dataset for basis modeling and evaluation."""

    # Build per-day ETF frames to keep memory bounded.
    frames: list[pd.DataFrame] = []
    for date in dates:
        # Load ETF bars and compute ETF features and label.
        etf_day = load_etf_minute_bars(etf1m_root=etf1m_root, date=int(date), etf_code_int=etf_code_int)
        etf_features = build_etf_features_day(etf_day=etf_day, specs_root=specs_root, factor_set_name=factor_set_name)
        label = compute_label_from_close(etf_day=etf_day, horizon_minutes=horizon_minutes)

        # Assemble a day frame with stable columns for joins.
        day = etf_features.copy()
        day = day.rename(columns={"DateTime": "datetime", "Date": "date"})
        day["MinuteIndex"] = etf_day["MinuteIndex"].astype(int).to_numpy()
        day["label_etf_10m"] = label.astype(float).to_numpy()
        day["split"] = date_to_split(date=int(date))
        frames.append(day)

    # Concatenate into one dataset and drop horizon-truncated labels.
    full = pd.concat(frames, axis=0, ignore_index=True)
    full = full.loc[full["split"].isin(["train", "test"])].copy()
    full = full.dropna(subset=["label_etf_10m"])
    return full


def sample_spot_checks(
    etf1m_root: str,
    date: int,
    etf_code_int: int,
    num_checks: int,
    horizon_minutes: int,
    specs_root: str,
    factor_set_name: str,
) -> list[dict]:
    """Sample spot checks to demonstrate no look-ahead."""

    # Load the ETF day and compute label and features for inspection.
    etf_day = load_etf_minute_bars(etf1m_root=etf1m_root, date=date, etf_code_int=etf_code_int)
    features = build_etf_features_day(etf_day=etf_day, specs_root=specs_root, factor_set_name=factor_set_name)
    label = compute_label_from_close(etf_day=etf_day, horizon_minutes=horizon_minutes)

    # Pick deterministic indices away from the ends to allow t+10.
    indices = np.linspace(50, len(etf_day) - 50, num=num_checks, dtype=int)
    checks: list[dict] = []
    for idx in indices:
        # Extract t and t+10 information for explicit label verification.
        datetime_t = pd.to_datetime(etf_day.iloc[idx]["DateTime"])
        datetime_t_plus_10 = pd.to_datetime(etf_day.iloc[idx + 10]["DateTime"])
        close_t = float(etf_day.iloc[idx]["Close"])
        close_t_plus_10 = float(etf_day.iloc[idx + 10]["Close"])
        lbl = float(label.iloc[idx])
        ret_5m = float(features.iloc[idx]["ret_5m"])

        # Store into dict for markdown output.
        checks.append(
            {
                "date": int(date),
                "datetime_t": str(datetime_t),
                "close_t": close_t,
                "datetime_t_plus_10": str(datetime_t_plus_10),
                "close_t_plus_10": close_t_plus_10,
                "label": lbl,
                "ret_5m": ret_5m,
            }
        )
    return checks


def main() -> None:
    """Run the bottom-up synthesis pipeline for 510500."""

    # Define core constants from AGENTS.md.
    seed = 42
    etf_code_int = 510500
    label_horizon_minutes = 10
    train_start = 20210101
    train_end = 20231231
    test_start = 20240201
    test_end = 20251231

    # Define data locations.
    repo_root = os.path.dirname(os.path.abspath(__file__))
    weight_path = "/data/ashare/market/index_weight/000905.feather"
    etf1m_root = "/data/ashare/market/etf1m"
    stock1m_root = "/data/ashare/market/stock1m"
    factor_set_name = "stock_all"
    etf_factor_set_name = "etf_all"
    stock_cache_jobs = 8
    specs_root = os.path.join(repo_root, "features", "specs")
    stock_panel_base_cache_root = f"/data-cache/index500-predict/stock_base_days_v1__h{label_horizon_minutes}"
    stock_panel_feature_cache_root = f"/data-cache/index500-predict/stock_feature_days_v2__{factor_set_name}__h{label_horizon_minutes}"
    # Create output run directory under repo report folder grouped by month-day for easy browsing.
    now = dt.datetime.now()
    day_tag = now.strftime("%m%d")
    run_id = now.strftime("run_%Y%m%d_%H%M%S")
    report_day_dir = os.path.join(repo_root, "report", day_tag)
    report_dir = os.path.join(report_day_dir, run_id)
    ensure_dir(report_dir)

    # Set seeds for reproducibility.
    set_global_seed(seed=seed)

    # Discover available dates and keep only those needed for train/test.
    all_dates = list_available_dates_from_etf1m_dir(etf1m_dir=etf1m_root)
    train_dates = filter_dates_by_range(all_dates=all_dates, start_date=train_start, end_date=train_end)
    test_dates = filter_dates_by_range(all_dates=all_dates, start_date=test_start, end_date=test_end)

    # Use the full training date range as requested (2021-01-01 ~ 2023-12-31).
    train_dates = list(train_dates)
    dates_needed = list(train_dates) + list(test_dates)
    print(f"[INFO] Available dates={len(all_dates)}, using train_dates={len(train_dates)}, test_dates={len(test_dates)}.")

    # Load weights once for all dates.
    weights = load_index_weights(weight_path=weight_path)
    print(
        f"[INFO] Loaded weights rows={len(weights)}. trade_date_min={weights['trade_date'].min()} trade_date_max={weights['trade_date'].max()}."
    )

    # Stage 1: Prebuild stock feature caches before training and evaluation.
    cache_prebuild = prebuild_stock_feature_cache(
        dates=dates_needed,
        weight_path=weight_path,
        stock1m_root=stock1m_root,
        horizon_minutes=label_horizon_minutes,
        base_cache_root=stock_panel_base_cache_root,
        feature_cache_root=stock_panel_feature_cache_root,
        specs_root=specs_root,
        factor_set_name=factor_set_name,
        n_jobs=stock_cache_jobs,
    )
    stock_feature_cols = get_stock_feature_cols(specs_root=specs_root, factor_set_name=factor_set_name)
    print(
        f"[INFO] Stock cache prebuild days={int(cache_prebuild['days'])} hits={int(cache_prebuild['cache_hits'])} misses={int(cache_prebuild['cache_misses'])}."
    )

    # Write factor manifest for provenance and future factor-set driven training.
    registry = cached_load_registry(specs_root=specs_root)
    stock_manifest = build_factor_set_manifest(registry=registry, factor_set_name=factor_set_name)
    stock_manifest["cache_prebuild"] = cache_prebuild
    etf_manifest = build_factor_set_manifest(registry=registry, factor_set_name=etf_factor_set_name)
    write_yaml(path=os.path.join(report_dir, "feature_manifest.yaml"), obj={"stock": stock_manifest, "etf": etf_manifest})

    # Stage 2: Fit stock-level models on a deterministic subsample.
    train_sample = collect_stock_training_sample(
        dates=list(train_dates),
        weights=weights,
        stock1m_root=stock1m_root,
        horizon_minutes=label_horizon_minutes,
        base_cache_root=stock_panel_base_cache_root,
        feature_cache_root=stock_panel_feature_cache_root,
        specs_root=specs_root,
        factor_set_name=factor_set_name,
        feature_cols=stock_feature_cols,
        max_rows=2_000_000,
        sample_mod=50,
    )
    train_sample = train_sample.rename(columns={"label_stock_10m": "label"})
    train_sample["month"] = (train_sample["date"].astype(int) // 100).astype(int)
    val_month = int(train_sample["month"].max())
    val = train_sample.loc[train_sample["month"] == val_month].copy()
    train_fit = train_sample.loc[train_sample["month"] < val_month].copy()
    print(f"[INFO] Stock train_fit rows={len(train_fit)}, val rows={len(val)} (sampled).")

    # Stage 2: Fit feature/label z-score statistics on the true training fold only.
    stock_feature_stats = fit_frame_zscore_stats(frame=train_fit, columns=stock_feature_cols)
    stock_label_stats = fit_series_zscore_stats(series=train_fit["label"], name="label_stock_10m")
    train_fit = transform_frame_zscore(frame=train_fit, stats=stock_feature_stats)
    val = transform_frame_zscore(frame=val, stats=stock_feature_stats)
    train_fit["label"] = transform_series_zscore(series=train_fit["label"], stats=stock_label_stats)
    val["label"] = transform_series_zscore(series=val["label"], stats=stock_label_stats)

    # Stage 2: Train the required XGBoost + LightGBM unified panel models.
    stock_xgb_model = fit_xgb_model(train=train_fit, val=val, features=stock_feature_cols, seed=seed)
    stock_lgbm_model = fit_lgbm_model(train=train_fit, val=val, features=stock_feature_cols, seed=seed)

    # Stage 2: Export stock-model feature importance artifacts for the report.
    stock_xgb_importance = [
        {"feature": str(f), "importance": float(w)} for f, w in zip(stock_feature_cols, stock_xgb_model.feature_importances_)
    ]
    stock_lgbm_importance = [
        {"feature": str(f), "importance": float(w)} for f, w in zip(stock_feature_cols, stock_lgbm_model.feature_importances_)
    ]
    plot_feature_importance(
        importances=stock_xgb_importance,
        out_path=os.path.join(report_dir, "fig_stock_xgb_feature_importance.png"),
        top_k=30,
        title="Stock XGB Feature Importance",
    )
    plot_feature_importance(
        importances=stock_lgbm_importance,
        out_path=os.path.join(report_dir, "fig_stock_lgbm_feature_importance.png"),
        top_k=30,
        title="Stock LightGBM Feature Importance",
    )

    # Stage 2: Generate stock metrics (test) and basket predictions (train+test).
    minute_ic_tables: dict[str, list[pd.DataFrame]] = {k: [] for k in ["xgb", "lgbm"]}
    stock_ts_stats_tables: list[pd.DataFrame] = []
    stock_daily_ic_tables: list[pd.DataFrame] = []
    basket_rows: list[pd.DataFrame] = []
    for date in dates_needed:
        # Load cached day panel and keep train/test rows only.
        day = load_or_build_stock_panel_day(
            date=int(date),
            weights=weights,
            stock1m_root=stock1m_root,
            horizon_minutes=label_horizon_minutes,
            base_cache_root=stock_panel_base_cache_root,
            feature_cache_root=stock_panel_feature_cache_root,
            specs_root=specs_root,
            factor_set_name=factor_set_name,
        )
        day = day.loc[day["split"].isin(["train", "test"])].copy()
        day = day.dropna(subset=["label_stock_10m"])

        # Predict stock returns with the panel XGB model.
        day_features = transform_frame_zscore(frame=day.loc[:, stock_feature_cols], stats=stock_feature_stats)
        xgb_pred_z = stock_xgb_model.predict(day_features.to_numpy())
        lgbm_pred_z = stock_lgbm_model.predict(day_features.to_numpy())
        xgb_pred = inverse_series_zscore(values=xgb_pred_z, stats=stock_label_stats)
        lgbm_pred = inverse_series_zscore(values=lgbm_pred_z, stats=stock_label_stats)

        # Compute stock-level panel IC tables for the test split only.
        if str(day["split"].iloc[0]) == "test":
            minute_ic_tables["xgb"].append(compute_panel_ic_by_minute(day_panel=day, pred=xgb_pred, label_col="label_stock_10m"))
            minute_ic_tables["lgbm"].append(compute_panel_ic_by_minute(day_panel=day, pred=lgbm_pred, label_col="label_stock_10m"))
            stock_ts_stats_tables.append(compute_stock_sufficient_stats(day_panel=day, pred=xgb_pred, label_col="label_stock_10m"))

            # Compute per-stock within-day IC for robustness diagnostics.
            stock_daily_ic_tables.append(compute_stock_daily_ic_table(day_panel=day, pred=xgb_pred, label_col="label_stock_10m"))

        # Aggregate stock predictions into basket-level minute predictions.
        basket_xgb = aggregate_day_to_basket(stock_day=day, pred=xgb_pred, pred_col_name="pred_stock_xgb")
        basket_xgb["model_name"] = "basket_stock_xgb"
        basket_lgbm = aggregate_day_to_basket(stock_day=day, pred=lgbm_pred, pred_col_name="pred_stock_lgbm")
        basket_lgbm["model_name"] = "basket_stock_lgbm"
        basket_rows.append(pd.concat([basket_xgb, basket_lgbm], axis=0, ignore_index=True))

    # Stage 3: Evaluate Stock Alpha and write basket_pred.parquet (deliverable).
    stock_alpha_overall: dict[str, dict] = {}
    for model_name in ["xgb", "lgbm"]:
        # Summarize pooled minute IC and daily IC for each stock model.
        minute_ic = pd.concat(minute_ic_tables[model_name], axis=0, ignore_index=True)
        daily_ic = summarize_panel_ic_daily(minute_ic=minute_ic)
        daily_ic_mean = float(daily_ic["ic_mean"].mean())
        daily_rank_ic_mean = float(daily_ic["rank_ic_mean"].mean())
        daily_ic_std = float(daily_ic["ic_mean"].std())
        daily_rank_ic_std = float(daily_ic["rank_ic_mean"].std())
        stock_alpha_overall[str(model_name)] = {
            "panel_ic_test_mean": float(minute_ic["ic"].mean()),
            "panel_rank_ic_test_mean": float(minute_ic["rank_ic"].mean()),
            "daily_ic_test_mean": daily_ic_mean,
            "daily_rank_ic_test_mean": daily_rank_ic_mean,
            "daily_ic_test_std": daily_ic_std,
            "daily_rank_ic_test_std": daily_rank_ic_std,
            "daily_icir_test": float(daily_ic_mean / daily_ic_std),
            "daily_rank_icir_test": float(daily_rank_ic_mean / daily_rank_ic_std),
        }
    minute_ic_xgb = pd.concat(minute_ic_tables["xgb"], axis=0, ignore_index=True)
    bucket_ic = summarize_panel_ic_by_minute_bucket(minute_ic=minute_ic_xgb, bucket_size=30)
    bucket_ic["minute_bucket"] = bucket_ic["minute_bucket"].astype(int)
    stock_alpha_metrics = {
        "panel_ic_test_mean": float(stock_alpha_overall["xgb"]["panel_ic_test_mean"]),
        "panel_rank_ic_test_mean": float(stock_alpha_overall["xgb"]["panel_rank_ic_test_mean"]),
        "daily_ic_test_mean": float(stock_alpha_overall["xgb"]["daily_ic_test_mean"]),
        "daily_rank_ic_test_mean": float(stock_alpha_overall["xgb"]["daily_rank_ic_test_mean"]),
        "overall": stock_alpha_overall,
        "minute_bucket_ic_test": bucket_ic.to_dict(orient="records"),
        "feature_importance": {
            "xgb": stock_xgb_importance,
            "lgbm": stock_lgbm_importance,
            "fig_stock_xgb_feature_importance": "fig_stock_xgb_feature_importance.png",
            "fig_stock_lgbm_feature_importance": "fig_stock_lgbm_feature_importance.png",
        },
    }
    # Compute per-stock time-series metrics on the test split for stock-level diagnostics.
    stock_ts_stats = pd.concat(stock_ts_stats_tables, axis=0, ignore_index=True)
    stock_ts_stats_sum = stock_ts_stats.groupby("stock_code", sort=True).sum(numeric_only=True).reset_index()
    stock_ts_metrics = finalize_stock_metrics_from_sufficient_stats(stats=stock_ts_stats_sum)

    # Compute per-stock within-day IC averages as a robustness cross-check.
    stock_daily_ic = pd.concat(stock_daily_ic_tables, axis=0, ignore_index=True)
    stock_daily_summary = (
        stock_daily_ic.groupby("stock_code", sort=True)
        .agg(
            daily_ic_mean=("daily_ic", "mean"),
            daily_rank_ic_mean=("daily_rank_ic", "mean"),
            days=("daily_ic", "size"),
            daily_n_sum=("n", "sum"),
            daily_ic_neg_rate=("daily_ic", lambda s: float(np.mean(s.to_numpy(dtype=float) < 0.0))),
        )
        .reset_index()
    )

    # Merge minute-level and within-day summary tables to one per-stock evaluation table.
    stock_metrics = stock_ts_metrics.merge(stock_daily_summary, on="stock_code", how="left")

    # Tag stocks as good/bad/neutral using an IC t-stat threshold.
    ic_t_threshold = 2.0
    ic_abs_threshold = 0.05
    stock_metrics["verdict"] = "neutral"
    stock_metrics.loc[(stock_metrics["ic"] >= ic_abs_threshold) & (stock_metrics["ic_t"] >= ic_t_threshold), "verdict"] = "good"
    stock_metrics.loc[(stock_metrics["ic"] <= -ic_abs_threshold) & (stock_metrics["ic_t"] <= -ic_t_threshold), "verdict"] = "bad"

    # Write the per-stock table as a parquet artifact for detailed inspection.
    stock_metrics_path = os.path.join(report_dir, "stock_metrics_test.parquet")
    stock_metrics.to_parquet(stock_metrics_path, index=False)

    # Add a compact per-stock summary into metrics.yaml for the markdown report.
    ic_vals = stock_metrics["ic"].to_numpy(dtype=float)
    ic_vals = ic_vals[np.isfinite(ic_vals)]
    q05, q50, q95 = np.quantile(ic_vals, [0.05, 0.50, 0.95]) if len(ic_vals) else [float("nan")] * 3
    verdict_counts = stock_metrics["verdict"].value_counts(dropna=False).to_dict()
    stock_alpha_metrics["per_stock_test"] = {
        "ic_t_threshold": float(ic_t_threshold),
        "ic_abs_threshold": float(ic_abs_threshold),
        "n_stocks": int(len(stock_metrics)),
        "verdict_counts": {str(k): int(v) for k, v in verdict_counts.items()},
        "ic_quantiles": {"q05": float(q05), "q50": float(q50), "q95": float(q95)},
        "top_ic": stock_metrics.sort_values("ic", ascending=False)
        .head(10)
        .loc[:, ["stock_code", "weight_mean", "ic", "ic_t", "direction_acc", "rmse", "mae"]]
        .to_dict(orient="records"),
        "bottom_ic": stock_metrics.sort_values("ic", ascending=True)
        .head(10)
        .loc[:, ["stock_code", "weight_mean", "ic", "ic_t", "direction_acc", "rmse", "mae"]]
        .to_dict(orient="records"),
        "artifact_stock_metrics_test": "stock_metrics_test.parquet",
    }
    # Compute per-stock time-series metrics on the test split for stock-level diagnostics.
    stock_ts_stats = pd.concat(stock_ts_stats_tables, axis=0, ignore_index=True)
    stock_ts_stats_sum = stock_ts_stats.groupby("stock_code", sort=True).sum(numeric_only=True).reset_index()
    stock_ts_metrics = finalize_stock_metrics_from_sufficient_stats(stats=stock_ts_stats_sum)

    # Compute per-stock within-day IC averages as a robustness cross-check.
    stock_daily_ic = pd.concat(stock_daily_ic_tables, axis=0, ignore_index=True)
    stock_daily_summary = (
        stock_daily_ic.groupby("stock_code", sort=True)
        .agg(
            daily_ic_mean=("daily_ic", "mean"),
            daily_rank_ic_mean=("daily_rank_ic", "mean"),
            days=("daily_ic", "size"),
            daily_n_sum=("n", "sum"),
            daily_ic_neg_rate=("daily_ic", lambda s: float(np.mean(s.to_numpy(dtype=float) < 0.0))),
        )
        .reset_index()
    )

    # Merge minute-level and within-day summary tables to one per-stock evaluation table.
    stock_metrics = stock_ts_metrics.merge(stock_daily_summary, on="stock_code", how="left")

    # Tag stocks as good/bad/neutral using an IC t-stat threshold.
    ic_t_threshold = 2.0
    ic_abs_threshold = 0.05
    stock_metrics["verdict"] = "neutral"
    stock_metrics.loc[(stock_metrics["ic"] >= ic_abs_threshold) & (stock_metrics["ic_t"] >= ic_t_threshold), "verdict"] = "good"
    stock_metrics.loc[(stock_metrics["ic"] <= -ic_abs_threshold) & (stock_metrics["ic_t"] <= -ic_t_threshold), "verdict"] = "bad"

    # Write the per-stock table as a parquet artifact for detailed inspection.
    stock_metrics_path = os.path.join(report_dir, "stock_metrics_test.parquet")
    stock_metrics.to_parquet(stock_metrics_path, index=False)

    # Add a compact per-stock summary into metrics.yaml for the markdown report.
    ic_vals = stock_metrics["ic"].to_numpy(dtype=float)
    ic_vals = ic_vals[np.isfinite(ic_vals)]
    q05, q50, q95 = np.quantile(ic_vals, [0.05, 0.50, 0.95]) if len(ic_vals) else [float("nan")] * 3
    verdict_counts = stock_metrics["verdict"].value_counts(dropna=False).to_dict()
    stock_alpha_metrics["per_stock_test"] = {
        "ic_t_threshold": float(ic_t_threshold),
        "ic_abs_threshold": float(ic_abs_threshold),
        "n_stocks": int(len(stock_metrics)),
        "verdict_counts": {str(k): int(v) for k, v in verdict_counts.items()},
        "ic_quantiles": {"q05": float(q05), "q50": float(q50), "q95": float(q95)},
        "top_ic": stock_metrics.sort_values("ic", ascending=False)
        .head(10)
        .loc[:, ["stock_code", "weight_mean", "ic", "ic_t", "direction_acc", "rmse", "mae"]]
        .to_dict(orient="records"),
        "bottom_ic": stock_metrics.sort_values("ic", ascending=True)
        .head(10)
        .loc[:, ["stock_code", "weight_mean", "ic", "ic_t", "direction_acc", "rmse", "mae"]]
        .to_dict(orient="records"),
        "artifact_stock_metrics_test": "stock_metrics_test.parquet",
    }

    basket_all = pd.concat(basket_rows, axis=0, ignore_index=True)
    basket_pred_path = os.path.join(report_dir, "basket_pred.parquet")
    basket_all.to_parquet(basket_pred_path, index=False)
    basket_pred_tables: list[pd.DataFrame] = []
    for model_name, part in basket_all.groupby("model_name", sort=True):
        basket_pred_tables.append(make_basket_prediction_table(basket=part, model_name=str(model_name)))
    basket_pred_table = pd.concat(basket_pred_tables, axis=0, ignore_index=True)
    basket_metrics = compute_metrics(pred_table=basket_pred_table)

    # Stage 4: Fit basis models and synthesize ETF predictions for multiple branches.
    etf_dataset = build_etf_minute_dataset(
        dates=dates_needed,
        etf1m_root=etf1m_root,
        etf_code_int=etf_code_int,
        horizon_minutes=label_horizon_minutes,
        specs_root=specs_root,
        factor_set_name=etf_factor_set_name,
    )

    # Stage 4: Evaluate basket predictions against the real ETF label (Basis evaluation).
    basis_pred_tables: list[pd.DataFrame] = []
    basis_join_rows: list[pd.DataFrame] = []
    for model_name, part in basket_all.groupby("model_name", sort=True):
        # Join basket_pred to the ETF dataset for a like-for-like evaluation table.
        basket_pred = part.loc[:, ["date", "datetime", "split", "basket_pred"]].copy()
        joined = etf_dataset.merge(basket_pred, on=["date", "datetime", "split"], how="inner")

        # Append a standardized prediction table for compute_metrics.
        basis_pred_tables.append(
            make_etf_prediction_table(
                frame=joined,
                model_name=f"basis_{str(model_name)}",
                pred=joined["basket_pred"].to_numpy(dtype=float),
                label_col="label_etf_10m",
            )
        )

        # Append a compact per-minute basis table for research inspection.
        out = joined.loc[:, ["date", "datetime", "split", "MinuteIndex", "basket_pred", "label_etf_10m"]].copy()
        out["basis"] = out["basket_pred"].astype(float) - out["label_etf_10m"].astype(float)
        out["model_name"] = str(model_name)
        basis_join_rows.append(out)

    basis_pred_table = pd.concat(basis_pred_tables, axis=0, ignore_index=True)
    basis_metrics = compute_metrics(pred_table=basis_pred_table)
    # Enrich basis overall metrics with ICIR computed from the daily IC series (test split only).
    for model_name in list(basis_metrics["overall"].keys()):
        daily = pd.DataFrame(basis_metrics["daily"][model_name]).copy()
        ic_mean = float(daily["ic"].mean())
        ic_std = float(daily["ic"].std())
        rank_ic_mean = float(daily["rank_ic"].mean())
        rank_ic_std = float(daily["rank_ic"].std())
        basis_metrics["overall"][model_name]["test"]["icir"] = ic_mean / ic_std
        basis_metrics["overall"][model_name]["test"]["rank_icir"] = rank_ic_mean / rank_ic_std
    basis_pred_path = os.path.join(report_dir, "basis_pred_vs_etf.parquet")
    pd.concat(basis_join_rows, axis=0, ignore_index=True).to_parquet(basis_pred_path, index=False)

    # Stage 4: Pick the best basket branch by test IC and plot its daily IC time series.
    best_basis_model = pick_best_model_by_metric(etf_overall=basis_metrics["overall"], split="test", metric_name="ic")
    best_basis_daily = pd.DataFrame(basis_metrics["rolling"][best_basis_model]).copy()
    plot_daily_timeseries(
        daily=best_basis_daily,
        value_col="ic",
        roll_20_col="ic_roll_20",
        roll_60_col="ic_roll_60",
        title=f"Basis IC (Basket pred vs ETF, {best_basis_model}, test)",
        out_path=os.path.join(report_dir, "fig_basis_best_daily_ic.png"),
    )
    # Compare synthetic basket realized returns to the real ETF realized returns.
    basket_realized = (
        basket_all.groupby(["date", "datetime", "split"], sort=True)
        .agg(
            basket_label=("basket_label", "mean"),
            weight_coverage_pred=("weight_coverage_pred", "mean"),
        )
        .reset_index()
    )
    synthetic_join = etf_dataset.merge(basket_realized, on=["date", "datetime", "split"], how="inner")

    # Compute synthetic-vs-real error summaries for both splits.
    synthetic_vs_real: dict = {}
    for split_name in ["train", "test"]:
        # Build split arrays for stable summary metrics.
        part = synthetic_join.loc[synthetic_join["split"] == split_name].copy()
        pred = part["basket_label"].to_numpy(dtype=float)
        label = part["label_etf_10m"].to_numpy(dtype=float)
        summary = compute_error_summary(pred=pred, label=label)
        summary["ols"] = compute_ols_summary(x=pred, y=label)
        synthetic_vs_real[str(split_name)] = summary

    # Compute minute-bucket error diagnostics on the test split only.
    synthetic_test = synthetic_join.loc[synthetic_join["split"] == "test"].copy()
    synthetic_bucket = summarize_error_by_minute_bucket(
        frame=synthetic_test,
        pred_col="basket_label",
        label_col="label_etf_10m",
        minute_col="MinuteIndex",
        bucket_size=30,
    )

    # Write a detailed synthetic-vs-real join parquet for the research report.
    synthetic_join_path = os.path.join(report_dir, "synthetic_vs_real_etf.parquet")
    synthetic_out = synthetic_join.loc[:, ["date", "datetime", "split", "MinuteIndex", "basket_label", "label_etf_10m", "weight_coverage_pred"]].copy()
    synthetic_out["delta"] = synthetic_out["basket_label"].astype(float) - synthetic_out["label_etf_10m"].astype(float)
    synthetic_out.to_parquet(synthetic_join_path, index=False)

    # Plot synthetic vs real scatter and error distribution on test.
    synthetic_pred_table = make_etf_prediction_table(
        frame=synthetic_test,
        model_name="synthetic_basket_label_vs_etf",
        pred=synthetic_test["basket_label"].to_numpy(dtype=float),
        label_col="label_etf_10m",
    )
    plot_prediction_scatter(
        pred_table=synthetic_pred_table,
        model_name="synthetic_basket_label_vs_etf",
        out_path=os.path.join(report_dir, "fig_synth_vs_real_scatter_test.png"),
    )
    plot_histogram(
        values=(synthetic_test["basket_label"].to_numpy(dtype=float) - synthetic_test["label_etf_10m"].to_numpy(dtype=float)),
        bins=80,
        title="Synthetic minus Real ETF (10m return, test)",
        out_path=os.path.join(report_dir, "fig_synth_vs_real_error_hist_test.png"),
    )
    daily_delta = (
        synthetic_out.loc[synthetic_out["split"] == "test"]
        .groupby("date", sort=True)
        .agg(delta_mean=("delta", "mean"))
        .reset_index()
        .sort_values("date", ascending=True)
    )
    daily_delta["delta_roll_20"] = daily_delta["delta_mean"].rolling(window=20, min_periods=20).mean()
    daily_delta["delta_roll_60"] = daily_delta["delta_mean"].rolling(window=60, min_periods=60).mean()
    plot_daily_timeseries(
        daily=daily_delta,
        value_col="delta_mean",
        roll_20_col="delta_roll_20",
        roll_60_col="delta_roll_60",
        title="Synthetic - Real ETF (Daily mean error, test)",
        out_path=os.path.join(report_dir, "fig_synth_vs_real_daily_delta_test.png"),
    )
    basket_branches = {
        "basket_stock_xgb": basket_all.loc[basket_all["model_name"] == "basket_stock_xgb", ["date", "datetime", "split", "basket_pred"]].copy(),
        "basket_stock_lgbm": basket_all.loc[basket_all["model_name"] == "basket_stock_lgbm", ["date", "datetime", "split", "basket_pred"]].copy(),
    }

    # Stage 4: Build ETF prediction tables for all basket branches.
    etf_pred_tables: list[pd.DataFrame] = []
    for branch_name, basket_pred in basket_branches.items():
        # Build the raw join table for this basket branch.
        joined = etf_dataset.merge(basket_pred, on=["date", "datetime", "split"], how="inner")
        raw_model_name = f"raw_{branch_name}_vs_etf"
        etf_pred_tables.append(
            make_etf_prediction_table(
                frame=joined,
                model_name=raw_model_name,
                pred=joined["basket_pred"].to_numpy(dtype=float),
                label_col="label_etf_10m",
            )
        )

    # Stage 5: Compute ETF-level metrics and write predictions parquet.
    etf_pred_table = pd.concat(etf_pred_tables, axis=0, ignore_index=True)
    etf_pred_path = os.path.join(report_dir, "predictions.parquet")
    etf_pred_table.to_parquet(etf_pred_path, index=False)
    etf_metrics = compute_metrics(pred_table=etf_pred_table)

    # Stage 5: Choose the best model by ETF test RankIC (selection rule in AGENTS.md).
    best_model = pick_best_model_by_metric(etf_overall=etf_metrics["overall"], split="test", metric_name="rank_ic")
    best_row = etf_metrics["overall"][best_model]["test"]

    # Stage 5: Assemble one metrics.yaml for the full three-layer evaluation.
    metrics = {
        "config": {
            "seed": int(seed),
            "etf_code_int": int(etf_code_int),
            "label_horizon_minutes": int(label_horizon_minutes),
            "train_range": [int(train_start), int(train_end)],
            "test_start": int(test_start),
            "test_end": int(test_end),
            "used_train_days": int(len(train_dates)),
            "used_test_days": int(len(test_dates)),
        },
        "stock_alpha": stock_alpha_metrics,
        "basis_eval": {
            "overall": basis_metrics["overall"],
            "selection": {
                "selection_key": "test_ic",
                "selected_model": str(best_basis_model),
                "selected_test_ic": float(basis_metrics["overall"][best_basis_model]["test"]["ic"]),
                "selected_test_rank_ic": float(basis_metrics["overall"][best_basis_model]["test"]["rank_ic"]),
                "selected_test_rmse": float(basis_metrics["overall"][best_basis_model]["test"]["rmse"]),
                "selected_test_mae": float(basis_metrics["overall"][best_basis_model]["test"]["mae"]),
            },
            "artifacts": {
                "basis_pred_vs_etf_parquet": "basis_pred_vs_etf.parquet",
                "fig_basis_best_daily_ic": "fig_basis_best_daily_ic.png",
            },
        },
        "selection": {
            "selection_key": "test_ic",
            "selected_model": str(best_basis_model),
            "selected_test_ic": float(basis_metrics["overall"][best_basis_model]["test"]["ic"]),
            "selected_test_rank_ic": float(basis_metrics["overall"][best_basis_model]["test"]["rank_ic"]),
            "selected_test_rmse": float(basis_metrics["overall"][best_basis_model]["test"]["rmse"]),
            "selected_test_mae": float(basis_metrics["overall"][best_basis_model]["test"]["mae"]),
        },
        "synthetic_vs_real_etf": {
            "overall": synthetic_vs_real,
            "minute_bucket_test": synthetic_bucket.to_dict(orient="records"),
            "artifacts": {
                "synthetic_vs_real_etf_parquet": "synthetic_vs_real_etf.parquet",
                "fig_synth_vs_real_scatter_test": "fig_synth_vs_real_scatter_test.png",
                "fig_synth_vs_real_error_hist_test": "fig_synth_vs_real_error_hist_test.png",
                "fig_synth_vs_real_daily_delta_test": "fig_synth_vs_real_daily_delta_test.png",
            },
        },
        "basket_synthesis": {"overall": basket_metrics["overall"]},
        "etf_level": {"overall": etf_metrics["overall"]},
        "selection_etf": {
            "selection_key": "test_rank_ic",
            "selected_model": str(best_model),
            "selected_test_ic": float(best_row["ic"]),
            "selected_test_rank_ic": float(best_row["rank_ic"]),
            "selected_test_rmse": float(best_row["rmse"]),
            "selected_test_mae": float(best_row["mae"]),
        },
    }
    write_yaml(os.path.join(report_dir, "metrics.yaml"), metrics)
    write_bottom_up_report_md(report_dir=report_dir, run_id=run_id, metrics=metrics)

    # Stage 5: Remove the deprecated stock-panel symlink from earlier runs.
    deprecated_stock_panel_path = os.path.join(repo_root, "stock_panel.parquet")
    if os.path.islink(deprecated_stock_panel_path) or os.path.exists(deprecated_stock_panel_path):
        os.remove(deprecated_stock_panel_path)

    # Stage 5: Create stable top-level deliverable links for the caller.
    for name, target in [
        ("basket_pred.parquet", basket_pred_path),
    ]:
        out_path = os.path.join(repo_root, name)
        if os.path.islink(out_path) or os.path.exists(out_path):
            os.remove(out_path)
        os.symlink(target, out_path)

    # Stage 5: Plot unified comparisons and stability figures for report.md.
    plot_baseline_comparison(metrics=basket_metrics, split="test", out_path=os.path.join(report_dir, "fig_basket_branch_compare_test.png"))
    plot_baseline_comparison(metrics=etf_metrics, split="test", out_path=os.path.join(report_dir, "fig_etf_branch_compare_test.png"))
    best_daily = pd.DataFrame(etf_metrics["rolling"][best_model]).copy()
    plot_daily_timeseries(
        daily=best_daily,
        value_col="ic",
        roll_20_col="ic_roll_20",
        roll_60_col="ic_roll_60",
        title=f"Daily IC ({best_model}, test)",
        out_path=os.path.join(report_dir, "fig_best_daily_ic.png"),
    )
    plot_cumulative_metric(
        daily=pd.DataFrame(etf_metrics["daily"][best_model]).copy(),
        value_col="ic",
        title=f"Cumulative IC ({best_model}, test)",
        out_path=os.path.join(report_dir, "fig_best_cum_ic.png"),
    )
    best_monthly = pd.DataFrame(etf_metrics["by_month"][best_model]).copy()
    plot_monthly_timeseries(
        monthly=best_monthly,
        value_col="ic",
        title=f"Monthly IC ({best_model}, test)",
        out_path=os.path.join(report_dir, "fig_best_monthly_ic.png"),
    )

    # Print final report directory for the caller.
    print(f"[INFO] Done. report_dir={report_dir}")


if __name__ == "__main__":
    main()
