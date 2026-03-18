import datetime as dt
import os
import pathlib
import random
import time

import numpy as np
import pandas as pd

from eval.metrics import compute_metrics
from eval.feature_importance import compute_feature_ic_table
from eval.walk_forward import run_walk_forward_validation
from features.components import compute_component_features
from features.etf import compute_etf_features, compute_label_from_close
from models.baselines import compute_baseline_preds, fit_linear_baseline
from models.xgb import fit_xgb_model
from eval.plots import (
    plot_baseline_comparison,
    plot_cumulative_metric,
    plot_daily_timeseries,
    plot_feature_importance,
    plot_prediction_scatter,
    plot_prediction_timeseries,
)
from eval.writers import (
    compute_coverage_audit_tables,
    write_data_audit_md,
    write_report_md,
    write_report_tex,
    write_summary_md,
    write_yaml,
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


def load_etf_minute_bars(etf1m_root: str, date: int, etf_code_int: int) -> pd.DataFrame:
    """Load one-day ETF 1m bars for a specific ETF."""

    # Build the path and load the day file.
    year = str(date)[:4]
    file_path = os.path.join(etf1m_root, year, f"{date}.feather")
    day = pd.read_feather(file_path)

    # Filter to the target ETF and keep necessary columns.
    day = day.loc[day["StockCode"] == etf_code_int, ["DateTime", "Close", "Vol", "Amount", "Date"]].copy()
    day = day.sort_values("DateTime", ascending=True)
    return day


def load_stock_minute_bars(stock1m_root: str, date: int, constituents: np.ndarray) -> pd.DataFrame:
    """Load one-day stock 1m bars filtered by constituents."""

    # Build the path and load the day file.
    year = str(date)[:4]
    file_path = os.path.join(stock1m_root, year, f"{date}.feather")
    day = pd.read_feather(file_path)

    # Filter to constituents and keep only required columns.
    mask = day["StockCode"].isin(constituents)
    day = day.loc[mask, ["StockCode", "DateTime", "Close", "Vol"]].copy()
    day = day.sort_values(["StockCode", "DateTime"], ascending=[True, True])
    return day


def build_day_frame(
    date: int,
    weights: pd.DataFrame,
    etf1m_root: str,
    stock1m_root: str,
    etf_code_int: int,
    horizon_minutes: int,
) -> pd.DataFrame:
    """Build one-day modeling frame with features and label."""

    # Load ETF bars and compute ETF features and label.
    etf_day = load_etf_minute_bars(etf1m_root=etf1m_root, date=date, etf_code_int=etf_code_int)
    etf_features = compute_etf_features(etf_day=etf_day)
    label = compute_label_from_close(etf_day=etf_day, horizon_minutes=horizon_minutes)
    label_frame = pd.DataFrame({"DateTime": etf_day["DateTime"], "label": label})

    # Load weights as-of date and constituent stock bars for aggregation.
    day_weights = get_constituent_weights_for_date(weights=weights, date=date)
    constituents = day_weights["con_int"].to_numpy()
    stock_day = load_stock_minute_bars(stock1m_root=stock1m_root, date=date, constituents=constituents)
    comp_features = compute_component_features(stock_day=stock_day, day_weights=day_weights)

    # Merge ETF features, component features, and label on DateTime.
    frame = etf_features.merge(comp_features, on="DateTime", how="left")
    frame = frame.merge(label_frame, on="DateTime", how="left")

    # Add intraday time encoding features based on minute index within the day.
    frame = frame.sort_values("DateTime", ascending=True).reset_index(drop=True)
    minutes_in_day = int(len(frame))
    minute_index = np.arange(minutes_in_day, dtype=int)
    frame["minute_index"] = minute_index.astype(float)
    minute_phase = 2.0 * np.pi * minute_index.astype(float) / float(minutes_in_day)
    frame["minute_sin"] = np.sin(minute_phase)
    frame["minute_cos"] = np.cos(minute_phase)

    # Add ETF-constituent dispersion/breadth derived features.
    frame["etf_minus_comp_ret_1m"] = frame["ret_1m"] - frame["comp_w_ret_1m"]
    return frame


def date_to_split(date: int) -> str:
    """Map date to split label using AGENTS.md time partition."""

    # Apply the required split rules.
    if 20210101 <= date <= 20231231:
        return "train"
    if date >= 20240201:
        return "test"
    return "ignore"


def load_or_build_day_frame(
    date: int,
    weights: pd.DataFrame,
    cache_root: str,
    etf1m_root: str,
    stock1m_root: str,
    etf_code_int: int,
    horizon_minutes: int,
) -> pd.DataFrame | None:
    """Load cached day frame or build and cache it."""

    # Decide cache file path.
    ensure_dir(cache_root)
    cache_path = os.path.join(cache_root, f"{date}.parquet")

    # Load from cache if available.
    if os.path.exists(cache_path):
        day = pd.read_parquet(cache_path)
        # Normalize inf to NaN to keep downstream models stable.
        day = day.replace([np.inf, -np.inf], np.nan)
        return day

    # Build the day frame and cache it.
    try:
        day = build_day_frame(
            date=date,
            weights=weights,
            etf1m_root=etf1m_root,
            stock1m_root=stock1m_root,
            etf_code_int=etf_code_int,
            horizon_minutes=horizon_minutes,
        )
    except Exception as exc:
        print(f"[ERROR] Failed to build day frame for {date}: {exc}")
        return None

    # Persist to cache to speed up subsequent runs.
    day = day.replace([np.inf, -np.inf], np.nan)
    day.to_parquet(cache_path, index=False)
    return day


def build_full_dataset(
    dates: list[int],
    weights: pd.DataFrame,
    cache_root: str,
    etf1m_root: str,
    stock1m_root: str,
    etf_code_int: int,
    horizon_minutes: int,
) -> pd.DataFrame:
    """Build the full dataset over dates, using /data-cache for caching."""

    # Iterate day-by-day to keep memory bounded and allow skipping failures.
    frames: list[pd.DataFrame] = []
    start_time = time.time()
    for idx, date in enumerate(dates):
        # Print periodic progress without spamming.
        if idx % 50 == 0:
            elapsed = time.time() - start_time
            print(f"[INFO] Processing {idx}/{len(dates)} days. Elapsed={elapsed:.1f}s.")

        # Load or build day frame and skip on errors.
        day = load_or_build_day_frame(
            date=date,
            weights=weights,
            cache_root=cache_root,
            etf1m_root=etf1m_root,
            stock1m_root=stock1m_root,
            etf_code_int=etf_code_int,
            horizon_minutes=horizon_minutes,
        )
        if day is None:
            continue

        # Attach split and date columns for later grouping and filtering.
        split = date_to_split(date=date)
        day = day.copy()
        day["date"] = date
        day["split"] = split
        frames.append(day)

    # Concatenate into a single frame.
    full = pd.concat(frames, axis=0, ignore_index=True)

    # Drop rows with unusable splits or label NaNs from horizon truncation.
    full = full.loc[full["split"].isin(["train", "test"])].copy()
    full = full.dropna(subset=["label"])

    # Sort for correct baseline computations.
    full = full.sort_values(["split", "DateTime"], ascending=[True, True]).reset_index(drop=True)
    return full

def make_prediction_table(
    frame: pd.DataFrame,
    model_name: str,
    pred: np.ndarray,
) -> pd.DataFrame:
    """Build a standardized prediction table for output."""

    # Standardize output columns required by AGENTS.md.
    out = pd.DataFrame(
        {
            "datetime": pd.to_datetime(frame["DateTime"]).astype("datetime64[ns]"),
            "date": frame["date"].astype(int),
            "pred": pred.astype(float),
            "label": frame["label"].astype(float),
            "split": frame["split"].astype(str),
            "model_name": model_name,
        }
    )
    return out


def sample_spot_checks(
    etf1m_root: str,
    date: int,
    etf_code_int: int,
    num_checks: int,
    horizon_minutes: int,
) -> list[dict]:
    """Sample spot checks to demonstrate no look-ahead."""

    # Load the ETF day and compute label and features for inspection.
    etf_day = load_etf_minute_bars(etf1m_root=etf1m_root, date=date, etf_code_int=etf_code_int)
    features = compute_etf_features(etf_day=etf_day)
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
    """Run end-to-end pipeline for 510500 10-minute return prediction."""

    # Define core constants from AGENTS.md.
    seed = 42
    etf_code_int = 510500
    label_horizon_minutes = 10
    train_start = 20210101
    train_end = 20231231
    test_start = 20240201

    # Define data locations.
    repo_root = os.path.dirname(os.path.abspath(__file__))
    weight_path = os.path.join(repo_root, "data/ashare/market/index_weight/000905.feather")
    etf1m_root = "/data/ashare/market/etf1m"
    stock1m_root = "/data/ashare/market/stock1m"
    cache_root = "/data-cache/index500-predict/510500_10m_frame_v3"

    # Create output run directory under repo report folder.
    run_id = dt.datetime.now().strftime("run_%Y%m%d_%H%M%S")
    report_dir = os.path.join(repo_root, "report", run_id)
    ensure_dir(report_dir)

    # Set seeds for reproducibility.
    set_global_seed(seed=seed)

    # Discover available dates and keep only those needed for train/test.
    all_dates = list_available_dates_from_etf1m_dir(etf1m_dir=etf1m_root)
    dates_needed = [d for d in all_dates if (train_start <= d <= train_end) or (d >= test_start)]
    print(f"[INFO] Available dates={len(all_dates)}, using dates={len(dates_needed)}.")

    # Load weights once for all dates.
    weights = load_index_weights(weight_path=weight_path)
    print(f"[INFO] Loaded weights rows={len(weights)}. trade_date_min={weights['trade_date'].min()} trade_date_max={weights['trade_date'].max()}.")

    # Build full dataset with cached day frames.
    dataset = build_full_dataset(
        dates=dates_needed,
        weights=weights,
        cache_root=cache_root,
        etf1m_root=etf1m_root,
        stock1m_root=stock1m_root,
        etf_code_int=etf_code_int,
        horizon_minutes=label_horizon_minutes,
    )
    print(f"[INFO] Full dataset rows={len(dataset)} cols={len(dataset.columns)}.")

    # Define features and create time-based train/val split within training range.
    feature_cols = [
        "ret_1m",
        "ret_5m",
        "ret_10m",
        "ret_30m",
        "amt_chg_1m",
        "rv_10",
        "rv_20",
        "vol_roll_mean_20",
        "vol_roll_std_20",
        "amt_roll_mean_20",
        "amt_roll_std_20",
        "minute_index",
        "minute_sin",
        "minute_cos",
        "comp_w_ret_1m",
        "comp_w_vol_chg_1m",
        "comp_xs_ret_std_1m",
        "comp_breadth_pos_ret_1m",
        "etf_minus_comp_ret_1m",
    ]
    train = dataset.loc[dataset["split"] == "train"].copy()
    test = dataset.loc[dataset["split"] == "test"].copy()

    # Create a time-ordered validation month for early stopping (no hard-coded dates).
    train_month = (train["date"].astype(int) // 100).astype(int)
    months = np.sort(train_month.unique())
    val_month = int(months[-1])
    val = train.loc[train_month == val_month].copy()
    train_fit = train.loc[train_month < val_month].copy()
    print(f"[INFO] Train fit rows={len(train_fit)}, val rows={len(val)}, test rows={len(test)}.")

    # Run walk-forward validation on the training split to check stability across regimes.
    walk_forward_rows = run_walk_forward_validation(train=train, features=feature_cols, seed=seed, min_train_months=12)
    print(f"[INFO] Walk-forward folds={len(walk_forward_rows)} (monthly, expanding window).")

    # Compute naive baselines.
    dataset_with_baselines = compute_baseline_preds(frame=dataset, horizon_minutes=label_horizon_minutes)

    # Fit linear model baseline and generate its predictions.
    lin_features = ["ret_1m", "ret_5m", "ret_10m", "ret_30m"]
    linear_model = fit_linear_baseline(train=train_fit, features=lin_features)
    linear_pred = np.full(shape=(len(dataset_with_baselines),), fill_value=np.nan, dtype=float)
    linear_mask = dataset_with_baselines[lin_features].notna().all(axis=1).to_numpy()
    linear_pred[linear_mask] = linear_model.predict(dataset_with_baselines.loc[linear_mask, lin_features].to_numpy())
    dataset_with_baselines["pred_linear_model"] = linear_pred

    # Fit main model and generate predictions.
    xgb_model = fit_xgb_model(train=train_fit, val=val, features=feature_cols, seed=seed)
    xgb_pred = xgb_model.predict(dataset_with_baselines[feature_cols].to_numpy())
    dataset_with_baselines["pred_xgb"] = xgb_pred

    # Assemble long-form prediction tables for all models.
    pred_tables: list[pd.DataFrame] = []
    pred_tables.append(make_prediction_table(dataset_with_baselines, "zero", dataset_with_baselines["pred_zero"].to_numpy()))
    pred_tables.append(
        make_prediction_table(dataset_with_baselines, "last_value", dataset_with_baselines["pred_last_value"].to_numpy())
    )
    pred_tables.append(
        make_prediction_table(dataset_with_baselines, "rolling_mean", dataset_with_baselines["pred_rolling_mean"].to_numpy())
    )
    pred_tables.append(
        make_prediction_table(dataset_with_baselines, "linear_model", dataset_with_baselines["pred_linear_model"].to_numpy())
    )
    pred_tables.append(make_prediction_table(dataset_with_baselines, "xgb", dataset_with_baselines["pred_xgb"].to_numpy()))
    predictions = pd.concat(pred_tables, axis=0, ignore_index=True)

    # Write predictions to parquet as required.
    predictions_path = os.path.join(report_dir, "predictions.parquet")
    predictions.to_parquet(predictions_path, index=False)

    # Compute metrics and prepare metrics.yaml payload.
    metrics = compute_metrics(pred_table=predictions)
    metrics["walk_forward"] = {"xgb": walk_forward_rows}
    new_features = [
        "minute_index",
        "minute_sin",
        "minute_cos",
        "amt_chg_1m",
        "rv_10",
        "rv_20",
        "comp_breadth_pos_ret_1m",
        "etf_minus_comp_ret_1m",
    ]
    metrics["feature_ic"] = {"test": compute_feature_ic_table(dataset=dataset, features=new_features, split="test")}
    metrics["xgb_feature_importance"] = [
        {"feature": str(name), "importance": float(val)}
        for name, val in sorted(
            zip(feature_cols, xgb_model.feature_importances_),
            key=lambda x: float(x[1]),
            reverse=True,
        )
    ]

    # Compute coverage audit once and persist the summary into metrics for the main report.
    coverage_audit = compute_coverage_audit_tables(dataset=dataset)
    metrics["data_audit_summary"] = {
        "overall_join_retention_mean": float(coverage_audit["overall"]["join_retention_mean"]),
        "first_minute_weight_coverage_mean": float(coverage_audit["first_minute"]["weight_coverage_mean"]),
        "first_minute_missing_rate_mean": float(coverage_audit["first_minute"]["missing_rate_mean"]),
        "first_minute_join_retention_mean": float(coverage_audit["first_minute"]["join_retention_mean"]),
        "last_minute_weight_coverage_mean": float(coverage_audit["last_minute"]["weight_coverage_mean"]),
        "last_minute_missing_rate_mean": float(coverage_audit["last_minute"]["missing_rate_mean"]),
        "last_minute_join_retention_mean": float(coverage_audit["last_minute"]["join_retention_mean"]),
    }

    # Write metrics.yaml after all report-relevant summaries are attached.
    write_yaml(os.path.join(report_dir, "metrics.yaml"), metrics)

    # Write data audit and spot checks.
    first_train_date = int(min([d for d in dates_needed if 20210101 <= d <= 20231231]))
    last_train_date = int(max([d for d in dates_needed if 20210101 <= d <= 20231231]))
    first_test_date = int(min([d for d in dates_needed if d >= test_start]))
    spot_dates = [first_train_date, last_train_date, first_test_date]
    spot_checks: list[dict] = []
    for date in spot_dates:
        spot_checks.extend(
            sample_spot_checks(
                etf1m_root=etf1m_root,
                date=date,
                etf_code_int=etf_code_int,
                num_checks=1,
                horizon_minutes=label_horizon_minutes,
            )
        )
    write_data_audit_md(
        report_dir=report_dir,
        spot_checks=spot_checks,
        horizon_minutes=label_horizon_minutes,
        coverage_audit=coverage_audit,
    )

    # Write summary and report markdown.
    write_summary_md(report_dir=report_dir, run_id=run_id, metrics=metrics, feature_cols=feature_cols)
    write_report_md(report_dir=report_dir, run_id=run_id, metrics=metrics)
    write_report_tex(
        report_dir=os.path.join(repo_root, "report"),
        run_id=run_id,
        metrics=metrics,
        out_name="report0318.tex",
        asset_prefix=run_id,
    )
    write_report_tex(
        report_dir=report_dir,
        run_id=run_id,
        metrics=metrics,
        out_name="report0318.tex",
        asset_prefix="",
    )

    # Create figures based on the main model daily series.
    daily_merged = pd.DataFrame(metrics["rolling"]["xgb"])
    plot_daily_timeseries(
        daily=daily_merged,
        value_col="ic",
        roll_20_col="ic_roll_20",
        roll_60_col="ic_roll_60",
        title="Daily IC (test)",
        out_path=os.path.join(report_dir, "fig_ic_timeseries.png"),
    )
    plot_cumulative_metric(
        daily=daily_merged,
        value_col="ic",
        title="Cumulative IC (test)",
        out_path=os.path.join(report_dir, "fig_ic_cum.png"),
    )
    plot_daily_timeseries(
        daily=daily_merged,
        value_col="rank_ic",
        roll_20_col="rank_ic_roll_20",
        roll_60_col="rank_ic_roll_60",
        title="Daily Rank IC (test)",
        out_path=os.path.join(report_dir, "fig_rank_ic_timeseries.png"),
    )
    plot_daily_timeseries(
        daily=daily_merged,
        value_col="direction_acc",
        roll_20_col="dir_acc_roll_20",
        roll_60_col="dir_acc_roll_60",
        title="Daily Direction Accuracy (test)",
        out_path=os.path.join(report_dir, "fig_direction_acc_timeseries.png"),
    )
    plot_baseline_comparison(metrics=metrics, split="test", out_path=os.path.join(report_dir, "fig_baseline_comparison.png"))
    plot_feature_importance(
        importances=metrics["xgb_feature_importance"],
        out_path=os.path.join(report_dir, "fig_xgb_feature_importance.png"),
        top_k=20,
    )
    plot_prediction_scatter(pred_table=predictions, model_name="xgb", out_path=os.path.join(report_dir, "fig_prediction_scatter.png"))
    plot_prediction_timeseries(
        pred_table=predictions, model_name="xgb", out_path=os.path.join(report_dir, "fig_prediction_timeseries.png")
    )

    # Print final report directory for the caller.
    print(f"[INFO] Done. report_dir={report_dir}")


if __name__ == "__main__":
    main()
