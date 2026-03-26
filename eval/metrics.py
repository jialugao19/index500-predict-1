import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Pearson correlation for 1D arrays."""

    # Remove NaNs before correlation.
    mask = np.isfinite(a) & np.isfinite(b)
    a_clean = a[mask]
    b_clean = b[mask]
    # Return NaN for degenerate inputs to avoid runtime warnings.
    if len(a_clean) < 2:
        return float("nan")
    if float(np.std(a_clean)) == 0.0:
        return float("nan")
    if float(np.std(b_clean)) == 0.0:
        return float("nan")
    return float(np.corrcoef(a_clean, b_clean)[0, 1])


def safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Spearman rank correlation for 1D arrays."""

    # Remove NaNs and use pandas rank to get Spearman.
    mask = np.isfinite(a) & np.isfinite(b)
    a_rank = pd.Series(a[mask]).rank().to_numpy()
    b_rank = pd.Series(b[mask]).rank().to_numpy()
    return safe_corr(a_rank, b_rank)


def compute_metrics(pred_table: pd.DataFrame) -> dict:
    """Compute overall and daily metrics for each model and split."""

    # Prepare output container with stable dict keys.
    results: dict = {
        "overall": {},
        "daily": {},
        "by_month": {},
        "rolling": {},
    }

    # Evaluate each model_name separately to avoid mixing baselines and main model.
    for model_name, part in pred_table.groupby("model_name", sort=True):
        # Compute overall split metrics.
        split_rows: dict = {}
        for split_name, sp in part.groupby("split", sort=True):
            # Compute basic regression and correlation metrics on the split.
            pred = sp["pred"].to_numpy(dtype=float)
            label = sp["label"].to_numpy(dtype=float)
            mask = np.isfinite(pred) & np.isfinite(label)
            pred_clean = pred[mask]
            label_clean = label[mask]
            split_rows[split_name] = {
                "ic": safe_corr(pred, label),
                "rank_ic": safe_spearman(pred, label),
                "direction_acc": float(np.mean((pred_clean > 0.0) == (label_clean > 0.0))) if len(pred_clean) else float("nan"),
                "rmse": float(np.sqrt(mean_squared_error(label_clean, pred_clean))) if len(pred_clean) else float("nan"),
                "mae": float(mean_absolute_error(label_clean, pred_clean)) if len(pred_clean) else float("nan"),
                "n": int(len(pred_clean)),
            }
        results["overall"][model_name] = split_rows

        # Compute daily metrics for the test split as a time series.
        test_part = part.loc[part["split"] == "test"].copy()
        daily_rows: list[dict] = []
        for date, day in test_part.groupby("date", sort=True):
            # Compute per-day metrics using within-day minute samples.
            pred = day["pred"].to_numpy(dtype=float)
            label = day["label"].to_numpy(dtype=float)
            mask = np.isfinite(pred) & np.isfinite(label)
            pred_clean = pred[mask]
            label_clean = label[mask]
            daily_rows.append(
                {
                    "date": int(date),
                    "ic": safe_corr(pred, label),
                    "rank_ic": safe_spearman(pred, label),
                    "direction_acc": float(np.mean((pred_clean > 0.0) == (label_clean > 0.0)))
                    if len(pred_clean)
                    else float("nan"),
                    "rmse": float(np.sqrt(mean_squared_error(label_clean, pred_clean))) if len(pred_clean) else float("nan"),
                    "mae": float(mean_absolute_error(label_clean, pred_clean)) if len(pred_clean) else float("nan"),
                    "n": int(len(pred_clean)),
                }
            )
        daily = pd.DataFrame(daily_rows).sort_values("date", ascending=True)
        results["daily"][model_name] = daily.to_dict(orient="records")

        # Compute by-month metrics for the test split.
        test_part["month"] = (test_part["date"].astype(int) // 100).astype(int)
        month_rows: list[dict] = []
        for month, mp in test_part.groupby("month", sort=True):
            # Compute month-level IC and error metrics.
            pred = mp["pred"].to_numpy(dtype=float)
            label = mp["label"].to_numpy(dtype=float)
            mask = np.isfinite(pred) & np.isfinite(label)
            pred_clean = pred[mask]
            label_clean = label[mask]
            month_rows.append(
                {
                    "month": int(month),
                    "ic": safe_corr(pred, label),
                    "rank_ic": safe_spearman(pred, label),
                    "direction_acc": float(np.mean((pred_clean > 0.0) == (label_clean > 0.0)))
                    if len(pred_clean)
                    else float("nan"),
                    "rmse": float(np.sqrt(mean_squared_error(label_clean, pred_clean))) if len(pred_clean) else float("nan"),
                    "mae": float(mean_absolute_error(label_clean, pred_clean)) if len(pred_clean) else float("nan"),
                    "n": int(len(pred_clean)),
                }
            )
        monthly = pd.DataFrame(month_rows).sort_values("month", ascending=True)
        results["by_month"][model_name] = monthly.to_dict(orient="records")

        # Compute rolling window metrics on daily series.
        roll = daily.copy()
        roll["ic_roll_20"] = roll["ic"].rolling(window=20, min_periods=20).mean()
        roll["ic_roll_60"] = roll["ic"].rolling(window=60, min_periods=60).mean()
        roll["rank_ic_roll_20"] = roll["rank_ic"].rolling(window=20, min_periods=20).mean()
        roll["rank_ic_roll_60"] = roll["rank_ic"].rolling(window=60, min_periods=60).mean()
        roll["dir_acc_roll_20"] = roll["direction_acc"].rolling(window=20, min_periods=20).mean()
        roll["dir_acc_roll_60"] = roll["direction_acc"].rolling(window=60, min_periods=60).mean()
        results["rolling"][model_name] = roll.to_dict(orient="records")

    return results


def compute_error_summary(pred: np.ndarray, label: np.ndarray) -> dict:
    """Compute a compact error distribution summary for two aligned series."""

    # Filter to finite pairs to match compute_metrics conventions.
    mask = np.isfinite(pred) & np.isfinite(label)
    pred_clean = pred[mask].astype(float)
    label_clean = label[mask].astype(float)
    err = pred_clean - label_clean

    # Compute central tendency and tail quantiles for tracking diagnostics.
    q05, q50, q95 = np.quantile(err, [0.05, 0.50, 0.95]) if len(err) else [float("nan")] * 3
    abs_err = np.abs(err)
    abs_q50, abs_q95 = np.quantile(abs_err, [0.50, 0.95]) if len(abs_err) else [float("nan")] * 2

    # Return plain Python numbers for YAML serialization.
    return {
        "n": int(len(err)),
        "ic": safe_corr(pred_clean, label_clean),
        "rank_ic": safe_spearman(pred_clean, label_clean),
        "direction_acc": float(np.mean((pred_clean > 0.0) == (label_clean > 0.0))) if len(err) else float("nan"),
        "rmse": float(np.sqrt(mean_squared_error(label_clean, pred_clean))) if len(err) else float("nan"),
        "mae": float(mean_absolute_error(label_clean, pred_clean)) if len(err) else float("nan"),
        "bias_mean": float(np.mean(err)) if len(err) else float("nan"),
        "bias_std": float(np.std(err)) if len(err) else float("nan"),
        "err_q05": float(q05),
        "err_q50": float(q50),
        "err_q95": float(q95),
        "abs_err_q50": float(abs_q50),
        "abs_err_q95": float(abs_q95),
        "pred_std": float(np.std(pred_clean)) if len(err) else float("nan"),
        "label_std": float(np.std(label_clean)) if len(err) else float("nan"),
    }


def compute_ols_summary(x: np.ndarray, y: np.ndarray) -> dict:
    """Fit y = a + b*x by OLS and return slope/intercept/R2."""

    # Filter to finite pairs and build the design matrix.
    mask = np.isfinite(x) & np.isfinite(y)
    x_clean = x[mask].astype(float)
    y_clean = y[mask].astype(float)
    x_mat = np.column_stack([np.ones(len(x_clean), dtype=float), x_clean])

    # Solve least squares and compute fit quality diagnostics.
    coef, *_ = np.linalg.lstsq(x_mat, y_clean, rcond=None)
    intercept = float(coef[0])
    slope = float(coef[1])
    y_hat = intercept + slope * x_clean
    ss_res = float(np.sum((y_clean - y_hat) * (y_clean - y_hat)))
    ss_tot = float(np.sum((y_clean - float(np.mean(y_clean))) * (y_clean - float(np.mean(y_clean)))))
    r2 = 1.0 - ss_res / ss_tot

    # Return plain floats for report serialization.
    return {"n": int(len(x_clean)), "intercept": intercept, "slope": slope, "r2": float(r2)}


def summarize_error_by_minute_bucket(
    frame: pd.DataFrame,
    pred_col: str,
    label_col: str,
    minute_col: str,
    bucket_size: int,
) -> pd.DataFrame:
    """Summarize prediction error by minute-of-day buckets."""

    # Build bucket keys and keep the minimal columns needed.
    part = frame.loc[:, [minute_col, pred_col, label_col]].copy()
    part["minute_bucket"] = (part[minute_col].astype(int) // int(bucket_size)).astype(int)

    # Compute per-bucket error stats with finite-pair filtering.
    rows: list[dict] = []
    for bucket, bp in part.groupby("minute_bucket", sort=True):
        # Compute correlations and errors inside the bucket.
        pred = bp[pred_col].to_numpy(dtype=float)
        label = bp[label_col].to_numpy(dtype=float)
        mask = np.isfinite(pred) & np.isfinite(label)
        pred_clean = pred[mask]
        label_clean = label[mask]
        err = pred_clean - label_clean
        rows.append(
            {
                "minute_bucket": int(bucket),
                "ic": safe_corr(pred_clean, label_clean),
                "rank_ic": safe_spearman(pred_clean, label_clean),
                "rmse": float(np.sqrt(mean_squared_error(label_clean, pred_clean))) if len(err) else float("nan"),
                "mae": float(mean_absolute_error(label_clean, pred_clean)) if len(err) else float("nan"),
                "bias_mean": float(np.mean(err)) if len(err) else float("nan"),
                "bias_std": float(np.std(err)) if len(err) else float("nan"),
                "n": int(len(err)),
            }
        )

    # Return a stable table for markdown and YAML writers.
    out = pd.DataFrame(rows).sort_values("minute_bucket", ascending=True)
    return out
