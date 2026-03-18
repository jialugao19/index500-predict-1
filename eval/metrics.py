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
