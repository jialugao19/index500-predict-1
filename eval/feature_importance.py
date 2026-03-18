import numpy as np
import pandas as pd

from eval.metrics import safe_corr


def compute_feature_ic_table(
    dataset: pd.DataFrame,
    features: list[str],
    split: str,
) -> list[dict]:
    """Compute per-feature IC and IR diagnostics on a given split."""

    # Filter to the requested split and ensure consistent sorting.
    part = dataset.loc[dataset["split"] == split].sort_values(["date", "DateTime"], ascending=[True, True]).copy()

    # Compute per-feature daily IC time series for IR estimation.
    rows: list[dict] = []
    for feature_name in features:
        # Compute overall IC using all samples on the split.
        feature = part[feature_name].to_numpy(dtype=float)
        label = part["label"].to_numpy(dtype=float)
        overall_ic = safe_corr(feature, label)

        # Compute daily IC series for IR as mean/std across days.
        daily_ics: list[float] = []
        for date, day in part.groupby("date", sort=True):
            daily_feature = day[feature_name].to_numpy(dtype=float)
            daily_label = day["label"].to_numpy(dtype=float)
            daily_ics.append(safe_corr(daily_feature, daily_label))
        daily_ics_arr = np.asarray(daily_ics, dtype=float)
        daily_ics_clean = daily_ics_arr[np.isfinite(daily_ics_arr)]
        ir = float(np.mean(daily_ics_clean) / np.std(daily_ics_clean)) if len(daily_ics_clean) >= 2 else float("nan")

        # Store summary diagnostics for the report.
        rows.append(
            {
                "feature": str(feature_name),
                "split": str(split),
                "ic": float(overall_ic),
                "daily_ic_mean": float(np.mean(daily_ics_clean)) if len(daily_ics_clean) > 0 else float("nan"),
                "daily_ic_std": float(np.std(daily_ics_clean)) if len(daily_ics_clean) > 0 else float("nan"),
                "ir": ir,
                "n_days": int(len(daily_ics_clean)),
                "n_rows": int(len(part)),
            }
        )

    # Return as a list of dicts for YAML/markdown output.
    rows_sorted = sorted(rows, key=lambda x: (np.nan_to_num(x["ic"], nan=-1e9)), reverse=True)
    return rows_sorted

