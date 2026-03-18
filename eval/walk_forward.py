import numpy as np
import pandas as pd

from eval.metrics import safe_corr, safe_spearman
from models.xgb import fit_xgb_model


def run_walk_forward_validation(
    train: pd.DataFrame,
    features: list[str],
    seed: int,
    min_train_months: int,
) -> list[dict]:
    """Run an expanding-window walk-forward validation on the training split."""

    # Build a sorted month index to define time-ordered folds.
    month = (train["date"].astype(int) // 100).astype(int)
    months = np.sort(month.unique())
    folds: list[dict] = []

    # Iterate month-by-month with an expanding training window.
    for idx in range(int(min_train_months), int(len(months))):
        # Define fold train/val partitions by month.
        train_months = months[:idx]
        val_month = int(months[idx])
        fold_train = train.loc[month.isin(train_months)].copy()
        fold_val = train.loc[month == val_month].copy()

        # Fit the model on fold training data and evaluate on fold validation data.
        model = fit_xgb_model(train=fold_train, val=fold_val, features=features, seed=seed)
        pred = model.predict(fold_val[features].to_numpy())
        label = fold_val["label"].to_numpy(dtype=float)

        # Store fold metrics for stability analysis.
        folds.append(
            {
                "val_month": val_month,
                "train_month_start": int(train_months.min()),
                "train_month_end": int(train_months.max()),
                "ic": safe_corr(pred, label),
                "rank_ic": safe_spearman(pred, label),
                "n": int(len(label)),
            }
        )

    # Return a plain list of dicts for YAML/markdown serialization.
    return folds

