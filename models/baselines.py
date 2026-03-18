import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def fit_linear_baseline(train: pd.DataFrame, features: list[str]) -> LinearRegression:
    """Fit a simple linear regression baseline."""

    # Select rows with no NaNs in the feature set.
    mask = train[features].notna().all(axis=1)
    x_train = train.loc[mask, features].to_numpy()
    y_train = train.loc[mask, "label"].to_numpy()

    # Fit linear regression with intercept.
    model = LinearRegression()
    model.fit(x_train, y_train)
    return model


def compute_baseline_preds(frame: pd.DataFrame, horizon_minutes: int) -> pd.DataFrame:
    """Compute required baselines for each split."""

    # Enforce that the baseline lag cannot see prices after time t.
    shift_n = int(horizon_minutes) + 1
    assert shift_n > int(horizon_minutes)

    # Compute baselines within each split to avoid cross-split leakage.
    frame = frame.sort_values(["split", "DateTime"], ascending=[True, True]).copy()
    pieces: list[pd.DataFrame] = []
    for split_name, part in frame.groupby("split", sort=False):
        # Compute baselines using only the label series in that split.
        part = part.sort_values("DateTime", ascending=True).copy()
        part["pred_zero"] = 0.0
        part["pred_last_value"] = part["label"].shift(shift_n)
        part["pred_rolling_mean"] = part["label"].shift(shift_n).rolling(window=20, min_periods=20).mean()
        pieces.append(part)

    # Recombine split pieces back into one frame.
    combined = pd.concat(pieces, axis=0, ignore_index=True)
    return combined

