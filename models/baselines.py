import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LinearRegression, Ridge


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


def fit_ridge_regression(train: pd.DataFrame, features: list[str], alpha: float) -> dict:
    """Fit a Ridge regression baseline with train-only impute and z-score."""

    # Build the raw training matrix and label vector.
    x_raw = train.loc[:, features].to_numpy(dtype=float)
    y = train.loc[:, "label"].to_numpy(dtype=float)

    # Compute train-only imputation values for missing entries.
    impute_values = np.nanmean(x_raw, axis=0)
    x_imputed = np.where(np.isfinite(x_raw), x_raw, impute_values[None, :])

    # Compute train-only z-score parameters and standardize features.
    means = np.mean(x_imputed, axis=0)
    stds = np.std(x_imputed, axis=0)
    # Stabilize constant features to avoid NaNs from division by zero.
    stds_safe = np.where(stds == 0.0, 1.0, stds)
    x = (x_imputed - means[None, :]) / stds_safe[None, :]

    # Fit Ridge regression with an explicit L2 penalty.
    model = Ridge(alpha=float(alpha), fit_intercept=True, random_state=0)
    model.fit(x, y)
    return {"model": model, "impute_values": impute_values, "means": means, "stds": stds_safe, "features": list(features)}


def fit_lasso_regression(train: pd.DataFrame, features: list[str], alpha: float) -> dict:
    """Fit a Lasso regression baseline with train-only impute and z-score."""

    # Build the raw training matrix and label vector.
    x_raw = train.loc[:, features].to_numpy(dtype=float)
    y = train.loc[:, "label"].to_numpy(dtype=float)

    # Compute train-only imputation values for missing entries.
    impute_values = np.nanmean(x_raw, axis=0)
    x_imputed = np.where(np.isfinite(x_raw), x_raw, impute_values[None, :])

    # Compute train-only z-score parameters and standardize features.
    means = np.mean(x_imputed, axis=0)
    stds = np.std(x_imputed, axis=0)
    # Stabilize constant features to avoid NaNs from division by zero.
    stds_safe = np.where(stds == 0.0, 1.0, stds)
    x = (x_imputed - means[None, :]) / stds_safe[None, :]

    # Fit Lasso regression with an explicit L1 penalty.
    model = Lasso(alpha=float(alpha), fit_intercept=True, max_iter=5000, random_state=0)
    model.fit(x, y)
    return {"model": model, "impute_values": impute_values, "means": means, "stds": stds_safe, "features": list(features)}


def predict_linear_model(bundle: dict, frame: pd.DataFrame) -> np.ndarray:
    """Predict with a pre-fitted linear model bundle on a raw feature frame."""

    # Extract bundle components for deterministic preprocessing.
    features = bundle["features"]
    model = bundle["model"]
    impute_values = np.asarray(bundle["impute_values"], dtype=float)
    means = np.asarray(bundle["means"], dtype=float)
    stds = np.asarray(bundle["stds"], dtype=float)

    # Build the raw feature matrix and apply the same impute + standardize transforms.
    x_raw = frame.loc[:, features].to_numpy(dtype=float)
    x_imputed = np.where(np.isfinite(x_raw), x_raw, impute_values[None, :])
    x = (x_imputed - means[None, :]) / stds[None, :]
    pred = model.predict(x).astype(float)
    return pred


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
