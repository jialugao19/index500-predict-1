import numpy as np
import pandas as pd
from xgboost import XGBRegressor


def fit_xgb_model(train: pd.DataFrame, val: pd.DataFrame, features: list[str], seed: int) -> XGBRegressor:
    """Fit the XGBoost model for minute-level regression."""

    # Prepare dense numpy arrays as XGBoost inputs.
    x_train = train[features].to_numpy()
    y_train = train["label"].to_numpy()
    x_val = val[features].to_numpy()
    y_val = val["label"].to_numpy()

    # Train a small but strong tree ensemble for minute-level regression.
    model = XGBRegressor(
        n_estimators=1500,
        learning_rate=0.03,
        max_depth=2,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=3.0,
        random_state=seed,
        n_jobs=8,
        objective="reg:squarederror",
        eval_metric="rmse",
        early_stopping_rounds=50,
    )
    model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)
    return model

