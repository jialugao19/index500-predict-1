import pandas as pd
from lightgbm import LGBMRegressor, early_stopping, log_evaluation


def fit_lgbm_model(train: pd.DataFrame, val: pd.DataFrame, features: list[str], seed: int) -> LGBMRegressor:
    """Fit the LightGBM model for minute-level regression."""

    # Prepare dense numpy arrays as LightGBM inputs.
    x_train = train.loc[:, features].to_numpy()
    y_train = train.loc[:, "label"].to_numpy()
    x_val = val.loc[:, features].to_numpy()
    y_val = val.loc[:, "label"].to_numpy()

    # Train a compact but strong GBDT ensemble with early stopping.
    model = LGBMRegressor(
        n_estimators=8000,
        learning_rate=0.02,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=80,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=2.0,
        random_state=int(seed),
        n_jobs=8,
        objective="regression",
    )
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_val, y_val)],
        eval_metric="l2",
        callbacks=[early_stopping(stopping_rounds=100), log_evaluation(period=0)],
    )
    return model
