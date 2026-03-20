import pandas as pd
from sklearn.linear_model import LinearRegression


def fit_basis_linear_model(train: pd.DataFrame, features: list[str], label_col: str) -> LinearRegression:
    """Fit a lightweight linear basis model on residual targets."""

    # Select rows with no NaNs in the feature set.
    mask = train[features].notna().all(axis=1)
    x_train = train.loc[mask, features].to_numpy()
    y_train = train.loc[mask, label_col].to_numpy()

    # Fit linear regression with intercept.
    model = LinearRegression()
    model.fit(x_train, y_train)
    return model

