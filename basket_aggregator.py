import numpy as np
import pandas as pd


def aggregate_day_to_basket(
    stock_day: pd.DataFrame,
    pred: np.ndarray,
    pred_col_name: str,
) -> pd.DataFrame:
    """Aggregate stock-level predictions into a basket-level minute series for one day."""

    # Attach prediction into the day panel for aggregation.
    day = stock_day.loc[:, ["date", "datetime", "weight", "label_stock_10m", "split"]].copy()
    day[pred_col_name] = pred.astype(float)

    # Compute weighted basket prediction and realized basket return label.
    day["w_pred"] = day["weight"] * day[pred_col_name]
    day["w_label"] = day["weight"] * day["label_stock_10m"]
    grouped = day.groupby(["date", "datetime", "split"], sort=True)
    basket = grouped.agg(
        basket_pred=("w_pred", "sum"),
        basket_label=("w_label", "sum"),
        total_weight=("weight", "sum"),
        covered_weight=(pred_col_name, lambda s: float(np.sum(day.loc[s.index, "weight"][np.isfinite(s.to_numpy())]))),
    )

    # Normalize by weight sums to be robust to missing predictions.
    basket["basket_pred"] = basket["basket_pred"] / basket["covered_weight"]
    basket["basket_label"] = basket["basket_label"] / basket["total_weight"]
    basket["weight_coverage_pred"] = basket["covered_weight"] / basket["total_weight"]
    basket = basket.reset_index()
    return basket.loc[:, ["date", "datetime", "split", "basket_pred", "basket_label", "weight_coverage_pred"]].copy()

