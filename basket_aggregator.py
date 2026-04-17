import numpy as np
import pandas as pd


def _apply_topk_by_weight(day: pd.DataFrame, top_k: int, weight_col: str) -> pd.DataFrame:
    """Filter a stock-day panel to top-k stocks by weight."""

    # Keep the top-k weight stocks for the day.
    weights = day.loc[:, ["stock_code", weight_col]].drop_duplicates(subset=["stock_code"], keep="first").copy()
    top_codes = weights.sort_values(weight_col, ascending=False).head(int(top_k))["stock_code"].astype(int).to_numpy()
    out = day.loc[day["stock_code"].astype(int).isin(top_codes)].copy()
    return out


def _compute_effective_weight(day: pd.DataFrame, weight_mode: str) -> pd.Series:
    """Compute a per-row effective weight series for aggregation."""

    # Compute effective weights according to the selected mode.
    if str(weight_mode) == "index":
        return day["weight"].astype(float)
    if str(weight_mode) == "weight_squared":
        w = day["weight"].astype(float)
        return w * w
    if str(weight_mode) == "weight_times_amount":
        return day["weight"].astype(float) * day["Amount"].astype(float)
    if str(weight_mode) == "effective_amount_weight":
        amount_ok = np.isfinite(day["Amount"].astype(float).to_numpy())
        return day["weight"].astype(float) * amount_ok.astype(float)
    raise ValueError(f"unknown weight_mode={weight_mode}")


def aggregate_day_to_basket_variant(
    stock_day: pd.DataFrame,
    pred: np.ndarray,
    pred_col_name: str,
    label_col: str,
    top_k_by_weight: int,
    weight_mode: str,
) -> pd.DataFrame:
    """Aggregate stock-level predictions into a basket minute series with top-k and weight variants."""

    # Attach prediction into the day panel for aggregation.
    day = stock_day.loc[:, ["date", "datetime", "stock_code", "weight", label_col, "split", "Amount"]].copy()
    day[pred_col_name] = pred.astype(float)

    # Filter to top-k weights when requested.
    if int(top_k_by_weight) > 0:
        day = _apply_topk_by_weight(day=day, top_k=int(top_k_by_weight), weight_col="weight")

    # Compute effective weight and weighted sums.
    day["w_eff"] = _compute_effective_weight(day=day, weight_mode=str(weight_mode)).astype(float)
    day["w_pred"] = day["w_eff"] * day[pred_col_name].astype(float)
    day["w_label"] = day["w_eff"] * day[label_col].astype(float)

    # Aggregate into minute-level basket series.
    grouped = day.groupby(["date", "datetime", "split"], sort=True)
    basket = grouped.agg(
        basket_pred=("w_pred", "sum"),
        basket_label=("w_label", "sum"),
        total_weight=("w_eff", "sum"),
        covered_weight=(pred_col_name, lambda s: float(np.sum(day.loc[s.index, "w_eff"][np.isfinite(s.to_numpy())]))),
    )

    # Normalize by weight sums to be robust to missing predictions.
    basket["basket_pred"] = basket["basket_pred"] / basket["covered_weight"]
    basket["basket_label"] = basket["basket_label"] / basket["total_weight"]
    basket["weight_coverage_pred"] = basket["covered_weight"] / basket["total_weight"]
    basket = basket.reset_index()
    return basket.loc[:, ["date", "datetime", "split", "basket_pred", "basket_label", "weight_coverage_pred"]].copy()


def aggregate_day_to_basket(
    stock_day: pd.DataFrame,
    pred: np.ndarray,
    pred_col_name: str,
) -> pd.DataFrame:
    """Aggregate stock-level predictions into a basket-level minute series for one day."""

    # Delegate to the general variant aggregator with baseline settings.
    return aggregate_day_to_basket_variant(
        stock_day=stock_day,
        pred=pred,
        pred_col_name=pred_col_name,
        label_col="label_stock_10m",
        top_k_by_weight=0,
        weight_mode="index",
    )
