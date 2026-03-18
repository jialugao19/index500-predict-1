import numpy as np
import pandas as pd


def compute_component_features(stock_day: pd.DataFrame, day_weights: pd.DataFrame) -> pd.DataFrame:
    """Compute weighted aggregated features from constituent stock minute bars."""

    # Compute per-stock 1m returns and 1m volume changes.
    stock_day = stock_day.copy()
    stock_day["ret_1m"] = stock_day["Close"] / stock_day.groupby("StockCode")["Close"].shift(1) - 1.0
    stock_day["vol_chg_1m"] = stock_day["Vol"] / stock_day.groupby("StockCode")["Vol"].shift(1) - 1.0
    # Replace inf values from zero volumes with NaN.
    stock_day = stock_day.replace([np.inf, -np.inf], np.nan)

    # Join weights so each stock row has its index weight.
    merged = stock_day.merge(day_weights, left_on="StockCode", right_on="con_int", how="inner")

    # Aggregate into per-minute features using current-minute info only.
    merged["w_ret"] = merged["weight_frac"] * merged["ret_1m"]
    merged["w_vol_chg"] = merged["weight_frac"] * merged["vol_chg_1m"]

    # Compute per-minute coverage diagnostics relative to the full index weights.
    total_weight = float(day_weights["weight_frac"].sum())
    total_constituents = int(day_weights["con_int"].nunique())
    valid_mask = np.isfinite(merged["ret_1m"].to_numpy())
    covered_weight = (
        merged.loc[valid_mask, ["DateTime", "weight_frac"]].groupby("DateTime", sort=True)["weight_frac"].sum(min_count=1)
    )
    covered_count = merged.loc[valid_mask, ["DateTime", "con_int"]].groupby("DateTime", sort=True)["con_int"].nunique()
    pos_mask = valid_mask & (merged["ret_1m"].to_numpy() > 0.0)
    pos_weight = (
        merged.loc[pos_mask, ["DateTime", "weight_frac"]].groupby("DateTime", sort=True)["weight_frac"].sum(min_count=1)
    )

    # Compute weighted mean and cross-sectional std at each DateTime.
    grouped = merged.groupby("DateTime", sort=True)
    weighted_ret = grouped["w_ret"].sum(min_count=1) / grouped["weight_frac"].sum(min_count=1)
    weighted_vol_chg = grouped["w_vol_chg"].sum(min_count=1) / grouped["weight_frac"].sum(min_count=1)
    xs_ret_std = grouped["ret_1m"].std()

    # Pack into a DataFrame aligned by DateTime.
    comp = pd.DataFrame(
        {
            "DateTime": weighted_ret.index,
            "comp_w_ret_1m": weighted_ret.values,
            "comp_w_vol_chg_1m": weighted_vol_chg.values,
            "comp_xs_ret_std_1m": xs_ret_std.values,
            "comp_weight_coverage": (covered_weight / total_weight).reindex(weighted_ret.index).to_numpy(),
            "comp_missing_rate": (1.0 - covered_count / total_constituents).reindex(weighted_ret.index).to_numpy(),
            "comp_breadth_pos_ret_1m": (pos_weight / total_weight).reindex(weighted_ret.index).to_numpy(),
        }
    )
    return comp

