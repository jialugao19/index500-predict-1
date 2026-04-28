import numpy as np
import pandas as pd

from models.lgbm import fit_lgbm_model
from models.xgb import fit_xgb_model
from models.zscore import (
    fit_frame_zscore_stats,
    fit_series_zscore_stats,
    inverse_series_zscore,
    transform_frame_zscore,
    transform_series_zscore,
)


def build_component_basis_features_day(stock_day: pd.DataFrame) -> pd.DataFrame:
    """Build constituent-level basis features for one trading day."""

    # Prepare a clean stock panel ordered for cached factor return construction.
    day = stock_day.loc[
        :, ["date", "datetime", "split", "MinuteIndex", "stock_code", "weight", "Amount", "ret_1", "ret_5", "ret_10"]
    ].copy()
    day = day.sort_values(["stock_code", "datetime"], ascending=[True, True])
    grp = day.groupby("stock_code", sort=False)

    # Normalize stock return factor names into minute-horizon names.
    for window in [1, 5, 10]:
        ret_col = f"ret_{window}m"
        source_col = f"ret_{window}"
        day[ret_col] = day[source_col].astype(float)
        day[f"w_ret_{window}m"] = day["weight"].astype(float) * day[ret_col].astype(float)

    # Compute one-minute amount pressure as a liquidity-shock proxy.
    day["amount_chg_1m"] = day["Amount"].astype(float) / grp["Amount"].shift(1).astype(float) - 1.0
    day["w_amount_chg_1m"] = day["weight"].astype(float) * day["amount_chg_1m"].astype(float)
    day = day.replace([np.inf, -np.inf], np.nan)

    # Mark leader constituents by static index weight inside this day.
    stock_weights = day.loc[:, ["stock_code", "weight"]].drop_duplicates(subset=["stock_code"], keep="first")
    top20_codes = stock_weights.sort_values("weight", ascending=False).head(20)["stock_code"].astype(int).to_numpy()
    top50_codes = stock_weights.sort_values("weight", ascending=False).head(50)["stock_code"].astype(int).to_numpy()
    day["is_top20"] = day["stock_code"].astype(int).isin(top20_codes)
    day["is_top50"] = day["stock_code"].astype(int).isin(top50_codes)
    day["is_rest20"] = (~day["is_top20"]).astype(bool)

    # Aggregate broad basket state at each ETF timestamp.
    grouped = day.groupby(["date", "datetime", "split", "MinuteIndex"], sort=True)
    out = grouped.agg(total_weight=("weight", "sum"), stock_count=("stock_code", "nunique")).reset_index()

    # Attach weighted returns, dispersion, breadth, and synchronization by horizon.
    for window in [1, 5, 10]:
        ret_col = f"ret_{window}m"
        w_ret_col = f"w_ret_{window}m"
        grouped_ret = day.groupby(["date", "datetime", "split", "MinuteIndex"], sort=True)
        sum_weight = grouped_ret["weight"].sum()
        weighted_ret = grouped_ret[w_ret_col].sum(min_count=1) / sum_weight
        dispersion = grouped_ret[ret_col].std()
        weighted_abs_ret = (day["weight"].astype(float) * day[ret_col].astype(float).abs()).groupby(
            [day["date"], day["datetime"], day["split"], day["MinuteIndex"]], sort=True
        ).sum(min_count=1) / sum_weight
        pos_weight = day.loc[day[ret_col].astype(float) > 0.0].groupby(
            ["date", "datetime", "split", "MinuteIndex"], sort=True
        )["weight"].sum()

        # Merge each horizon feature back to the output frame.
        feature = pd.DataFrame(
            {
                "date": weighted_ret.index.get_level_values(0).astype(int),
                "datetime": weighted_ret.index.get_level_values(1),
                "split": weighted_ret.index.get_level_values(2).astype(str),
                "MinuteIndex": weighted_ret.index.get_level_values(3).astype(int),
                f"comp_w_ret_{window}m": weighted_ret.to_numpy(dtype=float),
                f"comp_xs_ret_std_{window}m": dispersion.reindex(weighted_ret.index).to_numpy(dtype=float),
                f"comp_breadth_pos_ret_{window}m": (pos_weight / sum_weight).reindex(weighted_ret.index).to_numpy(dtype=float),
                f"comp_sync_abs_ret_{window}m": (
                    weighted_ret.astype(float).abs() / weighted_abs_ret.astype(float)
                ).to_numpy(dtype=float),
            }
        )
        out = out.merge(feature, on=["date", "datetime", "split", "MinuteIndex"], how="left")

    # Add top-weight relative strength features.
    for flag_col, tag in [("is_top20", "top20"), ("is_top50", "top50"), ("is_rest20", "rest20")]:
        part = day.loc[day[flag_col].astype(bool)].copy()
        grouped_part = part.groupby(["date", "datetime", "split", "MinuteIndex"], sort=True)
        for window in [1, 5, 10]:
            rel = grouped_part[f"w_ret_{window}m"].sum(min_count=1) / grouped_part["weight"].sum()
            rel_frame = pd.DataFrame(
                {
                    "date": rel.index.get_level_values(0).astype(int),
                    "datetime": rel.index.get_level_values(1),
                    "split": rel.index.get_level_values(2).astype(str),
                    "MinuteIndex": rel.index.get_level_values(3).astype(int),
                    f"comp_{tag}_ret_{window}m": rel.to_numpy(dtype=float),
                }
            )
            out = out.merge(rel_frame, on=["date", "datetime", "split", "MinuteIndex"], how="left")

    # Compute coverage and leader-minus-basket spreads.
    valid_1m = np.isfinite(day["ret_1m"].to_numpy(dtype=float))
    covered_weight = day.loc[valid_1m].groupby(["date", "datetime", "split", "MinuteIndex"], sort=True)["weight"].sum()
    out_key = [out["date"], out["datetime"], out["split"], out["MinuteIndex"]]
    key_index = pd.MultiIndex.from_arrays(out_key, names=["date", "datetime", "split", "MinuteIndex"])
    out["comp_weight_coverage"] = (covered_weight / out.set_index(["date", "datetime", "split", "MinuteIndex"])["total_weight"]).reindex(key_index).to_numpy(dtype=float)
    out["comp_missing_rate"] = 1.0 - out["comp_weight_coverage"].astype(float)
    for window in [1, 5, 10]:
        out[f"comp_top20_minus_rest20_ret_{window}m"] = out[f"comp_top20_ret_{window}m"].astype(float) - out[
            f"comp_rest20_ret_{window}m"
        ].astype(float)
        out[f"comp_top50_minus_basket_ret_{window}m"] = out[f"comp_top50_ret_{window}m"].astype(float) - out[
            f"comp_w_ret_{window}m"
        ].astype(float)

    # Add weighted amount shock and keep stable output ordering.
    amount_grouped = day.groupby(["date", "datetime", "split", "MinuteIndex"], sort=True)
    amount_shock = amount_grouped["w_amount_chg_1m"].sum(min_count=1) / amount_grouped["weight"].sum()
    amount_frame = pd.DataFrame(
        {
            "date": amount_shock.index.get_level_values(0).astype(int),
            "datetime": amount_shock.index.get_level_values(1),
            "split": amount_shock.index.get_level_values(2).astype(str),
            "MinuteIndex": amount_shock.index.get_level_values(3).astype(int),
            "comp_w_amount_chg_1m": amount_shock.to_numpy(dtype=float),
        }
    )
    out = out.merge(amount_frame, on=["date", "datetime", "split", "MinuteIndex"], how="left")
    out["datetime"] = pd.to_datetime(out["datetime"]).astype("datetime64[us]")
    return out


def build_basis_model_frame(etf_dataset: pd.DataFrame, basket_pred: pd.DataFrame, component_features: pd.DataFrame) -> pd.DataFrame:
    """Build the second-stage basis modeling frame."""

    # Join ETF features, basket prediction, and constituent proxy features.
    basket = basket_pred.loc[:, ["date", "datetime", "split", "basket_pred", "weight_coverage_pred"]].copy()
    basket["datetime"] = pd.to_datetime(basket["datetime"]).astype("datetime64[us]")
    comp = component_features.copy()
    comp["datetime"] = pd.to_datetime(comp["datetime"]).astype("datetime64[us]")
    frame = etf_dataset.merge(basket, on=["date", "datetime", "split"], how="inner")
    frame = frame.merge(comp, on=["date", "datetime", "split", "MinuteIndex"], how="left")

    # Define the residual target with the requested sign convention.
    frame["basis_label"] = frame["label_etf_10m"].astype(float) - frame["basket_pred"].astype(float)
    frame["abs_basket_pred"] = frame["basket_pred"].astype(float).abs()

    # Build ETF premium proxies against the constituent spot proxy.
    for window in [1, 5, 10]:
        etf_col = f"ret_{window}m"
        comp_col = f"comp_w_ret_{window}m"
        if etf_col in frame.columns and comp_col in frame.columns:
            frame[f"premium_proxy_{window}m"] = frame[etf_col].astype(float) - frame[comp_col].astype(float)

    # Add intraday rolling state features using only current and past rows.
    frame = frame.sort_values(["date", "datetime"], ascending=[True, True]).reset_index(drop=True)
    grouped = frame.groupby("date", sort=False)
    for col in ["basket_pred", "premium_proxy_1m", "premium_proxy_5m", "comp_xs_ret_std_1m"]:
        if col in frame.columns:
            frame[f"{col}_roll_mean_20"] = grouped[col].transform(lambda series: series.astype(float).rolling(window=20, min_periods=5).mean())
            frame[f"{col}_roll_std_20"] = grouped[col].transform(lambda series: series.astype(float).rolling(window=20, min_periods=5).std())

    # Add cumulative ETF-vs-basket deviation from the market open.
    if "ret_1m" in frame.columns and "comp_w_ret_1m" in frame.columns:
        frame["etf_cum_ret_from_open"] = grouped["ret_1m"].transform(lambda series: series.astype(float).fillna(0.0).cumsum())
        frame["basket_cum_ret_from_open"] = grouped["comp_w_ret_1m"].transform(lambda series: series.astype(float).fillna(0.0).cumsum())
        frame["intraday_premium_proxy"] = frame["etf_cum_ret_from_open"].astype(float) - frame["basket_cum_ret_from_open"].astype(float)

    # Return a frame sorted for deterministic downstream training.
    frame = frame.replace([np.inf, -np.inf], np.nan)
    return frame


def select_basis_feature_columns(frame: pd.DataFrame) -> list[str]:
    """Select numeric feature columns for the basis model."""

    # Exclude identifiers, labels, and raw price levels from model features.
    excluded = {
        "date",
        "datetime",
        "split",
        "label_etf_10m",
        "basis_label",
        "close",
        "DateTime",
        "Date",
    }

    # Keep numeric columns only so XGB and LightGBM receive a stable matrix.
    feature_cols: list[str] = []
    for col in frame.columns:
        if str(col) in excluded:
            continue
        if pd.api.types.is_numeric_dtype(frame[col]):
            feature_cols.append(str(col))
    assert len(feature_cols) > 0
    return feature_cols


def fit_basis_model_bundle(train_frame: pd.DataFrame, feature_cols: list[str], seed: int) -> dict:
    """Fit XGB and LightGBM models for the ETF-basket residual."""

    # Keep finite labels and split the OOF training frame by latest month.
    sample = train_frame.loc[train_frame["split"].astype(str) == "train"].copy()
    sample = sample.loc[np.isfinite(sample["basis_label"].astype(float).to_numpy(dtype=float))].copy()
    sample = sample.rename(columns={"basis_label": "label"})
    sample["month"] = (sample["date"].astype(int) // 100).astype(int)
    val_month = int(sample["month"].max())
    val = sample.loc[sample["month"] == val_month].copy()
    train_fit = sample.loc[sample["month"] < val_month].copy()
    assert len(train_fit) > 0
    assert len(val) > 0

    # Standardize features and residual labels using the training fold only.
    feature_stats = fit_frame_zscore_stats(frame=train_fit, columns=feature_cols)
    feature_cols_kept = list(feature_stats.columns)
    label_stats = fit_series_zscore_stats(series=train_fit["label"], name="basis_label")
    train_fit = transform_frame_zscore(frame=train_fit, stats=feature_stats)
    val = transform_frame_zscore(frame=val, stats=feature_stats)
    train_fit["label"] = transform_series_zscore(series=train_fit["label"], stats=label_stats)
    val["label"] = transform_series_zscore(series=val["label"], stats=label_stats)

    # Fit the two allowed residual model families.
    xgb_model = fit_xgb_model(train=train_fit, val=val, features=feature_cols_kept, seed=seed)
    lgbm_model = fit_lgbm_model(train=train_fit, val=val, features=feature_cols_kept, seed=seed)

    # Return all artifacts needed for deterministic prediction.
    return {
        "xgb_model": xgb_model,
        "lgbm_model": lgbm_model,
        "feature_stats": feature_stats,
        "label_stats": label_stats,
        "feature_cols": feature_cols_kept,
        "train_rows": int(len(train_fit)),
        "val_rows": int(len(val)),
        "val_month": int(val_month),
        "xgb_importance": [
            {"feature": str(name), "importance": float(value)}
            for name, value in zip(feature_cols_kept, xgb_model.feature_importances_)
        ],
        "lgbm_importance": [
            {"feature": str(name), "importance": float(value)}
            for name, value in zip(feature_cols_kept, lgbm_model.feature_importances_)
        ],
    }


def predict_basis_model_bundle(frame: pd.DataFrame, bundle: dict) -> dict[str, np.ndarray]:
    """Predict residual basis values from a fitted basis-model bundle."""

    # Standardize prediction features with the fitted basis feature stats.
    feature_cols = list(bundle["feature_cols"])
    feature_stats = bundle["feature_stats"]
    features = transform_frame_zscore(frame=frame.loc[:, feature_cols], stats=feature_stats)

    # Predict residuals and map them back into raw return units.
    xgb_pred_z = bundle["xgb_model"].predict(features.to_numpy())
    lgbm_pred_z = bundle["lgbm_model"].predict(features.to_numpy())
    xgb_pred = inverse_series_zscore(values=xgb_pred_z, stats=bundle["label_stats"])
    lgbm_pred = inverse_series_zscore(values=lgbm_pred_z, stats=bundle["label_stats"])
    return {"basis_xgb": xgb_pred.astype(float), "basis_lgbm": lgbm_pred.astype(float)}
