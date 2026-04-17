import datetime as dt
import os
import sys

import numpy as np
import pandas as pd
import yaml
from matplotlib import pyplot as plt

# Add repo root into sys.path so local modules can be imported when running as a script.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from basket_aggregator import aggregate_day_to_basket_variant  # noqa: E402
from eval.metrics import compute_metrics  # noqa: E402
from models.zscore import (  # noqa: E402
    fit_frame_zscore_stats,
    fit_series_zscore_stats,
    inverse_series_zscore,
    transform_frame_zscore,
    transform_series_zscore,
)
from models.xgb import fit_xgb_model  # noqa: E402
from models.lgbm import fit_lgbm_model  # noqa: E402
from pipeline import build_etf_minute_dataset, filter_dates_by_range, list_available_dates_from_etf1m_dir  # noqa: E402
from stock_panel_loader import (  # noqa: E402
    get_stock_feature_cols,
    load_index_weights,
    load_or_build_stock_panel_day,
)


def _ensure_dir(path: str) -> None:
    """Ensure a directory exists."""

    # Create directory recursively.
    os.makedirs(path, exist_ok=True)


def _write_yaml(path: str, obj: dict) -> None:
    """Write a Python dict as YAML."""

    # Serialize as stable YAML for reproducibility.
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, allow_unicode=True, sort_keys=False)


def _plot_topk_curve(rows: list[dict], out_path: str, title: str) -> None:
    """Plot IC vs top-k for quick visualization."""

    # Convert rows into numeric arrays for plotting.
    ks = [int(r["top_k"]) for r in rows]
    ic = [float(r["etf_test_ic"]) for r in rows]
    ric = [float(r["etf_test_rank_ic"]) for r in rows]

    # Draw a compact two-line chart.
    plt.figure(figsize=(8, 4))
    plt.plot(ks, ic, marker="o", linewidth=1.2, label="IC")
    plt.plot(ks, ric, marker="o", linewidth=1.2, label="RankIC")
    plt.title(title)
    plt.xlabel("top_k by index weight")
    plt.ylabel("metric (test)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _build_pred_table_from_join(joined: pd.DataFrame, model_name: str, pred_col: str, label_col: str) -> pd.DataFrame:
    """Build a standardized prediction table for compute_metrics."""

    # Assemble the minimal columns required by compute_metrics.
    out = pd.DataFrame(
        {
            "datetime": pd.to_datetime(joined["datetime"]).astype("datetime64[ns]"),
            "date": joined["date"].astype(int),
            "pred": joined[pred_col].astype(float),
            "label": joined[label_col].astype(float),
            "split": joined["split"].astype(str),
            "model_name": str(model_name),
        }
    )
    return out


def main() -> None:
    """Run experiments A/C and write a concentration research report."""

    # Define the experiment configuration explicitly to match the pipeline defaults.
    seed = 42
    etf_code_int = 510500
    horizon_minutes = 30
    train_start = 20210101
    train_end = 20231231
    test_start = 20240201
    test_end = 20251231
    factor_set_name = "stock_all"
    etf_factor_set_name = "etf_all"

    # Define data locations consistent with pipeline.
    weight_path = "/data/ashare/market/index_weight/000905.feather"
    etf1m_root = "/data/ashare/market/etf1m"
    stock1m_root = "/data/ashare/market/stock1m"
    specs_root = os.path.join(REPO_ROOT, "features", "specs")
    stock_panel_base_cache_root = f"/data-cache/index500-predict/stock_base_days_v1__h{horizon_minutes}"
    stock_panel_feature_cache_root = f"/data-cache/index500-predict/stock_feature_days_v2__{factor_set_name}__h{horizon_minutes}"

    # Prepare output directory under report/<mmdd>/.
    day_tag = dt.datetime.now().strftime("%m%d")
    run_id = dt.datetime.now().strftime("exp_ac_concentration_%Y%m%d_%H%M%S")
    out_dir = os.path.join(REPO_ROOT, "report", day_tag, run_id)
    _ensure_dir(out_dir)

    # Discover available dates and keep only those needed for train/test.
    all_dates = list_available_dates_from_etf1m_dir(etf1m_dir=etf1m_root)
    train_dates = filter_dates_by_range(all_dates=all_dates, start_date=train_start, end_date=train_end)
    test_dates = filter_dates_by_range(all_dates=all_dates, start_date=test_start, end_date=test_end)
    dates_needed = list(train_dates) + list(test_dates)

    # Load weights and stock feature columns.
    weights = load_index_weights(weight_path=weight_path)
    stock_feature_cols = get_stock_feature_cols(specs_root=specs_root, factor_set_name=factor_set_name)

    # Collect a deterministic training sample using the cached day panels.
    pieces: list[pd.DataFrame] = []
    for date in train_dates:
        # Load cached day panel and keep train split only.
        day = load_or_build_stock_panel_day(
            date=int(date),
            weights=weights,
            stock1m_root=stock1m_root,
            horizon_minutes=horizon_minutes,
            base_cache_root=stock_panel_base_cache_root,
            feature_cache_root=stock_panel_feature_cache_root,
            specs_root=specs_root,
            factor_set_name=factor_set_name,
        )
        day = day.loc[day["split"] == "train"].copy()
        day = day.dropna(subset=["label_stock_10m"])

        # Build a deterministic hash-based sample mask for reproducibility.
        key = (
            day["date"].astype(np.int64) * 1_000_003
            + day["stock_code"].astype(np.int64) * 10_007
            + day["MinuteIndex"].astype(np.int64) * 10_009
        )
        mask = (key % 50).astype(int) == 0
        sampled = day.loc[mask, ["date", "stock_code", "MinuteIndex", "label_stock_10m"] + stock_feature_cols].copy()
        pieces.append(sampled)

    # Split the sampled training frame into train_fit/val by last month.
    train_sample = pd.concat(pieces, axis=0, ignore_index=True)
    train_sample = train_sample.rename(columns={"label_stock_10m": "label"})
    train_sample["month"] = (train_sample["date"].astype(int) // 100).astype(int)
    val_month = int(train_sample["month"].max())
    val = train_sample.loc[train_sample["month"] == val_month].copy()
    train_fit = train_sample.loc[train_sample["month"] < val_month].copy()

    # Fit z-score stats and standardize features/label for model training.
    stock_feature_stats = fit_frame_zscore_stats(frame=train_fit, columns=stock_feature_cols)
    stock_feature_cols = list(stock_feature_stats.columns)
    stock_label_stats = fit_series_zscore_stats(series=train_fit["label"], name="label_stock_30m")
    train_fit = transform_frame_zscore(frame=train_fit, stats=stock_feature_stats)
    val = transform_frame_zscore(frame=val, stats=stock_feature_stats)
    train_fit["label"] = transform_series_zscore(series=train_fit["label"], stats=stock_label_stats)
    val["label"] = transform_series_zscore(series=val["label"], stats=stock_label_stats)

    # Train stock models (xgb/lgbm) on the standardized samples.
    xgb_model = fit_xgb_model(train=train_fit, val=val, features=stock_feature_cols, seed=seed)
    lgbm_model = fit_lgbm_model(train=train_fit, val=val, features=stock_feature_cols, seed=seed)

    # Build ETF dataset for test for like-for-like joins.
    etf_dataset = build_etf_minute_dataset(
        dates=list(test_dates),
        etf1m_root=etf1m_root,
        etf_code_int=etf_code_int,
        horizon_minutes=horizon_minutes,
        specs_root=specs_root,
        factor_set_name=etf_factor_set_name,
    )
    etf_dataset = etf_dataset.loc[etf_dataset["split"] == "test"].copy()

    # Define experiment A/C variant specs.
    topk_list = [10, 20, 50, 100, 200]
    topk_specs = [{"tag": f"topw{k}", "top_k_by_weight": int(k), "weight_mode": "index"} for k in topk_list]
    weight_specs = [
        {"tag": "w2", "top_k_by_weight": 0, "weight_mode": "weight_squared"},
        {"tag": "w_amount", "top_k_by_weight": 0, "weight_mode": "weight_times_amount"},
        {"tag": "w_effamt", "top_k_by_weight": 0, "weight_mode": "effective_amount_weight"},
    ]
    variant_specs = topk_specs + weight_specs

    # Generate basket series for test dates only to keep the experiment focused.
    topk_weight_share_sum: dict[int, float] = {int(k): 0.0 for k in topk_list}
    topk_weight_share_n: dict[int, int] = {int(k): 0 for k in topk_list}
    basket_rows: list[pd.DataFrame] = []
    for date in test_dates:
        # Load cached day panel and keep test rows only.
        day = load_or_build_stock_panel_day(
            date=int(date),
            weights=weights,
            stock1m_root=stock1m_root,
            horizon_minutes=horizon_minutes,
            base_cache_root=stock_panel_base_cache_root,
            feature_cache_root=stock_panel_feature_cache_root,
            specs_root=specs_root,
            factor_set_name=factor_set_name,
        )
        day = day.loc[day["split"] == "test"].copy()
        day = day.dropna(subset=["label_stock_10m"])

        # Compute daily top-k weight share diagnostics from static index weights.
        weights_day = day.loc[:, ["stock_code", "weight"]].drop_duplicates(subset=["stock_code"], keep="first").copy()
        weights_day = weights_day.sort_values("weight", ascending=False)
        total_weight = float(weights_day["weight"].astype(float).sum())
        for k in topk_list:
            top_weight = float(weights_day.head(int(k))["weight"].astype(float).sum())
            topk_weight_share_sum[int(k)] += float(top_weight / total_weight)
            topk_weight_share_n[int(k)] += 1

        # Predict stock returns using the trained models.
        day_features = transform_frame_zscore(frame=day.loc[:, stock_feature_cols], stats=stock_feature_stats)
        xgb_pred_z = xgb_model.predict(day_features.to_numpy())
        lgbm_pred_z = lgbm_model.predict(day_features.to_numpy())
        xgb_pred = inverse_series_zscore(values=xgb_pred_z, stats=stock_label_stats)
        lgbm_pred = inverse_series_zscore(values=lgbm_pred_z, stats=stock_label_stats)

        # Aggregate baseline baskets.
        basket_xgb = aggregate_day_to_basket_variant(
            stock_day=day,
            pred=xgb_pred,
            pred_col_name="pred_stock_xgb",
            label_col="label_stock_10m",
            top_k_by_weight=0,
            weight_mode="index",
        )
        basket_xgb["model_name"] = "basket_stock_xgb"
        basket_lgbm = aggregate_day_to_basket_variant(
            stock_day=day,
            pred=lgbm_pred,
            pred_col_name="pred_stock_lgbm",
            label_col="label_stock_10m",
            top_k_by_weight=0,
            weight_mode="index",
        )
        basket_lgbm["model_name"] = "basket_stock_lgbm"
        basket_rows.append(pd.concat([basket_xgb, basket_lgbm], axis=0, ignore_index=True))

        # Aggregate variant baskets for both models.
        for spec in variant_specs:
            tag = str(spec["tag"])
            top_k_by_weight = int(spec["top_k_by_weight"])
            weight_mode = str(spec["weight_mode"])

            basket_xgb_var = aggregate_day_to_basket_variant(
                stock_day=day,
                pred=xgb_pred,
                pred_col_name="pred_stock_xgb",
                label_col="label_stock_10m",
                top_k_by_weight=top_k_by_weight,
                weight_mode=weight_mode,
            )
            basket_xgb_var["model_name"] = f"basket_{tag}_stock_xgb"
            basket_lgbm_var = aggregate_day_to_basket_variant(
                stock_day=day,
                pred=lgbm_pred,
                pred_col_name="pred_stock_lgbm",
                label_col="label_stock_10m",
                top_k_by_weight=top_k_by_weight,
                weight_mode=weight_mode,
            )
            basket_lgbm_var["model_name"] = f"basket_{tag}_stock_lgbm"
            basket_rows.append(pd.concat([basket_xgb_var, basket_lgbm_var], axis=0, ignore_index=True))

    # Evaluate basket_pred vs basket_label (basket synthesis).
    basket_all = pd.concat(basket_rows, axis=0, ignore_index=True)
    basket_pred_table = pd.DataFrame(
        {
            "datetime": pd.to_datetime(basket_all["datetime"]).astype("datetime64[ns]"),
            "date": basket_all["date"].astype(int),
            "pred": basket_all["basket_pred"].astype(float),
            "label": basket_all["basket_label"].astype(float),
            "split": basket_all["split"].astype(str),
            "model_name": basket_all["model_name"].astype(str),
        }
    )
    basket_metrics = compute_metrics(pred_table=basket_pred_table)

    # Join each basket series with ETF label and evaluate ETF-level metrics.
    etf_pred_tables: list[pd.DataFrame] = []
    for model_name, part in basket_all.groupby("model_name", sort=True):
        # Join basket_pred to ETF label for like-for-like evaluation.
        basket_pred = part.loc[:, ["date", "datetime", "split", "basket_pred"]].copy()
        joined = etf_dataset.merge(basket_pred, on=["date", "datetime", "split"], how="inner")
        raw_model_name = f"raw_{str(model_name)}_vs_etf"
        etf_pred_tables.append(_build_pred_table_from_join(joined=joined, model_name=raw_model_name, pred_col="basket_pred", label_col="label_etf_10m"))
    etf_pred_table = pd.concat(etf_pred_tables, axis=0, ignore_index=True)
    etf_metrics = compute_metrics(pred_table=etf_pred_table)

    # Summarize experiment A: top-k weight.
    topk_rows: list[dict] = []
    for k in topk_list:
        model_name = f"raw_basket_topw{int(k)}_stock_lgbm_vs_etf"
        row = etf_metrics["overall"][model_name]["test"]
        weight_share_mean = float(topk_weight_share_sum[int(k)] / topk_weight_share_n[int(k)])
        topk_rows.append(
            {
                "top_k": int(k),
                "model_name": model_name,
                "topk_weight_share_mean": weight_share_mean,
                "etf_test_ic": float(row["ic"]),
                "etf_test_rank_ic": float(row["rank_ic"]),
                "etf_test_rmse": float(row["rmse"]),
                "etf_test_mae": float(row["mae"]),
                "n": int(row["n"]),
            }
        )

    # Summarize experiment C: alternative weights.
    weight_rows: list[dict] = []
    for tag in ["w2", "w_amount", "w_effamt"]:
        model_name = f"raw_basket_{tag}_stock_lgbm_vs_etf"
        row = etf_metrics["overall"][model_name]["test"]
        weight_rows.append(
            {
                "tag": str(tag),
                "model_name": model_name,
                "etf_test_ic": float(row["ic"]),
                "etf_test_rank_ic": float(row["rank_ic"]),
                "etf_test_rmse": float(row["rmse"]),
                "etf_test_mae": float(row["mae"]),
                "n": int(row["n"]),
            }
        )

    # Plot top-k curves for quick eyeballing.
    _plot_topk_curve(rows=topk_rows, out_path=os.path.join(out_dir, "fig_topk_etf_ic_curve.png"), title="ETF IC vs Top-k Weight (lgbm, test)")

    # Write parquet artifacts for inspection.
    basket_all.to_parquet(os.path.join(out_dir, "basket_variants_test.parquet"), index=False)
    etf_pred_table.to_parquet(os.path.join(out_dir, "etf_pred_variants_test.parquet"), index=False)

    # Write YAML metrics payload.
    payload = {
        "config": {
            "seed": int(seed),
            "etf_code_int": int(etf_code_int),
            "label_horizon_minutes": int(horizon_minutes),
            "train_range": [int(train_start), int(train_end)],
            "test_start": int(test_start),
            "test_end": int(test_end),
            "used_train_days": int(len(train_dates)),
            "used_test_days": int(len(test_dates)),
            "note": "This experiment evaluates basket/ETF on test only; stock models are trained on train split.",
        },
        "variants": {"topk_by_weight": topk_specs, "alt_weights": weight_specs},
        "diagnostics": {
            "topk_weight_share_mean": {
                str(k): float(topk_weight_share_sum[int(k)] / topk_weight_share_n[int(k)]) for k in topk_list
            }
        },
        "basket_metrics_overall": basket_metrics["overall"],
        "etf_metrics_overall": etf_metrics["overall"],
        "expA_topk_summary_lgbm": topk_rows,
        "expC_weight_summary_lgbm": weight_rows,
        "artifacts": {
            "basket_variants_test_parquet": "basket_variants_test.parquet",
            "etf_pred_variants_test_parquet": "etf_pred_variants_test.parquet",
            "fig_topk_etf_ic_curve": "fig_topk_etf_ic_curve.png",
        },
    }
    _write_yaml(os.path.join(out_dir, "concentration_metrics.yaml"), payload)

    # Write a concise markdown report in the project's research style.
    lines: list[str] = []
    lines.append(f"# 研究报告: 高权重股贡献集中度 (ExpA/ExpC, horizon={int(horizon_minutes)}m)")
    lines.append("")
    lines.append(f"- out_dir: `{out_dir}`")
    lines.append(f"- ETF: `{int(etf_code_int)}`; horizon: `{int(horizon_minutes)}m`.")
    lines.append(f"- Train days: `{int(len(train_dates))}`; Test days: `{int(len(test_dates))}`.")
    lines.append("")
    lines.append("## 方法 (Method)")
    lines.append("")
    lines.append("- ExpA: 用 top-k 权重股合成 `basket_pred`/`basket_label`, 评估 `basket_pred -> etf_label` 的 IC。")
    lines.append("- ExpC: 用替代权重(权重平方, 权重×成交额, 有效成交权重)替代静态指数权重, 评估 IC 变化。")
    lines.append("")
    lines.append("## 结果 (ETF-level, test, lgbm)")
    lines.append("")
    lines.append("### ExpA: Top-k by weight")
    lines.append("")
    lines.append("| top_k | topk_weight_share_mean | etf_ic | etf_rank_ic | rmse | mae | n |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for row in topk_rows:
        lines.append(
            f"| {int(row['top_k'])} | {float(row['topk_weight_share_mean']):.6f} | {float(row['etf_test_ic']):.6f} | {float(row['etf_test_rank_ic']):.6f} | {float(row['etf_test_rmse']):.6f} | {float(row['etf_test_mae']):.6f} | {int(row['n'])} |"
        )
    lines.append("")
    lines.append("![ETF IC vs Top-k](fig_topk_etf_ic_curve.png)")
    lines.append("")
    lines.append("### ExpC: Alternative weights")
    lines.append("")
    lines.append("| tag | etf_ic | etf_rank_ic | rmse | mae | n |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for row in weight_rows:
        lines.append(
            f"| {row['tag']} | {float(row['etf_test_ic']):.6f} | {float(row['etf_test_rank_ic']):.6f} | {float(row['etf_test_rmse']):.6f} | {float(row['etf_test_mae']):.6f} | {int(row['n'])} |"
        )
    lines.append("")
    lines.append("## 结论 (Conclusion)")
    lines.append("")
    lines.append("- 若 top-k 的 IC/RankIC 明显高于全量, 说明短周期信号贡献集中在头部权重, 静态全成分加权会稀释 alpha。")
    lines.append("- 若替代权重(如权重×成交额)提升 IC, 说明流动性在短周期驱动中占比更高, 需要动态权重刻画有效交易权重。")
    lines.append("")
    with open(os.path.join(out_dir, "concentration_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[INFO] Done. out_dir={out_dir}")


if __name__ == "__main__":
    main()
