import os

import numpy as np
import pandas as pd
import yaml


def write_yaml(path: str, obj: dict) -> None:
    """Write a Python dict as YAML."""

    # Use safe_dump for plain YAML output.
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, allow_unicode=True, sort_keys=False)


def write_summary_md(report_dir: str, run_id: str, metrics: dict, feature_cols: list[str]) -> None:
    """Write a concise summary markdown."""

    # Summarize key overall metrics for test split.
    lines: list[str] = []
    lines.append(f"# 摘要 ({run_id})")
    lines.append("")
    lines.append("## 模型 (test split)")
    for model_name in sorted(metrics["overall"].keys()):
        row = metrics["overall"][model_name].get("test", {})
        if len(row) == 0:
            continue
        lines.append(
            f"- {model_name}: IC={row['ic']:.4f}, RankIC={row['rank_ic']:.4f}, "
            f"DirAcc={row['direction_acc']:.4f}, RMSE={row['rmse']:.6f}, MAE={row['mae']:.6f}, n={row['n']}"
        )
    lines.append("")
    lines.append("## 特征 (Features)")
    for col in feature_cols:
        lines.append(f"- {col}")
    lines.append("")

    # Write to file.
    out_path = os.path.join(report_dir, "summary.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def compute_coverage_audit_tables(dataset: pd.DataFrame) -> dict:
    """Compute coverage and join-retention tables for the data audit report."""

    # Build a minute-of-day table for coverage and missingness diagnostics.
    minute_index_int = dataset["minute_index"].astype(int)
    per_minute = dataset.copy()
    per_minute["comp_weight_coverage_filled"] = per_minute["comp_weight_coverage"].fillna(0.0)
    per_minute["comp_missing_rate_filled"] = per_minute["comp_missing_rate"].fillna(1.0)
    per_minute["comp_breadth_pos_ret_1m_filled"] = per_minute["comp_breadth_pos_ret_1m"].fillna(0.0)
    per_minute["join_retained"] = per_minute["comp_w_ret_1m"].notna().astype(float)
    minute_table = (
        per_minute.groupby(minute_index_int, sort=True)
        .agg(
            weight_coverage=("comp_weight_coverage_filled", "mean"),
            missing_rate=("comp_missing_rate_filled", "mean"),
            breadth_pos=("comp_breadth_pos_ret_1m_filled", "mean"),
            join_retention=("join_retained", "mean"),
            n=("label", "size"),
        )
        .reset_index()
        .rename(columns={"minute_index": "minute_index"})
    )

    # Compute first/last minute availability across dates.
    first_minutes = dataset.loc[minute_index_int == 0].copy()
    last_minute_index = dataset.groupby("date", sort=False)["minute_index"].transform("max").astype(int)
    last_minutes = dataset.loc[minute_index_int == last_minute_index].copy()
    first_stats = {
        "weight_coverage_mean": float(first_minutes["comp_weight_coverage"].fillna(0.0).mean()),
        "missing_rate_mean": float(first_minutes["comp_missing_rate"].fillna(1.0).mean()),
        "join_retention_mean": float(first_minutes["comp_w_ret_1m"].notna().mean()),
        "n": int(len(first_minutes)),
    }
    last_stats = {
        "weight_coverage_mean": float(last_minutes["comp_weight_coverage"].fillna(0.0).mean()),
        "missing_rate_mean": float(last_minutes["comp_missing_rate"].fillna(1.0).mean()),
        "join_retention_mean": float(last_minutes["comp_w_ret_1m"].notna().mean()),
        "n": int(len(last_minutes)),
    }

    # Compute overall join retention on the final modeling dataset.
    overall = {
        "rows": int(len(dataset)),
        "join_retention_mean": float(dataset["comp_w_ret_1m"].notna().mean()),
    }

    # Return plain Python structures to keep the markdown writer simple.
    return {
        "minute_table": minute_table.to_dict(orient="records"),
        "first_minute": first_stats,
        "last_minute": last_stats,
        "overall": overall,
    }


def write_data_audit_md(
    report_dir: str,
    spot_checks: list[dict],
    horizon_minutes: int,
    coverage_audit: dict,
) -> None:
    """Write data audit markdown with spot checks and alignment rules."""

    # Compose audit content with required sections.
    lines: list[str] = []
    lines.append("# 数据审计 (Data Audit)")
    lines.append("")
    lines.append("## 标签定义 (Label)")
    lines.append("")
    lines.append(f"`label_t = Close_{{t+{int(horizon_minutes)}}} / Close_t - 1`")
    lines.append("")
    lines.append("## Baseline 泄漏检查 (Leakage Check)")
    lines.append("")
    shift_n = int(horizon_minutes) + 1
    lines.append(f"- Horizon minutes: `{int(horizon_minutes)}`")
    lines.append(f"- Baseline lag shift_n: `{shift_n}` (enforced `shift_n > horizon_minutes`)")
    lines.append("- Baseline uses `label_{t-shift_n}`; the latest price inside that label is `Close_{t-shift_n+horizon}`.")
    lines.append("- Because `shift_n > horizon`, `t-shift_n+horizon <= t-1`, so baseline cannot include prices after `t`.")
    lines.append("")
    lines.append("## 特征可用时间 (Feature Timing)")
    lines.append("")
    lines.append("- ETF 历史收益特征只使用 `Close_t` 和历史 `Close_{t-k}`.")
    lines.append("- 滚动成交量/成交额统计只使用截至 `t` 的 rolling window.")
    lines.append("- 成分股特征只使用同一分钟 `t` 的 `Close_t`/`Close_{t-1}` 与 `Vol_t`/`Vol_{t-1}`.")
    lines.append("")
    lines.append("## 对齐规则 (Alignment)")
    lines.append("")
    lines.append("- 所有特征都对齐到 ETF 的 `DateTime`.")
    lines.append("- Component features are aggregated by `DateTime` and left-joined to ETF minutes.")
    lines.append("- 因未来分钟不足导致 `label` 缺失的行会被丢弃.")
    lines.append("")
    lines.append("## 缺失值处理 (Missing Values)")
    lines.append("")
    lines.append("- 由 rolling window 或开盘首分钟导致的特征 NaN 会保留, 不做强行填充.")
    lines.append("- `linear_model` baseline 会在其最小特征集合上丢弃含 NaN 的行.")
    lines.append("")
    lines.append("## 覆盖率分析 (Coverage)")
    lines.append("")
    lines.append("- Definitions:")
    lines.append("  - 有效权重覆盖率: 在分钟 `t` 上, `ret_1m` 有效的成分股权重之和 / 当日总权重.")
    lines.append("  - 成分股缺失率: 在分钟 `t` 上, `1 - (available_constituents / total_constituents)`.")
    lines.append("  - Join 后样本保留率: `comp_w_ret_1m` 非 NaN 的 ETF 分钟占比.")
    lines.append("")
    lines.append("### 样本保留率 (Join Retention, Overall)")
    lines.append("")
    lines.append(f"- rows: `{coverage_audit['overall']['rows']}`")
    lines.append(f"- join_retention_mean: `{coverage_audit['overall']['join_retention_mean']:.6f}`")
    lines.append("")
    lines.append("### 首分钟/尾分钟可用性 (First/Last Minute)")
    lines.append("")
    lines.append("| minute | weight_coverage_mean | missing_rate_mean | join_retention_mean | n |")
    lines.append("| --- | --- | --- | --- | --- |")
    lines.append(
        f"| first | {coverage_audit['first_minute']['weight_coverage_mean']:.6f} | {coverage_audit['first_minute']['missing_rate_mean']:.6f} | {coverage_audit['first_minute']['join_retention_mean']:.6f} | {coverage_audit['first_minute']['n']} |"
    )
    lines.append(
        f"| last | {coverage_audit['last_minute']['weight_coverage_mean']:.6f} | {coverage_audit['last_minute']['missing_rate_mean']:.6f} | {coverage_audit['last_minute']['join_retention_mean']:.6f} | {coverage_audit['last_minute']['n']} |"
    )
    lines.append("")
    lines.append("## Appendix")
    lines.append("")
    lines.append("- 为了保持主报告聚焦结论, 分钟级长表移至附录。")
    lines.append("")
    lines.append("### 分钟级覆盖率表 (Per-Minute Table)")
    lines.append("")
    lines.append("| minute_index | weight_coverage | missing_rate | breadth_pos | join_retention | n |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for row in coverage_audit["minute_table"]:
        lines.append(
            f"| {int(row['minute_index'])} | {float(row['weight_coverage']):.6f} | {float(row['missing_rate']):.6f} | {float(row['breadth_pos']):.6f} | {float(row['join_retention']):.6f} | {int(row['n'])} |"
        )
    lines.append("")
    lines.append("## 抽样核对 (Spot Checks, No Look-Ahead)")
    lines.append("")
    for idx, item in enumerate(spot_checks, start=1):
        lines.append(f"### Spot Check {idx}")
        lines.append("")
        lines.append(f"- date: `{item['date']}`")
        lines.append(f"- datetime_t: `{item['datetime_t']}`")
        lines.append(f"- close_t: `{item['close_t']:.6f}`")
        lines.append(f"- datetime_t_plus_10: `{item['datetime_t_plus_10']}`")
        lines.append(f"- close_t_plus_10: `{item['close_t_plus_10']:.6f}`")
        lines.append(f"- label: `{item['label']:.8f}`")
        lines.append(f"- ret_5m_feature: `{item['ret_5m']:.8f}`")
        lines.append("")

    # Write to file.
    out_path = os.path.join(report_dir, "data_audit.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def write_report_md(report_dir: str, run_id: str, metrics: dict) -> None:
    """Write a main report markdown with key results and methodology."""

    # Convert YYYYMMDD integers into ISO dates for report readability.
    def _date_int_to_iso(date_int: int) -> str:
        """Convert an int YYYYMMDD into an ISO date string."""

        # Split into year, month, and day for stable formatting.
        year = int(date_int) // 10000
        month = (int(date_int) // 100) % 100
        day = int(date_int) % 100
        return f"{year:04d}-{month:02d}-{day:02d}"

    # Summarize walk-forward stability as compact statistics.
    def _summarize_walk_forward(wf_rows: list[dict]) -> dict:
        """Summarize walk-forward fold metrics into simple stats."""

        # Collect fold metrics into numeric arrays.
        ic_vals = np.array([float(row["ic"]) for row in wf_rows], dtype=float)
        rank_ic_vals = np.array([float(row["rank_ic"]) for row in wf_rows], dtype=float)

        # Compute negative-rate and quantiles for stability diagnostics.
        return {
            "folds": int(len(wf_rows)),
            "neg_ic_folds": int(np.sum(ic_vals < 0.0)),
            "neg_rank_ic_folds": int(np.sum(rank_ic_vals < 0.0)),
            "ic_min": float(np.nanmin(ic_vals)),
            "ic_median": float(np.nanmedian(ic_vals)),
            "ic_max": float(np.nanmax(ic_vals)),
            "rank_ic_min": float(np.nanmin(rank_ic_vals)),
            "rank_ic_median": float(np.nanmedian(rank_ic_vals)),
            "rank_ic_max": float(np.nanmax(rank_ic_vals)),
        }

    # Compose a conclusion-driven report with embedded artifacts.
    lines: list[str] = []
    lines.append(f"# 报告 (Report, {run_id})")
    lines.append("")
    # Add explicit test coverage range at the top.
    xgb_daily = metrics["daily"]["xgb"]
    test_start = int(min([int(row["date"]) for row in xgb_daily]))
    test_end = int(max([int(row["date"]) for row in xgb_daily]))
    lines.append("## 测试区间 (Test Coverage)")
    lines.append("")
    lines.append(f"- Test coverage: `{_date_int_to_iso(test_start)}` 至 `{_date_int_to_iso(test_end)}`.")
    lines.append("")
    lines.append("## Takeaways")
    lines.append("")
    # Highlight ranking vs point-error trade-off across models.
    xgb_test = metrics["overall"]["xgb"]["test"]
    linear_test = metrics["overall"]["linear_model"]["test"]
    zero_test = metrics["overall"]["zero"]["test"]
    lines.append(
        f"- **排序能力 (IC/Rank IC)**: `xgb` 在 test 上 IC={float(xgb_test['ic']):.4f}, RankIC={float(xgb_test['rank_ic']):.4f}, 明显领先, 更适合作为选股/排序信号的候选。"
    )
    lines.append(
        f"- **点预测误差 (RMSE/MAE)**: `linear_model` 的 RMSE={float(linear_test['rmse']):.6f}, MAE={float(linear_test['mae']):.6f} 更低, 更偏向“数值预测”口径的优势。"
    )
    lines.append(
        f"- **方向准确率 (Direction Acc)**: `zero` 的 DirAcc={float(zero_test['direction_acc']):.4f} 并不差, 说明在高噪声短周期里, 方向类指标可能被基准/样本分布主导, 不能替代 IC/误差口径。"
    )
    lines.append("- 结论上, 本任务应同时观察“排序能力”和“点预测误差”, 避免只用单一指标做模型选择。")
    lines.append("")
    lines.append("## Pipeline")
    lines.append("")
    lines.append("- Data: `/data/ashare/market/etf1m` and `/data/ashare/market/stock1m`.")
    lines.append("- Index weights: `data/ashare/market/index_weight/000905.feather` (symlinked).")
    lines.append("- Label horizon: 10 minutes.")
    lines.append(f"- Split: train `20210101-20231231`, test `{_date_int_to_iso(test_start)}` 至 `{_date_int_to_iso(test_end)}`.")
    lines.append("")
    lines.append("## 模型 (Models)")
    lines.append("")
    lines.append("- Baselines: `zero`, `last_value`, `rolling_mean`, `linear_model`.")
    lines.append("- Main model: `xgb` (XGBoost regression).")
    lines.append("")
    lines.append("## 测试集指标 (Test, Overall)")
    lines.append("")
    # Present overall metrics as a compact table for cross-model comparison.
    lines.append("| model | ic | rank_ic | dir_acc | rmse | mae | n |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for model_name in sorted(metrics["overall"].keys()):
        row = metrics["overall"][model_name]["test"]
        lines.append(
            f"| {model_name} | {float(row['ic']):.6f} | {float(row['rank_ic']):.6f} | {float(row['direction_acc']):.6f} | {float(row['rmse']):.6f} | {float(row['mae']):.6f} | {int(row['n'])} |"
        )
    lines.append("")
    if "walk_forward" in metrics and "xgb" in metrics["walk_forward"]:
        lines.append("## Walk-forward 稳定性 (Train, Expanding Window)")
        lines.append("")
        # Provide a stability summary before the fold table.
        wf_stats = _summarize_walk_forward(metrics["walk_forward"]["xgb"])
        lines.append(
            f"- folds: `{wf_stats['folds']}`, neg_ic_folds: `{wf_stats['neg_ic_folds']}`, neg_rank_ic_folds: `{wf_stats['neg_rank_ic_folds']}`."
        )
        lines.append(
            f"- ic (min/median/max): `{wf_stats['ic_min']:.4f}` / `{wf_stats['ic_median']:.4f}` / `{wf_stats['ic_max']:.4f}`."
        )
        lines.append(
            f"- rank_ic (min/median/max): `{wf_stats['rank_ic_min']:.4f}` / `{wf_stats['rank_ic_median']:.4f}` / `{wf_stats['rank_ic_max']:.4f}`."
        )
        lines.append("")
        wf = metrics["walk_forward"]["xgb"]
        lines.append("| val_month | train_month_start | train_month_end | ic | rank_ic | n |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for row in wf:
            lines.append(
                f"| {int(row['val_month'])} | {int(row['train_month_start'])} | {int(row['train_month_end'])} | {float(row['ic']):.6f} | {float(row['rank_ic']):.6f} | {int(row['n'])} |"
            )
        lines.append("")
    if "by_month" in metrics and "xgb" in metrics["by_month"]:
        lines.append("## 月度稳定性诊断 (Monthly, Test)")
        lines.append("")
        # Use by_month metrics to surface drawdown risk and regime breaks.
        by_month = metrics["by_month"]["xgb"]
        month_lookup = {int(row["month"]): row for row in by_month}
        for month in [202410, 202601]:
            row = month_lookup[month]
            lines.append(
                f"- 风险提示: `xgb` 在 `{month}` 出现负 IC, IC={float(row['ic']):.4f}, RankIC={float(row['rank_ic']):.4f}, DirAcc={float(row['direction_acc']):.4f}, 这类月份在实盘排序信号上可能对应明显回撤。"
            )
        lines.append("")
        lines.append("| month | ic | rank_ic | dir_acc | rmse | mae | n |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for row in by_month:
            lines.append(
                f"| {int(row['month'])} | {float(row['ic']):.6f} | {float(row['rank_ic']):.6f} | {float(row['direction_acc']):.6f} | {float(row['rmse']):.6f} | {float(row['mae']):.6f} | {int(row['n'])} |"
            )
        lines.append("")
    if "feature_ic" in metrics and "test" in metrics["feature_ic"]:
        lines.append("## 特征分析 (Feature IC vs. Model Importance)")
        lines.append("")
        # Explain why single-feature IC and model importance can disagree.
        lines.append(
            "- 单特征 IC 衡量的是“线性单调相关”, 而树模型的重要性衡量的是“被用于分裂带来的损失下降”, 两者口径不同, 出现冲突是常见现象。"
        )
        lines.append(
            "- `minute_index` 与 `etf_minus_comp_ret_1m` 在 test 上单特征 IC 为负, 但在 `xgb` 中重要性不低, 更可能表示它们在非线性/分段条件下提供了有效信息, 或用于识别不同状态并调节其他特征的边际作用。"
        )
        lines.append(
            "- 建议的验证路径: (1) 按分钟段或波动率分桶分别算 IC; (2) 做 permutation importance 或 SHAP 检查是否存在稳定贡献; (3) 检查这些特征在 202410/202601 是否发生分布漂移。"
        )
        lines.append("")
        lines.append("| feature | ic | ir | daily_ic_mean | daily_ic_std | n_days |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for row in metrics["feature_ic"]["test"]:
            lines.append(
                f"| {row['feature']} | {float(row['ic']):.6f} | {float(row['ir']):.6f} | {float(row['daily_ic_mean']):.6f} | {float(row['daily_ic_std']):.6f} | {int(row['n_days'])} |"
            )
        lines.append("")
    if "xgb_feature_importance" in metrics:
        lines.append("## XGB 特征重要性 (Feature Importance)")
        lines.append("")
        # Mark features with zero importance as candidates for removal.
        zero_imp = [row["feature"] for row in metrics["xgb_feature_importance"] if float(row["importance"]) == 0.0]
        lines.append(
            "- Importance 为 0 的特征可视为“待剔除/逻辑失效”的候选: " + ", ".join([f"`{name}`" for name in zero_imp]) + "."
        )
        lines.append("")
        lines.append("| feature | importance |")
        lines.append("| --- | --- |")
        for row in metrics["xgb_feature_importance"][:30]:
            lines.append(f"| {row['feature']} | {float(row['importance']):.6f} |")
        lines.append("")
    lines.append("## 数据审计摘要 (Data Audit Highlights)")
    lines.append("")
    # Pull the minimal audit numbers into the main report to avoid long tables.
    audit = metrics["data_audit_summary"]
    lines.append(f"- Overall retention (join_retention_mean): `{float(audit['overall_join_retention_mean']):.6f}`.")
    lines.append(
        f"- First minute: weight_coverage_mean=`{float(audit['first_minute_weight_coverage_mean']):.6f}`, "
        f"missing_rate_mean=`{float(audit['first_minute_missing_rate_mean']):.6f}`, "
        f"join_retention_mean=`{float(audit['first_minute_join_retention_mean']):.6f}`."
    )
    lines.append(
        f"- Last minute: weight_coverage_mean=`{float(audit['last_minute_weight_coverage_mean']):.6f}`, "
        f"missing_rate_mean=`{float(audit['last_minute_missing_rate_mean']):.6f}`, "
        f"join_retention_mean=`{float(audit['last_minute_join_retention_mean']):.6f}`."
    )
    lines.append("")
    lines.append("## 图表 (Figures)")
    lines.append("")
    # Embed key figures with one-sentence business captions.
    lines.append("![Cumulative IC (test)](fig_ic_cum.png)")
    lines.append("*Caption: IC 累计曲线更接近信号随时间累积贡献, 用于观察长期漂移与回撤段。*")
    lines.append("")
    lines.append("![Baseline comparison (test)](fig_baseline_comparison.png)")
    lines.append("*Caption: IC 与 RMSE 的并列对比强调“排序能力”和“点预测误差”往往不一致, 需要联合决策。*")
    lines.append("")
    lines.append("![XGB feature importance](fig_xgb_feature_importance.png)")
    lines.append("*Caption: Top 特征重要性分布用于识别模型依赖的关键信息源, 并筛出可剔除的低贡献特征。*")
    lines.append("")

    # Write to file.
    out_path = os.path.join(report_dir, "report.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def write_bottom_up_report_md(report_dir: str, run_id: str, metrics: dict) -> None:
    """Write a bottom-up synthesis report with stock, basket, and ETF sections."""

    # Compose a three-part report aligned with AGENTS.md deliverables.
    lines: list[str] = []
    lines.append(f"# Bottom-up Synthesis 报告 ({run_id})")
    lines.append("")
    lines.append("## 配置 (Config)")
    lines.append("")
    cfg = metrics["config"]
    lines.append(f"- ETF: `{int(cfg['etf_code_int'])}`.")
    lines.append(f"- Horizon minutes: `{int(cfg['label_horizon_minutes'])}`.")
    lines.append(f"- Train range (calendar): `{int(cfg['train_range'][0])}` - `{int(cfg['train_range'][1])}`.")
    lines.append(f"- Test start (calendar): `{int(cfg['test_start'])}`.")
    lines.append(f"- Used train days: `{int(cfg['used_train_days'])}`.")
    lines.append(f"- Used test days: `{int(cfg['used_test_days'])}`.")
    lines.append("")
    lines.append("## Part 1: Stock Alpha (个股端)")
    lines.append("")
    stock = metrics["stock_alpha"]
    lines.append(
        f"- Stock XGB panel IC (test, mean over minutes): IC={float(stock['panel_ic_test_mean']):.6f}, RankIC={float(stock['panel_rank_ic_test_mean']):.6f}."
    )
    lines.append(
        f"- Stock XGB daily IC (test, mean over days): IC={float(stock['daily_ic_test_mean']):.6f}, RankIC={float(stock['daily_rank_ic_test_mean']):.6f}."
    )
    lines.append("")
    lines.append("### 分钟 Bucket IC (test)")
    lines.append("")
    lines.append("| minute_bucket | ic_mean | rank_ic_mean | minutes | n_sum |")
    lines.append("| --- | --- | --- | --- | --- |")
    for row in stock["minute_bucket_ic_test"]:
        lines.append(
            f"| {int(row['minute_bucket'])} | {float(row['ic_mean']):.6f} | {float(row['rank_ic_mean']):.6f} | {int(row['minutes'])} | {int(row['n_sum'])} |"
        )
    lines.append("")
    lines.append("## Part 2: Synthesis (合成端, Basket)")
    lines.append("")
    basket = metrics["basket_synthesis"]["overall"]
    lines.append("| model | ic | rank_ic | dir_acc | rmse | mae | n |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for model_name in sorted(basket.keys()):
        row = basket[model_name]["test"]
        lines.append(
            f"| {model_name} | {float(row['ic']):.6f} | {float(row['rank_ic']):.6f} | {float(row['direction_acc']):.6f} | {float(row['rmse']):.6f} | {float(row['mae']):.6f} | {int(row['n'])} |"
        )
    lines.append("")
    lines.append("## Part 3: ETF (ETF 端)")
    lines.append("")
    etf = metrics["etf_level"]["overall"]
    lines.append("| model | ic | rank_ic | dir_acc | rmse | mae | n |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for model_name in sorted(etf.keys()):
        row = etf[model_name]["test"]
        lines.append(
            f"| {model_name} | {float(row['ic']):.6f} | {float(row['rank_ic']):.6f} | {float(row['direction_acc']):.6f} | {float(row['rmse']):.6f} | {float(row['mae']):.6f} | {int(row['n'])} |"
        )
    lines.append("")
    lines.append("### 选型结论 (Selection)")
    lines.append("")
    sel = metrics["selection"]
    lines.append(f"- Selected model: `{sel['selected_model']}` (by `{sel['selection_key']}` on test).")
    lines.append(
        f"- Selected test metrics: IC=`{float(sel['selected_test_ic']):.6f}`, RankIC=`{float(sel['selected_test_rank_ic']):.6f}`, RMSE=`{float(sel['selected_test_rmse']):.6f}`, MAE=`{float(sel['selected_test_mae']):.6f}`."
    )
    lines.append("")
    lines.append("### Raw vs Basis (Delta, test)")
    lines.append("")
    lines.append("| raw_model | basis_model | ic_delta | rank_ic_delta |")
    lines.append("| --- | --- | --- | --- |")
    for row in metrics["etf_level"]["raw_vs_basis_delta_test"]:
        lines.append(
            f"| {row['raw_model']} | {row['basis_model']} | {float(row['ic_delta']):.6f} | {float(row['rank_ic_delta']):.6f} |"
        )
    lines.append("")
    lines.append("## 图表 (Figures)")
    lines.append("")
    lines.append("![Basket branch comparison (test)](fig_basket_branch_compare_test.png)")
    lines.append("")
    lines.append("![ETF branch comparison (test)](fig_etf_branch_compare_test.png)")
    lines.append("")
    lines.append("![Raw vs basis delta (test)](fig_raw_vs_basis_delta_test.png)")
    lines.append("")
    lines.append("![Best model daily IC (test)](fig_best_daily_ic.png)")
    lines.append("")
    lines.append("![Best model cumulative IC (test)](fig_best_cum_ic.png)")
    lines.append("")
    lines.append("![Best model monthly IC (test)](fig_best_monthly_ic.png)")
    lines.append("")

    # Write to file.
    out_path = os.path.join(report_dir, "report.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def write_report_tex(report_dir: str, run_id: str, metrics: dict, out_name: str, asset_prefix: str) -> None:
    """Write a LaTeX research report with key results."""

    # Escape common LaTeX special characters for safe table rendering.
    def _tex_escape(text: str) -> str:
        """Escape a plain string for LaTeX text context."""

        # Apply minimal escaping for identifiers and file-like strings.
        return (
            str(text)
            .replace("\\", "\\textbackslash{}")
            .replace("&", "\\&")
            .replace("_", "\\_")
            .replace("%", "\\%")
            .replace("#", "\\#")
        )

    # Wrap filesystem paths so LaTeX accepts underscores in filenames.
    def _tex_path(path: str) -> str:
        """Wrap a path with \\detokenize{} for \\includegraphics."""

        # Use detokenize to avoid escaping every filename character.
        return "\\detokenize{" + path + "}"

    # Convert YYYYMMDD integers into ISO dates for report readability.
    def _date_int_to_iso(date_int: int) -> str:
        """Convert an int YYYYMMDD into an ISO date string."""

        # Split into year, month, and day for stable formatting.
        year = int(date_int) // 10000
        month = (int(date_int) // 100) % 100
        day = int(date_int) % 100
        return f"{year:04d}-{month:02d}-{day:02d}"

    # Collect overall test metrics for a compact table.
    overall = metrics.get("overall", {})
    model_names = sorted(overall.keys())
    test_rows: list[tuple[str, dict]] = []
    for model_name in model_names:
        row = overall[model_name].get("test", {})
        if len(row) == 0:
            continue
        test_rows.append((model_name, row))

    # Prepare test coverage range for the report header.
    xgb_daily = metrics["daily"]["xgb"]
    test_start = int(min([int(row["date"]) for row in xgb_daily]))
    test_end = int(max([int(row["date"]) for row in xgb_daily]))

    # Build a research-style LaTeX document aligned with the markdown depth.
    lines: list[str] = []
    lines.append("\\documentclass[11pt]{article}")
    lines.append("\\usepackage[margin=1in]{geometry}")
    lines.append("\\usepackage{booktabs}")
    lines.append("\\usepackage{longtable}")
    lines.append("\\usepackage{graphicx}")
    lines.append("\\usepackage{hyperref}")
    lines.append("\\title{510500 未来 10 分钟收益率预测研究报告}")
    lines.append(f"\\author{{Run: {run_id}}}")
    lines.append("\\date{}")
    lines.append("\\begin{document}")
    lines.append("\\maketitle")
    lines.append("")
    lines.append("\\section{任务定义}")
    lines.append("目标是预测 ETF 510500 在时刻 $t$ 的未来 10 分钟收益率, 仅允许使用 $t$ 及之前可观测信息构造特征与 baseline。")
    lines.append("")
    lines.append("\\section{数据与切分}")
    lines.append("\\begin{itemize}")
    lines.append("\\item ETF 1m: /data/ashare/market/etf1m")
    lines.append("\\item Stock 1m: /data/ashare/market/stock1m")
    lines.append("\\item Index weight: data/ashare/market/index\\_weight/000905.feather")
    lines.append(f"\\item Train: 20210101--20231231, Test: {_date_int_to_iso(test_start)}--{_date_int_to_iso(test_end)}")
    lines.append("\\end{itemize}")
    lines.append("")
    lines.append("\\section{Takeaways}")
    lines.append("\\begin{itemize}")
    xgb_test = metrics["overall"]["xgb"]["test"]
    linear_test = metrics["overall"]["linear_model"]["test"]
    zero_test = metrics["overall"]["zero"]["test"]
    lines.append(
        f"\\item 排序能力 (IC/Rank IC): xgb 在 test 上 IC={float(xgb_test['ic']):.4f}, RankIC={float(xgb_test['rank_ic']):.4f}, 更适合作为排序信号候选。"
    )
    lines.append(
        f"\\item 点预测误差 (RMSE/MAE): linear\\_model 的 RMSE={float(linear_test['rmse']):.6f}, MAE={float(linear_test['mae']):.6f} 更低, 更偏向数值预测优势。"
    )
    lines.append(
        f"\\item 方向准确率 (Direction Acc): zero 的 DirAcc={float(zero_test['direction_acc']):.4f}, 说明方向类指标可能被样本分布影响, 不能替代 IC/误差口径。"
    )
    lines.append("\\end{itemize}")
    lines.append("")
    lines.append("\\section{标签定义}")
    lines.append("$\\mathrm{label}_t = \\frac{\\mathrm{Close}_{t+10}}{\\mathrm{Close}_t} - 1$.")
    lines.append("")
    lines.append("\\section{模型与 Baseline}")
    lines.append("\\begin{itemize}")
    lines.append("\\item zero: $\\hat{y}_t = 0$")
    lines.append("\\item last\\_value / rolling\\_mean: 使用历史 label, 并通过 $\\mathrm{shift\\_n} = \\mathrm{horizon}+1$ 的时间滞后来避免泄漏。")
    lines.append("\\item linear\\_model: 线性回归 baseline。")
    lines.append("\\item xgb: XGBoost 回归主模型。")
    lines.append("\\end{itemize}")
    lines.append("")
    lines.append("\\section{测试集结果 (Overall)}")
    lines.append("\\begin{longtable}{lrrrrrr}")
    lines.append("\\toprule")
    lines.append("Model & IC & RankIC & DirAcc & RMSE & MAE & N\\\\")
    lines.append("\\midrule")
    for model_name, row in test_rows:
        lines.append(
            f"{model_name} & {float(row['ic']):.4f} & {float(row['rank_ic']):.4f} & {float(row['direction_acc']):.4f} & {float(row['rmse']):.6f} & {float(row['mae']):.6f} & {int(row['n'])}\\\\"
        )
    lines.append("\\bottomrule")
    lines.append("\\end{longtable}")
    lines.append("")
    lines.append("\\section{月度稳定性诊断 (Test)}")
    lines.append("特别关注出现负 IC 的月份, 其对应排序信号可能带来显著回撤风险。")
    lines.append("\\begin{longtable}{rrrrrrr}")
    lines.append("\\toprule")
    lines.append("Month & IC & RankIC & DirAcc & RMSE & MAE & N\\\\")
    lines.append("\\midrule")
    for row in metrics["by_month"]["xgb"]:
        lines.append(
            f"{int(row['month'])} & {float(row['ic']):.4f} & {float(row['rank_ic']):.4f} & {float(row['direction_acc']):.4f} & {float(row['rmse']):.6f} & {float(row['mae']):.6f} & {int(row['n'])}\\\\"
        )
    lines.append("\\bottomrule")
    lines.append("\\end{longtable}")
    lines.append("")
    lines.append("\\section{Walk-forward 稳定性 (Train)}")
    wf_rows = metrics["walk_forward"]["xgb"]
    folds = int(len(wf_rows))
    neg_ic_folds = int(sum([float(r["ic"]) < 0.0 for r in wf_rows]))
    neg_rank_ic_folds = int(sum([float(r["rank_ic"]) < 0.0 for r in wf_rows]))
    lines.append(f"Fold 数={folds}, 其中 IC<0 的 fold 数={neg_ic_folds}, RankIC<0 的 fold 数={neg_rank_ic_folds}。")
    lines.append("\\begin{longtable}{rrrrrr}")
    lines.append("\\toprule")
    lines.append("ValMonth & TrainStart & TrainEnd & IC & RankIC & N\\\\")
    lines.append("\\midrule")
    for row in wf_rows:
        lines.append(
            f"{int(row['val_month'])} & {int(row['train_month_start'])} & {int(row['train_month_end'])} & {float(row['ic']):.4f} & {float(row['rank_ic']):.4f} & {int(row['n'])}\\\\"
        )
    lines.append("\\bottomrule")
    lines.append("\\end{longtable}")
    lines.append("")
    lines.append("\\section{特征分析}")
    lines.append("单特征 IC 与模型重要性存在口径差异: 前者度量线性相关, 后者度量分裂贡献。")
    lines.append("\\subsection{Feature IC/IR (Test)}")
    lines.append("\\begin{longtable}{lrrrrr}")
    lines.append("\\toprule")
    lines.append("Feature & IC & IR & DailyICMean & DailyICStd & NDays\\\\")
    lines.append("\\midrule")
    for row in metrics["feature_ic"]["test"]:
        lines.append(
            f"{_tex_escape(row['feature'])} & {float(row['ic']):.4f} & {float(row['ir']):.4f} & {float(row['daily_ic_mean']):.4f} & {float(row['daily_ic_std']):.4f} & {int(row['n_days'])}\\\\"
        )
    lines.append("\\bottomrule")
    lines.append("\\end{longtable}")
    lines.append("")
    lines.append("\\subsection{XGB Feature Importance}")
    zero_imp = [row["feature"] for row in metrics["xgb_feature_importance"] if float(row["importance"]) == 0.0]
    lines.append("Importance 为 0 的特征建议优先剔除并复验: " + ", ".join([_tex_escape(name) for name in zero_imp]) + ".")
    lines.append("\\begin{longtable}{lr}")
    lines.append("\\toprule")
    lines.append("Feature & Importance\\\\")
    lines.append("\\midrule")
    for row in metrics["xgb_feature_importance"][:30]:
        lines.append(f"{_tex_escape(row['feature'])} & {float(row['importance']):.6f}\\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{longtable}")
    lines.append("")
    lines.append("\\section{数据审计摘要}")
    audit = metrics["data_audit_summary"]
    lines.append("\\begin{itemize}")
    lines.append(f"\\item Overall retention (join\\_retention\\_mean): {float(audit['overall_join_retention_mean']):.6f}.")
    lines.append(
        f"\\item First minute: weight\\_coverage\\_mean={float(audit['first_minute_weight_coverage_mean']):.6f}, missing\\_rate\\_mean={float(audit['first_minute_missing_rate_mean']):.6f}, join\\_retention\\_mean={float(audit['first_minute_join_retention_mean']):.6f}."
    )
    lines.append(
        f"\\item Last minute: weight\\_coverage\\_mean={float(audit['last_minute_weight_coverage_mean']):.6f}, missing\\_rate\\_mean={float(audit['last_minute_missing_rate_mean']):.6f}, join\\_retention\\_mean={float(audit['last_minute_join_retention_mean']):.6f}."
    )
    lines.append("\\end{itemize}")
    lines.append("")
    lines.append("\\section{关键图表}")
    # Include key figures; asset_prefix controls whether paths live under run_id/.
    def _fig(name: str) -> str:
        # Build a relative path to the artifact for LaTeX compilation.
        rel = os.path.join(asset_prefix, name) if len(asset_prefix) else name
        return _tex_path(rel)

    lines.append("\\begin{figure}[ht]")
    lines.append("\\centering")
    lines.append("\\includegraphics[width=\\linewidth]{" + _fig("fig_ic_cum.png") + "}")
    lines.append("\\caption{IC 累计曲线更接近信号随时间累积贡献, 用于观察长期漂移与回撤段.}")
    lines.append("\\end{figure}")
    lines.append("")
    lines.append("\\begin{figure}[ht]")
    lines.append("\\centering")
    lines.append("\\includegraphics[width=\\linewidth]{" + _fig("fig_baseline_comparison.png") + "}")
    lines.append("\\caption{IC 与 RMSE 的并列对比强调排序能力和点预测误差往往不一致, 需要联合决策.}")
    lines.append("\\end{figure}")
    lines.append("")
    lines.append("\\begin{figure}[ht]")
    lines.append("\\centering")
    lines.append("\\includegraphics[width=0.9\\linewidth]{" + _fig("fig_xgb_feature_importance.png") + "}")
    lines.append("\\caption{Top 特征重要性分布用于识别模型依赖的信息源, 并筛出可剔除的低贡献特征.}")
    lines.append("\\end{figure}")
    lines.append("")
    lines.append("\\section{产出物}")
    lines.append("本次运行的关键产出包括: metrics.yaml, predictions.parquet, data\\_audit.md, summary.md, report.md, 以及各类图表 (IC/RankIC/DirectionAcc 时序, baseline 对比, scatter, timeseries)。")
    lines.append("")
    lines.append("\\end{document}")

    # Write the LaTeX to the report directory.
    out_path = os.path.join(report_dir, out_name)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
