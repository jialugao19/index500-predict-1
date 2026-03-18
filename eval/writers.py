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

    # Compose a lightweight report with links to artifacts.
    lines: list[str] = []
    lines.append(f"# 报告 (Report, {run_id})")
    lines.append("")
    lines.append("## Pipeline")
    lines.append("")
    lines.append("- Data: `/data/ashare/market/etf1m` and `/data/ashare/market/stock1m`.")
    lines.append("- Index weights: `data/ashare/market/index_weight/000905.feather` (symlinked).")
    lines.append("- Label horizon: 10 minutes.")
    lines.append("- Split: train `20210101-20231231`, test `>=20240201`.")
    lines.append("")
    lines.append("## 模型 (Models)")
    lines.append("")
    lines.append("- Baselines: `zero`, `last_value`, `rolling_mean`, `linear_model`.")
    lines.append("- Main model: `xgb` (XGBoost regression).")
    lines.append("")
    lines.append("## 测试集指标 (Test, Overall)")
    lines.append("")
    for model_name in sorted(metrics["overall"].keys()):
        row = metrics["overall"][model_name].get("test", {})
        if len(row) == 0:
            continue
        lines.append(
            f"- {model_name}: IC={row['ic']:.4f}, RankIC={row['rank_ic']:.4f}, "
            f"DirAcc={row['direction_acc']:.4f}, RMSE={row['rmse']:.6f}, MAE={row['mae']:.6f}, n={row['n']}"
        )
    lines.append("")
    if "walk_forward" in metrics and "xgb" in metrics["walk_forward"]:
        lines.append("## Walk-forward 验证 (Train)")
        lines.append("")
        wf = metrics["walk_forward"]["xgb"]
        lines.append("| val_month | train_month_start | train_month_end | ic | rank_ic | n |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for row in wf:
            lines.append(
                f"| {int(row['val_month'])} | {int(row['train_month_start'])} | {int(row['train_month_end'])} | {float(row['ic']):.6f} | {float(row['rank_ic']):.6f} | {int(row['n'])} |"
            )
        lines.append("")
    if "feature_ic" in metrics and "test" in metrics["feature_ic"]:
        lines.append("## 新特征有效性 (Feature IC/IR, Test)")
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
        lines.append("| feature | importance |")
        lines.append("| --- | --- |")
        for row in metrics["xgb_feature_importance"][:30]:
            lines.append(f"| {row['feature']} | {float(row['importance']):.6f} |")
        lines.append("")
    lines.append("## 图表 (Figures)")
    lines.append("")
    lines.append("- `fig_ic_timeseries.png`")
    lines.append("- `fig_rank_ic_timeseries.png`")
    lines.append("- `fig_direction_acc_timeseries.png`")
    lines.append("- `fig_baseline_comparison.png`")
    lines.append("- `fig_prediction_scatter.png`")
    lines.append("- `fig_prediction_timeseries.png`")
    lines.append("")

    # Write to file.
    out_path = os.path.join(report_dir, "report.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def write_report_tex(report_dir: str, run_id: str, metrics: dict, out_name: str) -> None:
    """Write a LaTeX research report with key results."""

    # Collect overall test metrics for a compact table.
    overall = metrics.get("overall", {})
    model_names = sorted(overall.keys())
    test_rows: list[tuple[str, dict]] = []
    for model_name in model_names:
        row = overall[model_name].get("test", {})
        if len(row) == 0:
            continue
        test_rows.append((model_name, row))

    # Build a minimal LaTeX document that is easy to compile and review.
    lines: list[str] = []
    lines.append("\\documentclass[11pt]{article}")
    lines.append("\\usepackage[margin=1in]{geometry}")
    lines.append("\\usepackage{booktabs}")
    lines.append("\\usepackage{longtable}")
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
    lines.append("\\item Train: 20210101--20231231, Test: from 20240201")
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
    lines.append("\\section{产出物}")
    lines.append("本次运行的关键产出包括: metrics.yaml, predictions.parquet, data\\_audit.md, summary.md, report.md, 以及各类图表 (IC/RankIC/DirectionAcc 时序, baseline 对比, scatter, timeseries)。")
    lines.append("")
    lines.append("\\end{document}")

    # Write the LaTeX to the report directory.
    out_path = os.path.join(report_dir, out_name)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
