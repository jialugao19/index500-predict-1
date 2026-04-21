import os

import yaml


def _find_latest_run_with_metrics(report_root: str) -> str:
    """Find latest run dir that contains metrics.yaml."""

    # Collect completed run folders.
    candidates: list[str] = []
    for day_name in sorted(os.listdir(report_root)):
        day_dir = os.path.join(report_root, day_name)
        if not os.path.isdir(day_dir):
            continue
        for run_name in sorted(os.listdir(day_dir)):
            run_dir = os.path.join(day_dir, run_name)
            if not os.path.isdir(run_dir):
                continue
            if os.path.exists(os.path.join(run_dir, "metrics.yaml")):
                candidates.append(run_dir)

    # Pick the newest by mtime of metrics.yaml.
    candidates_sorted = sorted(candidates, key=lambda p: os.path.getmtime(os.path.join(p, "metrics.yaml")), reverse=True)
    assert len(candidates_sorted) > 0
    return str(candidates_sorted[0])


def _find_latest_run_with_metrics_by_horizon(report_root: str, horizon_minutes: int) -> str | None:
    """Find latest run dir with metrics.yaml matching a specific horizon."""

    # Scan completed runs and filter by config.horizon.
    candidates: list[str] = []
    for day_name in sorted(os.listdir(report_root)):
        day_dir = os.path.join(report_root, day_name)
        if not os.path.isdir(day_dir):
            continue
        for run_name in sorted(os.listdir(day_dir)):
            run_dir = os.path.join(day_dir, run_name)
            if not os.path.isdir(run_dir):
                continue
            metrics_path = os.path.join(run_dir, "metrics.yaml")
            if not os.path.exists(metrics_path):
                continue
            with open(metrics_path, "r", encoding="utf-8") as f:
                payload = yaml.safe_load(f)
            if int(payload["config"]["label_horizon_minutes"]) == int(horizon_minutes):
                candidates.append(run_dir)

    if len(candidates) == 0:
        return None

    candidates_sorted = sorted(candidates, key=lambda p: os.path.getmtime(os.path.join(p, "metrics.yaml")), reverse=True)
    return str(candidates_sorted[0])


def _find_latest_exp1_alignment(report_root: str) -> str:
    """Find latest exp1 output dir with target_alignment_metrics.yaml."""

    # Collect all exp1 dirs.
    candidates: list[str] = []
    for day_name in sorted(os.listdir(report_root)):
        day_dir = os.path.join(report_root, day_name)
        if not os.path.isdir(day_dir):
            continue
        for run_name in sorted(os.listdir(day_dir)):
            run_dir = os.path.join(day_dir, run_name)
            if not os.path.isdir(run_dir):
                continue
            if os.path.exists(os.path.join(run_dir, "target_alignment_metrics.yaml")):
                candidates.append(run_dir)

    candidates_sorted = sorted(
        candidates, key=lambda p: os.path.getmtime(os.path.join(p, "target_alignment_metrics.yaml")), reverse=True
    )
    assert len(candidates_sorted) > 0
    return str(candidates_sorted[0])


def _read_yaml(path: str) -> dict:
    """Read YAML into dict."""

    # Load YAML payload for report generation.
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    """Generate a compact experiment comparison report."""

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    report_root = os.path.join(repo_root, "report")

    # Locate the current 30m mainline and the historical alignment experiment.
    baseline_dir = _find_latest_run_with_metrics_by_horizon(report_root=report_root, horizon_minutes=30)
    assert baseline_dir is not None
    exp1_dir = _find_latest_exp1_alignment(report_root=report_root)

    baseline = _read_yaml(os.path.join(baseline_dir, "metrics.yaml"))
    exp1 = _read_yaml(os.path.join(exp1_dir, "target_alignment_metrics.yaml"))
    exp1_base_dir = str(exp1["base_run_dir"])

    # Extract the current 30m mainline ETF metrics.
    baseline_best = baseline["selection_etf"]
    baseline_etf = baseline["etf_level"]["overall"][baseline_best["selected_model"]]["test"]

    # Extract the best historical alignment result by test rank_ic.
    exp1_overall = exp1["metrics"]["overall"]
    best_exp1_model = None
    best_rank_ic = -1e9
    for model_name in sorted(exp1_overall.keys()):
        v = float(exp1_overall[model_name]["test"]["rank_ic"])
        if v > best_rank_ic:
            best_rank_ic = v
            best_exp1_model = model_name
    assert best_exp1_model is not None
    exp1_best = exp1_overall[best_exp1_model]["test"]

    # Compose markdown report.
    lines: list[str] = []
    lines.append("# 实验研究报告: 当前 30m 主线 vs 历史 Alignment 实验")
    lines.append("")
    lines.append("## 路径 (Paths)")
    lines.append("")
    lines.append(f"- mainline_30m_run_dir: `{baseline_dir}`")
    lines.append(f"- exp1_target_alignment_dir: `{exp1_dir}`")
    lines.append(f"- exp1_source_run_dir: `{exp1_base_dir}`")
    lines.append("")
    lines.append("## 结果对比 (test)")
    lines.append("")
    lines.append("| item | model | ic | rank_ic | rmse | mae | n |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    lines.append(
        f"| mainline_30m_etf | {baseline_best['selected_model']} | {float(baseline_etf['ic']):.6f} | {float(baseline_etf['rank_ic']):.6f} | "
        f"{float(baseline_etf['rmse']):.6f} | {float(baseline_etf['mae']):.6f} | {int(baseline_etf['n'])} |"
    )
    lines.append(
        f"| historical_exp1_target_alignment | {best_exp1_model} | {float(exp1_best['ic']):.6f} | {float(exp1_best['rank_ic']):.6f} | "
        f"{float(exp1_best['rmse']):.6f} | {float(exp1_best['mae']):.6f} | {int(exp1_best['n'])} |"
    )
    lines.append("")
    lines.append("## 备注")
    lines.append("")
    lines.append("- 当前默认主线已经统一为 `30m`，主线指标来自该 horizon 的 pipeline 输出 `metrics.yaml`。")
    lines.append("- `exp1_target_alignment` 是历史实验，来源于其 `base_run_dir` 对应的旧主线输出，不再代表当前默认主线。")
    lines.append("- 若要评估 alignment 在 `30m` 主线下是否仍然有效，应基于新的 `30m` baseline 重新生成一次 exp1 结果。")
    lines.append("")

    out_path = os.path.join(report_root, "experiment_research_report.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[INFO] Done. out_path={out_path}")


if __name__ == "__main__":
    main()
