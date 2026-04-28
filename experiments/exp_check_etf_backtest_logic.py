import datetime as dt
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

# Add repo root into sys.path so local modules can be imported when running as a script.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from pipeline import load_etf_minute_bars  # noqa: E402


def _ensure_dir(path: str) -> None:
    """Create a directory recursively."""

    # Create the output directory.
    os.makedirs(path, exist_ok=True)


def _read_yaml(path: str) -> dict:
    """Read a YAML file into a dict."""

    # Load the YAML payload.
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _write_yaml(path: str, obj: dict) -> None:
    """Write a dict as YAML."""

    # Dump the YAML payload.
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, allow_unicode=True, sort_keys=False)


def _summarize_curve(datetimes: pd.Series, curve: np.ndarray) -> dict:
    """Summarize one equity-like curve."""

    # Convert inputs into stable arrays.
    arr = np.asarray(curve, dtype=float)
    times = pd.to_datetime(datetimes).reset_index(drop=True)

    # Compute point-in-time extrema.
    max_idx = int(np.nanargmax(arr))
    min_idx = int(np.nanargmin(arr))
    running_max = np.maximum.accumulate(arr)
    drawdown = np.full(len(arr), np.nan, dtype=float)
    positive_mask = running_max > 0.0
    drawdown[positive_mask] = arr[positive_mask] / running_max[positive_mask] - 1.0
    dd_idx = int(np.nanargmin(drawdown)) if int(np.sum(np.isfinite(drawdown))) else 0

    # Build a compact summary dict.
    return {
        "start": float(arr[0]),
        "end": float(arr[-1]),
        "max": float(arr[max_idx]),
        "max_datetime": str(times.iloc[max_idx]),
        "min": float(arr[min_idx]),
        "min_datetime": str(times.iloc[min_idx]),
        "max_drawdown": float(drawdown[dd_idx]),
        "max_drawdown_datetime": str(times.iloc[dd_idx]),
    }


def _plot_benchmark_compare(
    datetimes: pd.Series,
    flawed_bench_curve: np.ndarray,
    corrected_bench_curve: np.ndarray,
    out_path: str,
) -> None:
    """Plot flawed and corrected benchmark curves on the same timestamps."""

    # Draw the level comparison panel.
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    ax1.plot(datetimes, flawed_bench_curve, label="Current chart benchmark: cumprod(1 + label_30m)", linewidth=1.1)
    ax1.plot(datetimes, corrected_bench_curve, label="Correct benchmark: ETF close / first close", linewidth=1.1)
    ax1.set_title("Benchmark logic check: flawed vs corrected long ETF curve")
    ax1.set_ylabel("Curve level")
    ax1.set_yscale("log")
    ax1.legend()

    # Draw the drawdown comparison panel.
    flawed_dd = flawed_bench_curve / np.maximum.accumulate(flawed_bench_curve) - 1.0
    corrected_dd = corrected_bench_curve / np.maximum.accumulate(corrected_bench_curve) - 1.0
    ax2.plot(datetimes, flawed_dd * 100.0, label="Flawed benchmark drawdown", linewidth=1.0)
    ax2.plot(datetimes, corrected_dd * 100.0, label="Correct benchmark drawdown", linewidth=1.0)
    ax2.axhline(0.0, color="black", linewidth=0.8, alpha=0.7)
    ax2.set_ylabel("Drawdown (%)")
    ax2.legend()

    # Save the figure.
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)


def _plot_current_chart_components(
    datetimes: pd.Series,
    flawed_strategy_curve: np.ndarray,
    flawed_bench_curve: np.ndarray,
    out_path: str,
) -> None:
    """Plot the current chart components to expose their scale explosion."""

    # Compute the current excess series.
    excess_curve_bps = (flawed_strategy_curve / flawed_bench_curve - 1.0) * 1e4

    # Draw the flawed level panel.
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    ax1.plot(datetimes, flawed_strategy_curve, label="Current strategy curve", linewidth=1.1)
    ax1.plot(datetimes, flawed_bench_curve, label="Current benchmark curve", linewidth=1.1)
    ax1.set_title("Current ETF chart components based on overlapping 30m forward returns")
    ax1.set_ylabel("Curve level")
    ax1.set_yscale("log")
    ax1.legend()

    # Draw the flawed excess panel.
    ax2.plot(datetimes, excess_curve_bps, label="Current excess curve (bps)", linewidth=1.0)
    ax2.axhline(0.0, color="black", linewidth=0.8, alpha=0.7)
    ax2.set_ylabel("Excess (bps)")
    ax2.legend()

    # Save the figure.
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)


def main() -> None:
    """Run a dedicated experiment to validate the ETF backtest chart logic."""

    # Resolve the current mainline run and selected ETF model.
    metrics_path = os.path.join(REPO_ROOT, "metrics.yaml")
    metrics = _read_yaml(metrics_path)
    run_dir = str(metrics["config"]["report_dir"])
    run_id = str(metrics["config"]["run_id"])
    etf_code_int = int(metrics["config"]["etf_code_int"])
    etf1m_root = str(metrics["config"]["data_roots"]["etf1m_root"])
    selected_model = str(metrics["selection_etf"]["selected_model"])

    # Create a dedicated experiment output directory.
    now = dt.datetime.now()
    out_day_dir = os.path.join(REPO_ROOT, "report", now.strftime("%m%d"))
    out_dir = os.path.join(out_day_dir, now.strftime("exp_check_etf_backtest_logic_%Y%m%d_%H%M%S"))
    _ensure_dir(out_dir)

    # Load the selected ETF prediction table on the test split.
    pred_path = os.path.join(run_dir, "predictions.parquet")
    pred_table = pd.read_parquet(pred_path)
    part = pred_table.loc[
        (pred_table["model_name"].astype(str) == selected_model) & (pred_table["split"].astype(str) == "test")
    ].copy()
    part = part.sort_values("datetime", ascending=True).reset_index(drop=True)

    # Rebuild the current chart logic from overlapping forward returns.
    pred = part["pred"].to_numpy(dtype=float)
    label_30m = part["label"].to_numpy(dtype=float)
    pos = np.sign(pred)
    flawed_strategy_curve = np.cumprod(1.0 + pos * label_30m)
    flawed_bench_curve = np.cumprod(1.0 + label_30m)
    flawed_excess_curve_bps = (flawed_strategy_curve / flawed_bench_curve - 1.0) * 1e4

    # Load actual ETF closes and build the corrected long-only benchmark on the same timestamps.
    etf_frames: list[pd.DataFrame] = []
    for date in sorted(part["date"].astype(int).unique().tolist()):
        # Load one trading day of ETF 1m bars.
        etf_day = load_etf_minute_bars(etf1m_root=etf1m_root, date=int(date), etf_code_int=etf_code_int)
        etf_frames.append(etf_day.loc[:, ["DateTime", "Close"]].rename(columns={"DateTime": "datetime", "Close": "close"}))
    etf_panel = pd.concat(etf_frames, axis=0, ignore_index=True).sort_values("datetime", ascending=True)
    corrected = part.loc[:, ["datetime"]].merge(etf_panel, on="datetime", how="left")
    corrected_bench_curve = corrected["close"].astype(float).to_numpy() / float(corrected["close"].astype(float).iloc[0])

    # Build scalar diagnostics and timestamped extrema summaries.
    overlap_rows_per_day = int(part.groupby("date", sort=True).size().iloc[0])
    findings = {
        "run_id": run_id,
        "run_dir": run_dir,
        "selected_model": selected_model,
        "horizon_minutes": int(metrics["config"]["label_horizon_minutes"]),
        "rows_per_day_in_prediction_table": overlap_rows_per_day,
        "current_chart_logic": {
            "benchmark_definition": "cumprod(1 + label_30m_forward)",
            "strategy_definition": "cumprod(1 + sign(pred) * label_30m_forward)",
        },
        "corrected_benchmark_logic": {
            "benchmark_definition": "ETF_close_t / ETF_close_0 on the same timestamps as prediction table",
        },
        "flawed_strategy_curve": _summarize_curve(part["datetime"], flawed_strategy_curve),
        "flawed_benchmark_curve": _summarize_curve(part["datetime"], flawed_bench_curve),
        "corrected_benchmark_curve": _summarize_curve(part["datetime"], corrected_bench_curve),
        "flawed_excess_curve_bps": _summarize_curve(part["datetime"], flawed_excess_curve_bps),
        "logic_check": {
            "has_logic_issue": True,
            "reason_1": "The current chart compounds overlapping 30m forward returns as if they were per-step realized returns.",
            "reason_2": "A long ETF benchmark should be marked from actual ETF price evolution, not from forward label compounding.",
            "reason_3": "The current strategy curve also uses overlapping forward returns, so it is not a tradable backtest curve.",
        },
        "artifacts": {
            "fig_benchmark_logic_compare": "fig_benchmark_logic_compare.png",
            "fig_current_chart_components": "fig_current_chart_components.png",
            "summary_yaml": "summary.yaml",
            "report_md": "report.md",
        },
    }
    _write_yaml(os.path.join(out_dir, "summary.yaml"), findings)

    # Plot the flawed and corrected benchmark comparison.
    _plot_benchmark_compare(
        datetimes=part["datetime"],
        flawed_bench_curve=flawed_bench_curve,
        corrected_bench_curve=corrected_bench_curve,
        out_path=os.path.join(out_dir, "fig_benchmark_logic_compare.png"),
    )

    # Plot the current chart components for direct inspection.
    _plot_current_chart_components(
        datetimes=part["datetime"],
        flawed_strategy_curve=flawed_strategy_curve,
        flawed_bench_curve=flawed_bench_curve,
        out_path=os.path.join(out_dir, "fig_current_chart_components.png"),
    )

    # Write a concise markdown report.
    lines: list[str] = []
    lines.append("# 实验报告: ETF backtest 图逻辑检查")
    lines.append("")
    lines.append("## 结论")
    lines.append("")
    lines.append("- 当前 `ETF excess return and drawdown (test)` 图存在逻辑漏洞。")
    lines.append("- 根因不是数据异常, 而是把 `30m forward return` 当成了可逐步连乘的已实现收益。")
    lines.append("- `benchmark: long ETF` 的正确设计应当基于 ETF 实际价格路径, 即 `Close_t / Close_0`, 而不是 `cumprod(1 + label_30m_forward)`。")
    lines.append("")
    lines.append("## 当前图为何会爆炸")
    lines.append("")
    lines.append(f"- 当前预测表每个交易日有 `{overlap_rows_per_day}` 条样本, 对应的是滚动的 `30m` forward returns, 样本之间高度重叠。")
    lines.append(
        f"- 当前图里的 benchmark 终值达到 `{findings['flawed_benchmark_curve']['end']:.4f}`, 最大回撤达到 `{findings['flawed_benchmark_curve']['max_drawdown'] * 100.0:.2f}%`。"
    )
    lines.append(
        f"- 正确的 long ETF benchmark 在同一批时间戳上的终值只有 `{findings['corrected_benchmark_curve']['end']:.4f}`, 最大回撤约 `{findings['corrected_benchmark_curve']['max_drawdown'] * 100.0:.2f}%`。"
    )
    lines.append("")
    lines.append("## 极值定位")
    lines.append("")
    lines.append(
        f"- 当前 excess 曲线峰值: `{findings['flawed_excess_curve_bps']['max']:.2f} bps`, 时间 `{findings['flawed_excess_curve_bps']['max_datetime']}`。"
    )
    lines.append(
        f"- 当前 excess 曲线谷值: `{findings['flawed_excess_curve_bps']['min']:.2f} bps`, 时间 `{findings['flawed_excess_curve_bps']['min_datetime']}`。"
    )
    lines.append(
        f"- 当前 benchmark 最大回撤时点: `{findings['flawed_benchmark_curve']['max_drawdown'] * 100.0:.2f}%`, 时间 `{findings['flawed_benchmark_curve']['max_drawdown_datetime']}`。"
    )
    lines.append(
        f"- 正确 benchmark 最大回撤时点: `{findings['corrected_benchmark_curve']['max_drawdown'] * 100.0:.2f}%`, 时间 `{findings['corrected_benchmark_curve']['max_drawdown_datetime']}`。"
    )
    lines.append("")
    lines.append("## 图表")
    lines.append("")
    lines.append("![Benchmark logic compare](fig_benchmark_logic_compare.png)")
    lines.append("")
    lines.append("![Current chart components](fig_current_chart_components.png)")
    lines.append("")

    # Persist the markdown report.
    with open(os.path.join(out_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[INFO] Done. out_dir={out_dir}")


if __name__ == "__main__":
    main()
