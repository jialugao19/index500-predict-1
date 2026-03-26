import os

import numpy as np
import pandas as pd
import yaml
from matplotlib import pyplot as plt


def write_yaml(path: str, obj: dict) -> None:
    """Write a Python dict as YAML."""

    # Serialize plain dict as YAML for reproducibility.
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, allow_unicode=True, sort_keys=False)


def compute_pooled_metrics_from_sufficient_stats(stats: pd.DataFrame) -> dict:
    """Compute pooled metrics by summing sufficient statistics across rows."""

    # Aggregate sufficient statistics into a single global row.
    totals = stats.loc[
        :,
        [
            "n",
            "pred_sum",
            "label_sum",
            "pred2_sum",
            "label2_sum",
            "pred_label_sum",
            "abs_err_sum",
            "sq_err_sum",
            "dir_correct_sum",
        ],
    ].sum(numeric_only=True)

    # Compute pooled mean/variance/covariance terms from second moments.
    n = float(totals["n"])
    pred_mean = float(totals["pred_sum"]) / n
    label_mean = float(totals["label_sum"]) / n
    pred_var = float(totals["pred2_sum"]) / n - pred_mean * pred_mean
    label_var = float(totals["label2_sum"]) / n - label_mean * label_mean
    cov = float(totals["pred_label_sum"]) / n - pred_mean * label_mean
    denom = float(np.sqrt(pred_var * label_var))

    # Compute pooled metrics aligned with eval.metrics conventions.
    ic = float(cov / denom) if denom != 0.0 and np.isfinite(denom) else float("nan")
    rmse = float(np.sqrt(float(totals["sq_err_sum"]) / n))
    mae = float(totals["abs_err_sum"]) / n
    direction_acc = float(totals["dir_correct_sum"]) / n

    # Return a compact dict that is stable for YAML serialization.
    return {
        "n": int(totals["n"]),
        "ic": ic,
        "rmse": rmse,
        "mae": mae,
        "direction_acc": direction_acc,
        "pred_mean": pred_mean,
        "label_mean": label_mean,
        "pred_std": float(np.sqrt(pred_var)) if pred_var >= 0.0 else float("nan"),
        "label_std": float(np.sqrt(label_var)) if label_var >= 0.0 else float("nan"),
        "bias_mean": float(pred_mean - label_mean),
    }


def compute_stratified_pooled_metrics(
    stock_metrics: pd.DataFrame,
    bin_col: str,
    num_bins: int,
) -> list[dict]:
    """Compute pooled metrics per quantile bin using sufficient statistics."""

    # Build bin ids using quantile buckets for stable group sizes.
    data = stock_metrics.copy()
    data["bin"] = pd.qcut(data[bin_col].astype(float), q=int(num_bins), labels=False, duplicates="drop")

    # Compute pooled metrics for each bin by summing sufficient statistics.
    rows: list[dict] = []
    for bin_id, part in data.groupby("bin", sort=True):
        # Compute pooled metrics from sufficient stats for stocks in this bin.
        pooled = compute_pooled_metrics_from_sufficient_stats(stats=part)
        rows.append(
            {
                "bin": int(bin_id),
                "bin_col": str(bin_col),
                "bin_min": float(part[bin_col].astype(float).min()),
                "bin_max": float(part[bin_col].astype(float).max()),
                "n_stocks": int(len(part)),
                "pooled": pooled,
                "ic_mean_stock": float(part["ic"].astype(float).mean()),
                "ic_median_stock": float(part["ic"].astype(float).median()),
                "weight_mean_stock": float(part["weight_mean"].astype(float).mean()),
            }
        )
    return rows


def plot_hist(values: np.ndarray, bins: int, title: str, out_path: str) -> None:
    """Plot a histogram for 1D numeric values."""

    # Filter to finite values for stable histogram rendering.
    vec = np.asarray(values, dtype=float)
    vec = vec[np.isfinite(vec)]

    # Draw and save a compact histogram figure.
    plt.figure(figsize=(8, 4))
    plt.hist(vec, bins=int(bins), alpha=0.85)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_scatter_with_fit(x: np.ndarray, y: np.ndarray, xlabel: str, ylabel: str, title: str, out_path: str) -> dict:
    """Plot scatter and fit a linear trend y = a + b*x."""

    # Filter to finite pairs for stable least squares fitting.
    x_vec = np.asarray(x, dtype=float)
    y_vec = np.asarray(y, dtype=float)
    mask = np.isfinite(x_vec) & np.isfinite(y_vec)
    x_clean = x_vec[mask]
    y_clean = y_vec[mask]

    # Fit a simple OLS line and compute fitted values.
    x_mat = np.column_stack([np.ones(len(x_clean), dtype=float), x_clean])
    coef, *_ = np.linalg.lstsq(x_mat, y_clean, rcond=None)
    intercept = float(coef[0])
    slope = float(coef[1])
    x_line = np.linspace(float(np.min(x_clean)), float(np.max(x_clean)), num=200)
    y_line = intercept + slope * x_line

    # Draw scatter and fitted line.
    plt.figure(figsize=(6, 5))
    plt.scatter(x_clean, y_clean, s=14, alpha=0.4)
    plt.plot(x_line, y_line, color="black", linewidth=1.2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

    # Return fitted coefficients for the markdown report.
    return {"n": int(len(x_clean)), "intercept": intercept, "slope": slope}


def plot_bin_bars(bins: list[dict], value_key: str, title: str, out_path: str) -> None:
    """Plot a bar chart for binned metric values."""

    # Extract x labels and y values from the bin rows.
    x = [int(row["bin"]) for row in bins]
    y = [float(row["pooled"][value_key]) for row in bins]

    # Draw and save a bar chart.
    plt.figure(figsize=(10, 4))
    plt.bar(x, y)
    plt.title(title)
    plt.xlabel("bin (qcut)")
    plt.ylabel(value_key)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_intraday_bucket_ic(bucket_rows: list[dict], title: str, out_path: str) -> None:
    """Plot intraday bucket IC mean curve from metrics.yaml."""

    # Convert bucket rows into a stable DataFrame.
    table = pd.DataFrame(bucket_rows).copy()
    table = table.sort_values("minute_bucket", ascending=True)

    # Draw and save the bucket IC curve chart.
    plt.figure(figsize=(10, 4))
    plt.plot(table["minute_bucket"].astype(int), table["ic_mean"].astype(float), marker="o", linewidth=1.2)
    plt.title(title)
    plt.xlabel("minute_bucket (bucket_size=30)")
    plt.ylabel("ic_mean")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def write_stock_report_md(out_dir: str, run_dir: str, payload: dict) -> None:
    """Write the stock-only research report markdown."""

    # Build markdown lines for a focused stock prediction report.
    lines: list[str] = []
    lines.append("# 个股预测效果报告（Stock Alpha, 10m）")
    lines.append("")
    lines.append("## 1. 运行与数据口径")
    lines.append("")
    lines.append(f"- run_dir: `{run_dir}`")
    lines.append("- label: `label_stock_10m = Close_{t+10}/Close_t - 1`")
    lines.append("- 一个样本: `date + stock_code + MinuteIndex` 的分钟级观测")
    lines.append("")
    lines.append("## 2. 指标口径（非常重要）")
    lines.append("")
    lines.append("- Panel IC: **每分钟横截面**（500只成分股）上 `corr(pred,label)`，再对测试期分钟取均值/分桶均值。")
    lines.append("- Pooled TS-IC: 把所有股票的分钟样本拼接成一个大样本（等价于对 sufficient stats 求和）再算 `corr(pred,label)`。")
    lines.append("- Per-stock TS-IC: 对每只股票单独在全测试期分钟序列上算 `corr(pred_i, label_i)`。")
    lines.append("")
    lines.append("## 3. 总体结果（Pooled, test）")
    lines.append("")
    pooled = payload["pooled_test"]
    lines.append("| metric | value |")
    lines.append("| --- | --- |")
    lines.append(f"| n | {int(pooled['n'])} |")
    lines.append(f"| pooled_ts_ic | {float(pooled['ic']):.6f} |")
    lines.append(f"| rmse | {float(pooled['rmse']):.6f} |")
    lines.append(f"| mae | {float(pooled['mae']):.6f} |")
    lines.append(f"| direction_acc | {float(pooled['direction_acc']):.6f} |")
    lines.append(f"| pred_std | {float(pooled['pred_std']):.6f} |")
    lines.append(f"| label_std | {float(pooled['label_std']):.6f} |")
    lines.append(f"| bias_mean(pred-label) | {float(pooled['bias_mean']):.8f} |")
    lines.append("")
    lines.append("### Panel IC（横截面排序能力, test）")
    lines.append("")
    panel = payload["panel_ic_test"]
    lines.append(f"- panel_ic_test_mean: `{float(panel['panel_ic_test_mean']):.6f}`")
    lines.append(f"- panel_rank_ic_test_mean: `{float(panel['panel_rank_ic_test_mean']):.6f}`")
    lines.append("")
    lines.append("![Intraday bucket IC](fig_stock_panel_ic_by_minute_bucket_test.png)")
    lines.append("")
    lines.append("## 4. Per-stock TS-IC 分布（test）")
    lines.append("")
    dist = payload["per_stock_ic_dist"]
    lines.append(
        f"- ic_mean: `{float(dist['mean']):.6f}`, ic_median: `{float(dist['median']):.6f}`, ic_q05/q95: `{float(dist['q05']):.6f}` / `{float(dist['q95']):.6f}`"
    )
    lines.append(
        f"- verdict_rule: `|IC| >= {float(payload['verdict_rule']['ic_abs_threshold']):.4f}` and `IC_t >= {float(payload['verdict_rule']['ic_t_threshold']):.2f}`"
    )
    lines.append(f"- verdict_counts: `{payload['verdict_counts']}`")
    lines.append("")
    lines.append("![Per-stock IC histogram](fig_stock_ic_hist.png)")
    lines.append("")
    lines.append("![Per-stock daily_ic_mean histogram](fig_stock_daily_ic_mean_hist.png)")
    lines.append("")
    lines.append("## 5. 分层结果（按权重 weight_mean, test）")
    lines.append("")
    lines.append("- 说明: 每个分桶内用 sufficient stats **pooled** 计算 IC/RMSE/MAE/DirAcc。")
    lines.append("")
    lines.append("| bin | bin_min | bin_max | n_stocks | pooled_ts_ic | rmse | mae | dir_acc | ic_median_stock | weight_mean_stock |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in payload["by_weight_decile_test"]:
        p = row["pooled"]
        lines.append(
            f"| {int(row['bin'])} | {float(row['bin_min']):.6f} | {float(row['bin_max']):.6f} | {int(row['n_stocks'])} | {float(p['ic']):.6f} | {float(p['rmse']):.6f} | {float(p['mae']):.6f} | {float(p['direction_acc']):.6f} | {float(row['ic_median_stock']):.6f} | {float(row['weight_mean_stock']):.6f} |"
        )
    lines.append("")
    lines.append("![Pooled IC by weight decile](fig_stock_pooled_ic_by_weight_decile_test.png)")
    lines.append("")
    fit = payload["weight_vs_ic_fit"]
    lines.append("- weight_mean vs per-stock IC 线性拟合: `ic = a + b*weight_mean`")
    lines.append(f"  - n={int(fit['n'])}, a={float(fit['intercept']):.6f}, b={float(fit['slope']):.6f}")
    lines.append("")
    lines.append("![Weight vs IC scatter](fig_stock_weight_vs_ic.png)")
    lines.append("")
    lines.append("## 6. 分层结果（按 label 波动 label_std, test）")
    lines.append("")
    lines.append("| bin | bin_min | bin_max | n_stocks | pooled_ts_ic | rmse | mae | dir_acc | ic_median_stock |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in payload["by_label_std_decile_test"]:
        p = row["pooled"]
        lines.append(
            f"| {int(row['bin'])} | {float(row['bin_min']):.6f} | {float(row['bin_max']):.6f} | {int(row['n_stocks'])} | {float(p['ic']):.6f} | {float(p['rmse']):.6f} | {float(p['mae']):.6f} | {float(p['direction_acc']):.6f} | {float(row['ic_median_stock']):.6f} |"
        )
    lines.append("")
    lines.append("## 7. 附录：文件与产物")
    lines.append("")
    lines.append(f"- stock_metrics_test: `{payload['artifact_stock_metrics_test']}`")
    lines.append("- stock_report_metrics: `stock_report_metrics.yaml`")
    lines.append("")

    # Write markdown to the requested output directory.
    out_path = os.path.join(out_dir, "stock_prediction_report_20260320.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    """Build a stock-only report under report/0320-01."""

    # Define input run directory and output report directory.
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    run_dir = os.path.join(repo_root, "report", "0320", "run_latest")
    out_dir = os.path.join(repo_root, "report", "0320-01")
    os.makedirs(out_dir, exist_ok=True)

    # Load summary metrics and the per-stock sufficient statistics artifact.
    with open(os.path.join(run_dir, "metrics.yaml"), "r", encoding="utf-8") as f:
        metrics = yaml.safe_load(f)
    stock_metrics = pd.read_parquet(os.path.join(run_dir, "stock_metrics_test.parquet"))

    # Compute pooled test metrics and per-stock IC distribution statistics.
    pooled_test = compute_pooled_metrics_from_sufficient_stats(stats=stock_metrics)
    ic_vec = stock_metrics["ic"].to_numpy(dtype=float)
    ic_finite = ic_vec[np.isfinite(ic_vec)]
    q05, q50, q95 = np.quantile(ic_finite, [0.05, 0.50, 0.95])
    per_stock_ic_dist = {
        "mean": float(np.mean(ic_finite)),
        "median": float(q50),
        "q05": float(q05),
        "q95": float(q95),
    }

    # Compute stratified pooled metrics by stock weight and by label volatility.
    by_weight_decile_test = compute_stratified_pooled_metrics(stock_metrics=stock_metrics, bin_col="weight_mean", num_bins=10)
    stock_metrics = stock_metrics.copy()
    stock_metrics["label_std"] = np.sqrt(stock_metrics["label_var"].to_numpy(dtype=float))
    by_label_std_decile_test = compute_stratified_pooled_metrics(stock_metrics=stock_metrics, bin_col="label_std", num_bins=10)

    # Generate figures for distribution and stratification diagnostics.
    plot_hist(values=ic_finite, bins=60, title="Per-stock TS-IC distribution (test)", out_path=os.path.join(out_dir, "fig_stock_ic_hist.png"))
    plot_hist(
        values=stock_metrics["daily_ic_mean"].to_numpy(dtype=float),
        bins=60,
        title="Per-stock daily_ic_mean distribution (test)",
        out_path=os.path.join(out_dir, "fig_stock_daily_ic_mean_hist.png"),
    )
    plot_intraday_bucket_ic(
        bucket_rows=metrics["stock_alpha"]["minute_bucket_ic_test"],
        title="Stock XGB panel IC by intraday minute bucket (test)",
        out_path=os.path.join(out_dir, "fig_stock_panel_ic_by_minute_bucket_test.png"),
    )
    plot_bin_bars(
        bins=by_weight_decile_test,
        value_key="ic",
        title="Pooled TS-IC by weight_mean decile (test)",
        out_path=os.path.join(out_dir, "fig_stock_pooled_ic_by_weight_decile_test.png"),
    )
    weight_vs_ic_fit = plot_scatter_with_fit(
        x=stock_metrics["weight_mean"].to_numpy(dtype=float),
        y=stock_metrics["ic"].to_numpy(dtype=float),
        xlabel="weight_mean",
        ylabel="per-stock TS-IC",
        title="weight_mean vs per-stock TS-IC (test)",
        out_path=os.path.join(out_dir, "fig_stock_weight_vs_ic.png"),
    )

    # Assemble a YAML payload for the markdown report and for downstream reuse.
    per_stock_cfg = metrics["stock_alpha"]["per_stock_test"]
    payload = {
        "run_dir": str(run_dir),
        "pooled_test": pooled_test,
        "panel_ic_test": {
            "panel_ic_test_mean": float(metrics["stock_alpha"]["panel_ic_test_mean"]),
            "panel_rank_ic_test_mean": float(metrics["stock_alpha"]["panel_rank_ic_test_mean"]),
        },
        "per_stock_ic_dist": per_stock_ic_dist,
        "verdict_rule": {
            "ic_t_threshold": float(per_stock_cfg["ic_t_threshold"]),
            "ic_abs_threshold": float(per_stock_cfg["ic_abs_threshold"]),
        },
        "verdict_counts": {str(k): int(v) for k, v in per_stock_cfg["verdict_counts"].items()},
        "by_weight_decile_test": by_weight_decile_test,
        "by_label_std_decile_test": by_label_std_decile_test,
        "weight_vs_ic_fit": weight_vs_ic_fit,
        "artifact_stock_metrics_test": "../0320/run_latest/stock_metrics_test.parquet",
    }

    # Write report artifacts (YAML + Markdown) into the required folder.
    write_yaml(os.path.join(out_dir, "stock_report_metrics.yaml"), payload)
    write_stock_report_md(out_dir=out_dir, run_dir=run_dir, payload=payload)


if __name__ == "__main__":
    main()
