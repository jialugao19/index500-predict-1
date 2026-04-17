import base64
import datetime as dt
import os
from pathlib import Path

import matplotlib.pyplot as plt
import yaml


def _find_latest_run_with_metrics_by_horizon(report_root: str, horizon_minutes: int) -> str:
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
        raise ValueError(f"no run found with metrics.yaml for horizon={horizon_minutes} under report_root={report_root}")

    # Pick the newest by mtime of metrics.yaml.
    candidates_sorted = sorted(candidates, key=lambda p: os.path.getmtime(os.path.join(p, "metrics.yaml")), reverse=True)
    return str(candidates_sorted[0])


def _read_yaml(path: str) -> dict:
    """Read YAML into dict."""

    # Load YAML payload for report generation.
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _fmt(x: float, nd: int) -> str:
    """Format a float value for table rendering."""

    # Convert and format deterministically.
    return f"{float(x):.{int(nd)}f}"


def _plot_horizon_curves(rows: list[dict], out_png_path: str) -> None:
    """Plot IC/RankIC and stock TS-IC vs horizon."""

    # Convert rows into aligned numeric series.
    horizons = [int(r["horizon_minutes"]) for r in rows]
    etf_ic = [float(r["etf_ic"]) for r in rows]
    etf_ric = [float(r["etf_rank_ic"]) for r in rows]
    stock_ts_ic = [float(r["stock_lgbm_ts_ic"]) for r in rows]
    stock_ts_ric = [float(r["stock_lgbm_ts_rank_ic"]) for r in rows]

    # Build a two-panel figure: ETF metrics and stock metrics.
    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    ax0, ax1 = axes

    # Plot ETF IC and RankIC vs horizon.
    ax0.plot(horizons, etf_ic, marker="o", linewidth=1.6, label="ETF IC (Pearson)")
    ax0.plot(horizons, etf_ric, marker="o", linewidth=1.6, label="ETF RankIC (Spearman)")
    ax0.set_ylabel("ETF metric (test)")
    ax0.set_title("Horizon vs ETF metrics (test)")
    ax0.grid(True, alpha=0.25)
    ax0.legend()

    # Plot stock pooled TS-IC and TS-RankIC for lgbm vs horizon.
    ax1.plot(horizons, stock_ts_ic, marker="o", linewidth=1.6, label="Stock TS-IC (lgbm, pooled)")
    ax1.plot(horizons, stock_ts_ric, marker="o", linewidth=1.6, label="Stock TS-RankIC (lgbm, pooled)")
    ax1.set_xlabel("label_horizon_minutes")
    ax1.set_ylabel("Stock metric (test)")
    ax1.set_title("Horizon vs stock metrics (test, pooled)")
    ax1.grid(True, alpha=0.25)
    ax1.legend()

    # Save and close the figure.
    fig.tight_layout()
    fig.savefig(out_png_path)
    plt.close(fig)


def main() -> None:
    """Generate a self-contained HTML report comparing horizon=10/20/30m."""

    # Resolve repo-root and report-root paths.
    repo_root = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    report_root = repo_root / "report"

    # Locate latest runs for each horizon.
    horizons = [10, 20, 30]
    run_dirs = {int(h): _find_latest_run_with_metrics_by_horizon(report_root=str(report_root), horizon_minutes=int(h)) for h in horizons}

    # Extract comparable ETF/stock metrics into a compact table.
    rows: list[dict] = []
    for h in horizons:
        run_dir = Path(run_dirs[int(h)])
        metrics = _read_yaml(str(run_dir / "metrics.yaml"))

        # Extract ETF best branch based on selection_etf.
        selected_model = str(metrics["selection_etf"]["selected_model"])
        etf_test = metrics["etf_level"]["overall"][selected_model]["test"]

        # Extract stock lgbm pooled TS-IC.
        stock_lgbm = metrics["stock_alpha"]["overall"]["lgbm"]

        # Store one comparable row for rendering.
        rows.append(
            {
                "horizon_minutes": int(h),
                "run_dir": str(run_dir),
                "selected_model": selected_model,
                "etf_ic": float(etf_test["ic"]),
                "etf_rank_ic": float(etf_test["rank_ic"]),
                "etf_rmse": float(etf_test["rmse"]),
                "etf_mae": float(etf_test["mae"]),
                "etf_n": int(etf_test["n"]),
                "stock_lgbm_ts_ic": float(stock_lgbm["ts_ic_test"]),
                "stock_lgbm_ts_rank_ic": float(stock_lgbm["ts_rank_ic_test"]),
                "stock_lgbm_panel_ic": float(stock_lgbm["panel_ic_test_mean"]),
                "stock_lgbm_panel_rank_ic": float(stock_lgbm["panel_rank_ic_test_mean"]),
            }
        )

    # Create a comparison plot and embed it via base64 data URI.
    out_day_dir = report_root / dt.datetime.now().strftime("%m%d")
    out_day_dir.mkdir(parents=True, exist_ok=True)
    fig_path = out_day_dir / "fig_horizon_compare_ic_curves.png"
    _plot_horizon_curves(rows=rows, out_png_path=str(fig_path))
    fig_b64 = base64.b64encode(fig_path.read_bytes()).decode("ascii")
    fig_data_uri = f"data:image/png;base64,{fig_b64}"

    # Define a compact CSS theme with larger base font to match prior reports.
    css = r"""
:root {
  --text: #111827;
  --muted: #6b7280;
  --bg: #ffffff;
  --card: #f9fafb;
  --border: #e5e7eb;
  --good: #16a34a;
  --bad: #dc2626;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, "Noto Sans", "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", sans-serif;
  font-size: 18px;
  line-height: 1.65;
  color: var(--text);
  background: var(--bg);
}
main {
  max-width: 980px;
  margin: 0 auto;
  padding: 28px 20px 60px;
}
h1 { font-size: 30px; margin: 0 0 10px; }
h2 { font-size: 22px; margin: 26px 0 10px; }
p { margin: 10px 0; }
.small { color: var(--muted); font-size: 14px; }
.card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 14px 16px;
}
ul { margin: 8px 0 0; padding-left: 18px; }
li { margin: 6px 0; }
code {
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
  font-size: 0.95em;
  background: #eef2ff;
  padding: 1px 6px;
  border-radius: 6px;
}
table {
  width: 100%;
  border-collapse: collapse;
  margin: 10px 0 0;
  font-size: 16px;
}
th, td {
  border: 1px solid var(--border);
  padding: 8px 10px;
  text-align: right;
  vertical-align: middle;
}
th:first-child, td:first-child { text-align: left; }
img.figure {
  display: block;
  width: 100%;
  max-width: 920px;
  margin: 12px auto 0;
  border: 1px solid var(--border);
  border-radius: 10px;
  background: #fff;
}
.caption { margin-top: 8px; color: var(--muted); font-size: 14px; }
hr { border: 0; border-top: 1px solid var(--border); margin: 24px 0; }
"""

    # Render the main comparison table.
    table_lines: list[str] = []
    table_lines.append("<table>")
    table_lines.append(
        "<thead><tr>"
        "<th>horizon(m)</th>"
        "<th>ETF IC</th>"
        "<th>ETF RankIC</th>"
        "<th>ETF RMSE</th>"
        "<th>ETF MAE</th>"
        "<th>ETF n</th>"
        "<th>Stock TS-IC (lgbm)</th>"
        "<th>Stock TS-RankIC (lgbm)</th>"
        "</tr></thead>"
    )
    table_lines.append("<tbody>")
    for r in rows:
        table_lines.append(
            "<tr>"
            f"<td>{int(r['horizon_minutes'])}</td>"
            f"<td>{_fmt(r['etf_ic'], 6)}</td>"
            f"<td>{_fmt(r['etf_rank_ic'], 6)}</td>"
            f"<td>{_fmt(r['etf_rmse'], 6)}</td>"
            f"<td>{_fmt(r['etf_mae'], 6)}</td>"
            f"<td>{int(r['etf_n'])}</td>"
            f"<td>{_fmt(r['stock_lgbm_ts_ic'], 6)}</td>"
            f"<td>{_fmt(r['stock_lgbm_ts_rank_ic'], 6)}</td>"
            "</tr>"
        )
    table_lines.append("</tbody></table>")
    table_html = "\n".join(table_lines)

    # Assemble the full HTML document.
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    out_html_path = out_day_dir / "research_report_horizon_10_20_30.html"
    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>研究报告: horizon=10/20/30m 对比</title>
  <style>{css}</style>
</head>
<body>
  <main>
    <h1>研究报告: horizon=10m vs 20m vs 30m (self-constrained)</h1>
    <p class="small">生成时间: {now}. 输出文件: <code>{out_html_path.as_posix()}</code></p>

    <div class="card">
      <h2>1. 背景与目的</h2>
      <p>在分钟级预测中, horizon 会显著影响噪声占比与可预测结构。本报告对比 <code>horizon=10m</code>, <code>20m</code>, <code>30m</code> 三个设置下, ETF 端聚合后指标与 stock 端指标的变化趋势。</p>
      <ul>
        <li>ETF 端指标来自: <code>basket_pred</code> 与 <code>label_etf_h</code> 的 join 后, 在 test split 上计算 pooled TS-IC 与 RankIC。</li>
        <li>Stock 端指标来自: pooled TS-IC/TS-RankIC (lgbm), 以及 panel IC 的均值(用于诊断截面信息)。</li>
      </ul>
      <p class="caption">说明: 代码字段名历史上使用 <code>label_*_10m</code> 命名, 但这里的数值均对应各自 horizon 的 forward return。</p>
    </div>

    <h2>2. 结果总览 (test)</h2>
    <div class="card">
      {table_html}
      <p class="caption">表: 每个 horizon 选择同一选择规则(<code>selection_etf.selected_model</code>)下的 ETF-level 指标, 以及 stock lgbm 的 pooled 指标。</p>
    </div>

    <h2>3. 趋势图</h2>
    <div class="card">
      <img class="figure" alt="Horizon compare curves" src="{fig_data_uri}" />
      <p class="caption">图: Horizon 对 ETF IC/RankIC 与 stock lgbm pooled TS-IC/TS-RankIC 的影响(均为 test split)。</p>
    </div>

    <h2>4. 运行路径 (Paths)</h2>
    <div class="card">
      <ul>
        <li>10m run_dir: <code>{rows[0]['run_dir']}</code></li>
        <li>20m run_dir: <code>{rows[1]['run_dir']}</code></li>
        <li>30m run_dir: <code>{rows[2]['run_dir']}</code></li>
      </ul>
    </div>

    <h2>5. 结论与建议</h2>
    <div class="card">
      <ul>
        <li>若 ETF 端 IC 随 horizon 增加显著提升而 stock 端 TS-IC 下降, 通常意味着: 更长 horizon 降低微结构噪声, 强化共同成分可预测性; 同时弱化了个股短周期 idio 信号。</li>
        <li>建议在确定业务目标(ETF-level vs stock-level)后选择 horizon: 若以 ETF 端稳定性为主, 可以偏向更长 horizon; 若强调个股层面短周期 alpha, 则需要接受聚合后 IC 稀释, 并在聚合层引入更贴合的结构特征。</li>
      </ul>
    </div>

    <hr />
    <p class="small">附: 图文件已内嵌, 无外部依赖。原始 png: <code>{fig_path.as_posix()}</code>。</p>
  </main>
</body>
</html>
"""

    # Write the HTML report to the report/<mmdd>/ folder.
    out_html_path.write_text(html, encoding="utf-8")
    print(f"[INFO] Done. out_html_path={out_html_path.as_posix()}")


if __name__ == "__main__":
    main()

