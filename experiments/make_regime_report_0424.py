"""Build a self-contained regime research report (0424)."""

from __future__ import annotations

import base64
import datetime as dt
import io
import os
import sys
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from eval.metrics import safe_corr, safe_spearman  # noqa: E402


@dataclass(frozen=True)
class RegimeGroupResult:
    """Per-regime group metrics for the report."""

    group: str
    pooled_ic: float
    pooled_rank_ic: float
    daily_ic_mean: float
    daily_ic_std: float
    daily_icir: float
    positive_day_ratio: float
    n_samples: int
    n_days: int
    strategy_total_return_pct: float
    strategy_max_drawdown_pct: float
    excess_end_bps: float


def read_yaml(path: str) -> dict:
    """Read a YAML file into a dict."""

    # Load structured config from YAML.
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_yaml(path: str, obj: dict) -> None:
    """Write a dict into YAML."""

    # Keep output readable and stable.
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, allow_unicode=True, sort_keys=False)


def load_etf_minute_bars(etf1m_root: str, date: int, etf_code_int: int) -> pd.DataFrame:
    """Load one-day ETF 1m bars for the given ETF code."""

    # Read the feather file and filter to the target ETF.
    year = str(int(date))[:4]
    file_path = os.path.join(etf1m_root, year, f"{int(date)}.feather")
    day = pd.read_feather(file_path)
    day = day.loc[
        day["StockCode"].astype(int) == int(etf_code_int),
        ["Date", "DateTime", "Close", "Vol", "Amount", "MinuteIndex"],
    ].copy()

    # Normalize column names to match pipeline artifacts.
    day = day.sort_values("DateTime", ascending=True).reset_index(drop=True)
    day = day.rename(
        columns={
            "Date": "date",
            "DateTime": "datetime",
            "Close": "close",
            "Vol": "vol",
            "Amount": "amount",
        }
    )
    return day


def intraday_bucket(ts: pd.Timestamp) -> str:
    """Assign timestamp into morning/afternoon/tail buckets."""

    # Use wall-clock time buckets for A-share session structure.
    hhmm = int(ts.hour) * 100 + int(ts.minute)
    if 930 <= hhmm <= 1130:
        return "morning"
    if 1300 <= hhmm < 1430:
        return "afternoon"
    if 1430 <= hhmm <= 1500:
        return "tail"
    return "other"


def compute_drawdown_min_pct(curve: np.ndarray) -> float:
    """Compute max drawdown percent from an equity curve."""

    # Compute drawdown series relative to running peak.
    peak = np.maximum.accumulate(curve)
    dd = (curve / peak - 1.0) * 100.0
    return float(np.min(dd))


def nonoverlap_backtest_summary(
    base: pd.DataFrame,
    etf_price_table: pd.DataFrame,
    horizon_minutes: int,
) -> dict:
    """Compute 30m non-overlap backtest summary for a filtered sample table."""

    # Select entry rows using the global minute index schedule.
    trades = base.loc[base["MinuteIndex"].astype(int) % int(horizon_minutes) == 0, ["date", "MinuteIndex", "pred", "label"]].copy()
    trades["position"] = np.sign(trades["pred"].astype(float))
    trades["strategy_ret"] = trades["position"].astype(float) * trades["label"].astype(float)

    # Map each entry to its exit timestamp using the full ETF minute grid.
    exit_map = etf_price_table.loc[:, ["date", "MinuteIndex", "datetime"]].copy()
    exit_map["exit_minute"] = exit_map["MinuteIndex"].astype(int)
    exit_map = exit_map.drop(columns=["MinuteIndex"]).rename(columns={"datetime": "exit_datetime"})
    trades["exit_minute"] = trades["MinuteIndex"].astype(int) + int(horizon_minutes)
    trades = trades.merge(exit_map, on=["date", "exit_minute"], how="inner")

    # Build a benchmark curve on the full minute grid.
    bench = etf_price_table.loc[:, ["datetime", "close"]].copy()
    bench = bench.sort_values("datetime", ascending=True).reset_index(drop=True)
    bench_curve = bench["close"].astype(float).to_numpy()
    bench_curve = bench_curve / float(bench_curve[0])

    # Expand trade returns into a step strategy curve with exit-time updates.
    equity = 1.0
    updates: list[dict] = []
    for row in trades.sort_values("exit_datetime", ascending=True).itertuples(index=False):
        equity = equity * (1.0 + float(row.strategy_ret))
        updates.append({"datetime": pd.to_datetime(row.exit_datetime), "equity": float(equity)})
    updates_df = pd.DataFrame(updates)
    curve = bench.loc[:, ["datetime"]].copy()
    curve = curve.merge(updates_df, on="datetime", how="left")
    curve["equity"] = curve["equity"].ffill().fillna(1.0)
    strat_curve = curve["equity"].astype(float).to_numpy()

    # Summarize total return, drawdown, and excess vs benchmark.
    excess_end_bps = float((strat_curve[-1] / bench_curve[-1] - 1.0) * 1e4)
    return {
        "trade_count": int(len(trades)),
        "strategy_total_return_pct": float((strat_curve[-1] - 1.0) * 100.0),
        "strategy_max_drawdown_pct": compute_drawdown_min_pct(strat_curve),
        "excess_end_bps": excess_end_bps,
    }


def daily_ic_table(part: pd.DataFrame) -> pd.DataFrame:
    """Compute daily IC series for a given minute-level table."""

    # Compute per-day correlation on minute samples.
    rows: list[dict] = []
    for date, day in part.groupby("date", sort=True):
        pred = day["pred"].to_numpy(dtype=float)
        label = day["label"].to_numpy(dtype=float)
        rows.append(
            {
                "date": int(date),
                "ic": safe_corr(pred, label),
                "rank_ic": safe_spearman(pred, label),
                "n": int(len(day)),
            }
        )
    return pd.DataFrame(rows).sort_values("date", ascending=True).reset_index(drop=True)


def eval_group(
    base: pd.DataFrame,
    etf_price_table: pd.DataFrame,
    horizon_minutes: int,
    group: str,
) -> RegimeGroupResult:
    """Evaluate one regime group into pooled/daily/backtest metrics."""

    # Compute pooled metrics on all minute samples.
    pred = base["pred"].to_numpy(dtype=float)
    label = base["label"].to_numpy(dtype=float)
    pooled_ic = safe_corr(pred, label)
    pooled_rank_ic = safe_spearman(pred, label)

    # Compute daily IC statistics on daily series.
    daily = daily_ic_table(base)
    daily_ic_mean = float(daily["ic"].mean())
    daily_ic_std = float(daily["ic"].std())
    daily_icir = float(daily_ic_mean / daily_ic_std) if daily_ic_std != 0.0 else float("nan")
    valid = daily.loc[np.isfinite(daily["ic"].to_numpy(dtype=float)), "ic"].to_numpy(dtype=float)
    positive_day_ratio = float(np.mean(valid > 0.0)) if len(valid) else float("nan")

    # Compute non-overlap backtest on the filtered sample set.
    bt = nonoverlap_backtest_summary(base=base, etf_price_table=etf_price_table, horizon_minutes=horizon_minutes)
    return RegimeGroupResult(
        group=str(group),
        pooled_ic=float(pooled_ic),
        pooled_rank_ic=float(pooled_rank_ic),
        daily_ic_mean=float(daily_ic_mean),
        daily_ic_std=float(daily_ic_std),
        daily_icir=float(daily_icir),
        positive_day_ratio=float(positive_day_ratio),
        n_samples=int(len(base)),
        n_days=int(base["date"].nunique()),
        strategy_total_return_pct=float(bt["strategy_total_return_pct"]),
        strategy_max_drawdown_pct=float(bt["strategy_max_drawdown_pct"]),
        excess_end_bps=float(bt["excess_end_bps"]),
    )


def plot_metric_bar(table: pd.DataFrame, x: str, y: str, title: str) -> str:
    """Plot a small bar chart and return base64-encoded PNG."""

    # Render a compact bar chart for one metric.
    fig, ax = plt.subplots(1, 1, figsize=(8, 3.2))
    ax.bar(table[x].astype(str), table[y].astype(float))
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    fig.tight_layout()

    # Encode as base64 PNG for self-contained HTML.
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140)
    payload = base64.b64encode(buf.getvalue()).decode("ascii")
    buf.close()
    plt.close(fig)
    return payload


def html_escape(text: str) -> str:
    """Escape HTML special chars in a plain string."""

    # Apply minimal escaping for report values.
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def render_table(rows: list[dict], cols: list[str]) -> str:
    """Render list-of-dicts as an HTML table."""

    # Render a compact table with numeric formatting.
    head = "<tr>" + "".join([f"<th>{html_escape(c)}</th>" for c in cols]) + "</tr>"
    body_rows: list[str] = []
    for row in rows:
        tds: list[str] = []
        for col in cols:
            v = row.get(col, "")
            if isinstance(v, float):
                tds.append(f"<td class='num'>{v:.6f}</td>")
            else:
                tds.append(f"<td>{html_escape(v)}</td>")
        body_rows.append("<tr>" + "".join(tds) + "</tr>")
    return "<table><thead>" + head + "</thead><tbody>" + "".join(body_rows) + "</tbody></table>"


def compute_daily_breadth_dispersion(stock_feature_day: pd.DataFrame) -> dict:
    """Compute daily breadth and dispersion from stock ret_1 and weights."""

    # Build weighted moment columns for fast groupby reductions.
    day = stock_feature_day.loc[:, ["MinuteIndex", "weight", "ret_1"]].copy()
    day["weight"] = day["weight"].astype(float)
    day["ret_1"] = day["ret_1"].astype(float)
    day["wx"] = day["weight"].to_numpy(dtype=float) * day["ret_1"].to_numpy(dtype=float)
    day["wx2"] = day["weight"].to_numpy(dtype=float) * np.square(day["ret_1"].to_numpy(dtype=float))
    day["wpos"] = day["weight"].to_numpy(dtype=float) * (day["ret_1"].to_numpy(dtype=float) > 0.0)

    # Reduce to minute-level breadth and dispersion using weighted moments.
    per_min = day.groupby("MinuteIndex", sort=True).agg(
        sum_w=("weight", "sum"),
        sum_wx=("wx", "sum"),
        sum_wx2=("wx2", "sum"),
        sum_wpos=("wpos", "sum"),
    )
    per_min["mean"] = per_min["sum_wx"].astype(float) / per_min["sum_w"].astype(float)
    per_min["var"] = per_min["sum_wx2"].astype(float) / per_min["sum_w"].astype(float) - np.square(per_min["mean"].astype(float))
    per_min["breadth"] = per_min["sum_wpos"].astype(float) / per_min["sum_w"].astype(float)
    per_min["dispersion"] = np.sqrt(np.maximum(per_min["var"].astype(float), 0.0))

    # Reduce to daily scalars by averaging across minutes.
    return {
        "breadth_pos_mean": float(per_min["breadth"].mean()),
        "dispersion_std_mean": float(per_min["dispersion"].mean()),
    }


def main() -> None:
    """Generate the 0424 regime report under report/0424."""

    # Resolve the current run and load the minimal artifacts.
    metrics = read_yaml(os.path.join(REPO_ROOT, "metrics.yaml"))
    report_dir = str(metrics["config"]["report_dir"])
    run_id = str(metrics["config"]["run_id"])
    etf_code_int = int(metrics["config"]["etf_code_int"])
    horizon_minutes = int(metrics["config"]["label_horizon_minutes"])
    etf1m_root = str(metrics["config"]["data_roots"]["etf1m_root"])
    best_model = str(metrics["selection_etf"]["selected_model"])

    # Load test prediction table for the selected model.
    pred = pd.read_parquet(os.path.join(report_dir, "predictions.parquet"))
    pred = pred.loc[(pred["split"].astype(str) == "test") & (pred["model_name"].astype(str) == best_model)].copy()
    pred = pred.sort_values(["date", "datetime"], ascending=True).reset_index(drop=True)
    test_dates = sorted(pred["date"].astype(int).drop_duplicates().tolist())

    # Load auxiliary parquet artifacts for basis/coverage regimes.
    basket = pd.read_parquet(os.path.join(REPO_ROOT, "basket_pred.parquet"))
    basket = basket.loc[basket["split"].astype(str) == "test"].copy()
    synth = pd.read_parquet(os.path.join(REPO_ROOT, "synthetic_vs_real_etf.parquet"))
    synth = synth.loc[synth["split"].astype(str) == "test"].copy()

    # Load ETF minute bars for all test dates.
    etf_frames: list[pd.DataFrame] = []
    for date in test_dates:
        etf_frames.append(load_etf_minute_bars(etf1m_root=etf1m_root, date=int(date), etf_code_int=etf_code_int))
    etf = pd.concat(etf_frames, axis=0, ignore_index=True)
    etf = etf.sort_values(["date", "MinuteIndex"], ascending=True).reset_index(drop=True)

    # Build minute-level enrichments and daily aggregates.
    etf["ret_1m"] = etf["close"].astype(float).pct_change()
    etf.loc[etf["MinuteIndex"].astype(int) == 0, "ret_1m"] = np.nan
    etf["intraday_time_regime"] = pd.to_datetime(etf["datetime"]).map(intraday_bucket)
    amount_daily = etf.groupby("date", sort=True)["amount"].sum().rename("daily_amount").reset_index()
    rvol_daily = etf.groupby("date", sort=True)["ret_1m"].apply(
        lambda s: float(np.sqrt(np.nansum(np.square(s.astype(float).to_numpy()))))
    ).rename("daily_rvol").reset_index()
    close_daily = etf.groupby("date", sort=True)["close"].agg(first="first", last="last").reset_index()
    close_daily["daily_ret"] = close_daily["last"].astype(float) / close_daily["first"].astype(float) - 1.0
    daily = amount_daily.merge(rvol_daily, on="date", how="inner").merge(close_daily[["date", "daily_ret"]], on="date", how="inner")
    daily["trend_strength"] = daily["daily_ret"].abs() / daily["daily_rvol"].replace(0.0, np.nan)

    # Compute breadth/dispersion daily scalars from cached stock feature panels.
    feat_root = str(metrics["config"]["cache_roots"]["stock_panel_feature_cache_root"])
    bd_rows: list[dict] = []
    for date in test_dates:
        day_path = os.path.join(feat_root, f"{int(date)}.parquet")
        day = pd.read_parquet(day_path, columns=["MinuteIndex", "weight", "ret_1", "split"]).copy()
        day = day.loc[day["split"].astype(str) == "test"].copy()
        scalars = compute_daily_breadth_dispersion(day)
        bd_rows.append({"date": int(date), **scalars})
    bd_daily = pd.DataFrame(bd_rows)
    daily = daily.merge(bd_daily, on="date", how="inner")

    # Join all enrichments into a single minute-level base table.
    base = pred.merge(etf[["date", "datetime", "MinuteIndex", "intraday_time_regime"]], on=["date", "datetime"], how="left")
    base = base.merge(daily, on="date", how="left")
    base = base.merge(synth[["date", "datetime", "delta", "weight_coverage_pred"]], on=["date", "datetime"], how="left")
    base = base.merge(
        basket[["date", "datetime", "weight_coverage_pred"]].rename(columns={"weight_coverage_pred": "basket_weight_coverage"}),
        on=["date", "datetime"],
        how="left",
    )
    base["abs_pred"] = base["pred"].astype(float).abs()
    base["abs_basis_delta"] = base["delta"].astype(float).abs()

    # Compute regime split thresholds.
    daily_amount_median = float(daily["daily_amount"].median())
    daily_rvol_median = float(daily["daily_rvol"].median())
    trend_strength_median = float(daily["trend_strength"].median())
    dispersion_median = float(daily["dispersion_std_mean"].median())
    abs_pred_median = float(base["abs_pred"].median())
    abs_basis_median = float(base["abs_basis_delta"].median())
    coverage_median = float(base["basket_weight_coverage"].median())

    # Assign regime labels on each minute sample.
    base["amount_regime"] = np.where(base["daily_amount"].astype(float) >= daily_amount_median, "high", "low")
    base["vol_regime"] = np.where(base["daily_rvol"].astype(float) >= daily_rvol_median, "high", "low")
    base["trend_regime"] = np.where(base["trend_strength"].astype(float) >= trend_strength_median, "trend", "choppy")
    base["market_direction_regime"] = np.where(base["daily_ret"].astype(float) >= 0.0, "up", "down")
    base["prediction_strength_regime"] = np.where(base["abs_pred"].astype(float) >= abs_pred_median, "strong", "weak")
    base["basis_regime"] = np.where(base["abs_basis_delta"].astype(float) >= abs_basis_median, "large", "small")
    base["coverage_regime"] = np.where(base["basket_weight_coverage"].astype(float) >= coverage_median, "high", "low")
    base["dispersion_regime"] = np.where(base["dispersion_std_mean"].astype(float) >= dispersion_median, "dispersed", "consensus")
    base["breadth_dispersion_regime"] = base["market_direction_regime"].astype(str) + "_" + base["dispersion_regime"].astype(str)

    # Define the regime list and group orders.
    regimes: dict[str, list[str]] = {
        "amount_regime": ["low", "high"],
        "vol_regime": ["low", "high"],
        "breadth_dispersion_regime": sorted(base["breadth_dispersion_regime"].dropna().unique().tolist()),
        "basis_regime": ["small", "large"],
        "intraday_time_regime": ["morning", "afternoon", "tail"],
        "trend_regime": ["choppy", "trend"],
        "market_direction_regime": ["down", "up"],
        "prediction_strength_regime": ["weak", "strong"],
        "coverage_regime": ["low", "high"],
    }

    # Compute per-regime group metrics and small embedded charts.
    out_rows: dict[str, list[dict]] = {}
    charts: dict[str, dict[str, str]] = {}
    for regime_name, groups in regimes.items():
        rows: list[dict] = []
        for group in groups:
            part = base.loc[base[regime_name].astype(str) == str(group), ["date", "datetime", "MinuteIndex", "pred", "label"]].copy()
            if len(part) == 0:
                continue
            res = eval_group(base=part, etf_price_table=etf, horizon_minutes=horizon_minutes, group=str(group))
            rows.append(
                {
                    "group": res.group,
                    "pooled_ic": res.pooled_ic,
                    "pooled_rank_ic": res.pooled_rank_ic,
                    "daily_ic_mean": res.daily_ic_mean,
                    "daily_ic_std": res.daily_ic_std,
                    "daily_icir": res.daily_icir,
                    "positive_day_ratio": res.positive_day_ratio,
                    "n_samples": res.n_samples,
                    "n_days": res.n_days,
                    "strategy_total_return_pct": res.strategy_total_return_pct,
                    "strategy_max_drawdown_pct": res.strategy_max_drawdown_pct,
                    "excess_end_bps": res.excess_end_bps,
                }
            )
        out_rows[regime_name] = rows
        table = pd.DataFrame(rows)
        if len(table) > 0:
            charts[regime_name] = {
                "pooled_ic": plot_metric_bar(table, x="group", y="pooled_ic", title=f"{regime_name}: pooled_ic"),
                "excess_end_bps": plot_metric_bar(table, x="group", y="excess_end_bps", title=f"{regime_name}: excess_end_bps"),
            }

    # Write YAML and HTML artifacts under report/0424.
    out_dir = os.path.join(REPO_ROOT, "report", "0424")
    os.makedirs(out_dir, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    yaml_path = os.path.join(out_dir, f"regime_report_{stamp}.yaml")
    html_path = os.path.join(out_dir, f"regime_report_{stamp}.html")
    latest_html_path = os.path.join(out_dir, "regime_report_latest.html")
    write_yaml(
        yaml_path,
        {
            "run_id": run_id,
            "report_dir": report_dir,
            "etf_code_int": etf_code_int,
            "horizon_minutes": horizon_minutes,
            "selected_model": best_model,
            "thresholds": {
                "daily_amount_median": daily_amount_median,
                "daily_rvol_median": daily_rvol_median,
                "trend_strength_median": trend_strength_median,
                "dispersion_median": dispersion_median,
                "abs_pred_median": abs_pred_median,
                "abs_basis_median": abs_basis_median,
                "coverage_median": coverage_median,
            },
            "regimes": out_rows,
        },
    )

    # Build a self-contained HTML report with embedded PNGs.
    css = """
    body { font-family: -apple-system, BlinkMacSystemFont, \"Segoe UI\", Arial, \"Noto Sans\", sans-serif; margin: 18px; color: #111; font-size: 15px; line-height: 1.5; }
    h1 { font-size: 22px; margin: 0 0 6px; }
    h2 { font-size: 16px; margin: 16px 0 8px; }
    .meta { background: #f6f8fa; padding: 10px 12px; border-radius: 10px; }
    .grid { display: grid; grid-template-columns: 1fr; gap: 12px; }
    .card { border: 1px solid #e5e7eb; border-radius: 10px; padding: 10px 12px; }
    table { border-collapse: collapse; width: 100%; font-size: 12.5px; }
    th, td { border-bottom: 1px solid #eee; padding: 6px 6px; }
    th { text-align: left; background: #fff; position: sticky; top: 0; }
    td.num { text-align: right; font-variant-numeric: tabular-nums; }
    code { background: #f3f4f6; padding: 1px 4px; border-radius: 4px; }
    img { max-width: 100%; height: auto; border: 1px solid #eee; border-radius: 8px; }
    ul { margin: 8px 0; padding-left: 18px; }
    """
    cols = [
        "group",
        "pooled_ic",
        "pooled_rank_ic",
        "daily_ic_mean",
        "daily_ic_std",
        "daily_icir",
        "positive_day_ratio",
        "n_samples",
        "n_days",
        "strategy_total_return_pct",
        "strategy_max_drawdown_pct",
        "excess_end_bps",
    ]

    parts: list[str] = []
    parts.append("<!doctype html><html><head><meta charset='utf-8' />")
    parts.append("<meta name='viewport' content='width=device-width, initial-scale=1' />")
    parts.append(f"<title>Regime Research 0424 ({html_escape(run_id)})</title>")
    parts.append(f"<style>{css}</style></head><body>")
    parts.append(f"<h1>Regime 研究报告 (0424, {html_escape(run_id)})</h1>")
    parts.append("<div class='meta'>")
    parts.append(
        f"<div>ETF: <code>{int(etf_code_int)}</code>, horizon: <code>{int(horizon_minutes)}</code> min, selected_model: <code>{html_escape(best_model)}</code></div>"
    )
    parts.append(f"<div>source_run_dir: <code>{html_escape(report_dir)}</code></div>")
    parts.append(f"<div>generated_at: <code>{html_escape(stamp)}</code></div>")
    parts.append("</div>")

    parts.append("<h2>Regime 列表与口径</h2>")
    parts.append("<div class='card'>")
    parts.append("<ul>")
    parts.append("<li><code>amount_regime</code>: daily ETF amount sum median split.</li>")
    parts.append("<li><code>vol_regime</code>: daily realized vol from ETF 1m returns median split.</li>")
    parts.append("<li><code>breadth_dispersion_regime</code>: 2x2 = market direction (ETF daily ret) x dispersion (stock ret_1 weighted std) split.</li>")
    parts.append("<li><code>basis_regime</code>: median split by <code>|delta|</code>, delta = ETF label - basket label.</li>")
    parts.append("<li><code>intraday_time_regime</code>: morning/afternoon/tail by wall-clock time.</li>")
    parts.append("<li><code>trend_regime</code>: median split by <code>|daily_ret| / daily_rvol</code>.</li>")
    parts.append("<li><code>market_direction_regime</code>: up/down by ETF daily ret sign.</li>")
    parts.append("<li><code>prediction_strength_regime</code>: median split by <code>|pred|</code> on samples.</li>")
    parts.append("<li><code>coverage_regime</code>: median split by basket weight coverage (minute-level).</li>")
    parts.append("</ul>")
    parts.append("</div>")

    parts.append("<h2>结果汇总</h2>")
    parts.append("<div class='grid'>")
    for regime_name, rows in out_rows.items():
        parts.append("<div class='card'>")
        parts.append(f"<h2>{html_escape(regime_name)}</h2>")
        parts.append(render_table(rows, cols))
        if regime_name in charts:
            parts.append("<div style='display:grid;grid-template-columns:1fr;gap:10px;margin-top:10px;'>")
            parts.append(f"<img src='data:image/png;base64,{charts[regime_name]['pooled_ic']}' />")
            parts.append(f"<img src='data:image/png;base64,{charts[regime_name]['excess_end_bps']}' />")
            parts.append("</div>")
        parts.append("</div>")
    parts.append("</div>")
    parts.append("<hr /><div style='font-size:12px;color:#555'>Self-contained HTML; all plots embedded as base64 PNG.</div>")
    parts.append("</body></html>")
    html = "\n".join(parts)

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    with open(latest_html_path, "w", encoding="utf-8") as f:
        f.write(html)

    print("[INFO] wrote", html_path)
    print("[INFO] wrote", latest_html_path)
    print("[INFO] wrote", yaml_path)


if __name__ == "__main__":
    main()
