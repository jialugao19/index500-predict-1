import os
import sys

import pandas as pd
import yaml

# Add repo root into sys.path so local modules can be imported when running as a script.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from eval.metrics import compute_metrics  # noqa: E402
from eval.plots import (  # noqa: E402
    compute_nonoverlap_backtest_summary,
    plot_etf_backtest_compare,
    plot_prediction_bucket_calibration_spread,
    plot_rolling_ic_rankic,
)
from eval.writers import write_bottom_up_report_html, write_bottom_up_report_md, write_yaml  # noqa: E402
from pipeline import load_etf_minute_bars  # noqa: E402


def _read_yaml(path: str) -> dict:
    """Read a YAML file into a dict."""

    # Load the file content.
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    """Refresh report figures and HTML for the current linked run."""

    # Resolve the current run from the stable metrics symlink.
    metrics_path = os.path.join(REPO_ROOT, "metrics.yaml")
    metrics = _read_yaml(metrics_path)
    report_dir = str(metrics["config"]["report_dir"])
    run_id = str(metrics["config"]["run_id"])

    # Load ETF prediction artifacts and recompute rolling metrics.
    pred_path = os.path.join(report_dir, "predictions.parquet")
    pred_table = pd.read_parquet(pred_path)
    etf_metrics = compute_metrics(pred_table=pred_table)
    best_model = str(metrics["selection_etf"]["selected_model"])
    best_daily = pd.DataFrame(etf_metrics["rolling"][best_model]).copy()
    etf_code_int = int(metrics["config"]["etf_code_int"])
    etf1m_root = str(metrics["config"]["data_roots"]["etf1m_root"])

    # Load the full ETF price panel for the test period.
    test_dates = sorted(
        pred_table.loc[pred_table["split"].astype(str) == "test", "date"].astype(int).drop_duplicates().tolist()
    )
    etf_price_frames: list[pd.DataFrame] = []
    for date in test_dates:
        # Load one full ETF 1m day for benchmark plotting.
        etf_day = load_etf_minute_bars(etf1m_root=etf1m_root, date=int(date), etf_code_int=etf_code_int)
        etf_price_frames.append(
            etf_day.loc[:, ["Date", "DateTime", "Close"]].rename(
                columns={"Date": "date", "DateTime": "datetime", "Close": "close"}
            )
        )
    etf_price_table = pd.concat(etf_price_frames, axis=0, ignore_index=True)

    # Rebuild the ETF diagnostic figures.
    plot_etf_backtest_compare(
        pred_table=pred_table,
        model_name=best_model,
        out_path=os.path.join(report_dir, "fig_etf_backtest_compare_test.png"),
        etf_price_table=etf_price_table,
        horizon_minutes=int(metrics["config"]["label_horizon_minutes"]),
    )
    backtest_summary = compute_nonoverlap_backtest_summary(
        pred_table=pred_table,
        model_name=best_model,
        etf_price_table=etf_price_table,
        horizon_minutes=int(metrics["config"]["label_horizon_minutes"]),
    )
    plot_rolling_ic_rankic(
        daily=best_daily,
        title=f"ETF Rolling IC / RankIC ({best_model}, test)",
        out_path=os.path.join(report_dir, "fig_etf_rolling_ic_rankic_test.png"),
    )
    plot_prediction_bucket_calibration_spread(
        pred_table=pred_table,
        model_name=best_model,
        out_path=os.path.join(report_dir, "fig_etf_pred_bucket_calibration_spread_test.png"),
        n_buckets=10,
    )

    # Refresh the selected-model summary in metrics.yaml.
    metrics["selection_etf"]["nonoverlap_backtest_test"] = backtest_summary
    write_yaml(path=os.path.join(report_dir, "metrics.yaml"), obj=metrics)

    # Rebuild the markdown and HTML reports.
    write_bottom_up_report_md(report_dir=report_dir, run_id=run_id, metrics=metrics)
    write_bottom_up_report_html(report_dir=report_dir, run_id=run_id, metrics=metrics)
    print(f"[INFO] Done. report_dir={report_dir}")


if __name__ == "__main__":
    main()
