import datetime as dt
import os
import sys

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LinearRegression

# Add repo root into sys.path so local modules can be imported when running as a script.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from eval.metrics import compute_metrics  # noqa: E402


def _ensure_dir(path: str) -> None:
    """Ensure a directory exists."""

    # Create directory recursively.
    os.makedirs(path, exist_ok=True)


def _find_latest_run_dir(report_root: str) -> str:
    """Find the latest pipeline run directory under report root."""

    # Collect all run_* folders under report/*/run_*.
    candidates: list[str] = []
    for day_name in sorted(os.listdir(report_root)):
        day_dir = os.path.join(report_root, day_name)
        if not os.path.isdir(day_dir):
            continue
        for run_name in sorted(os.listdir(day_dir)):
            if not run_name.startswith("run_"):
                continue
            run_dir = os.path.join(day_dir, run_name)
            if os.path.isdir(run_dir):
                candidates.append(run_dir)

    # Sort by mtime so we pick the newest completed run.
    candidates_sorted = sorted(candidates, key=lambda p: os.path.getmtime(p), reverse=True)
    assert len(candidates_sorted) > 0
    return str(candidates_sorted[0])


def _fit_alignment_model(train: pd.DataFrame) -> LinearRegression:
    """Fit a linear stacking model y = a + b*x on train split."""

    # Fit a one-feature OLS-style regression for target alignment.
    x = train.loc[:, ["basket_pred"]].to_numpy(dtype=float)
    y = train.loc[:, "label_etf_10m"].to_numpy(dtype=float)
    model = LinearRegression()
    model.fit(x, y)
    return model


def _build_pred_table(frame: pd.DataFrame, model_name: str, pred: np.ndarray) -> pd.DataFrame:
    """Build a standardized prediction table for compute_metrics."""

    # Assemble the minimal columns required for compute_metrics.
    out = pd.DataFrame(
        {
            "datetime": pd.to_datetime(frame["datetime"]).astype("datetime64[ns]"),
            "date": frame["date"].astype(int),
            "pred": pred.astype(float),
            "label": frame["label_etf_10m"].astype(float),
            "split": frame["split"].astype(str),
            "model_name": str(model_name),
        }
    )
    return out


def main() -> None:
    """Run the target alignment experiment on the latest pipeline output."""

    # Locate the latest completed run directory as experiment input.
    report_root = os.path.join(REPO_ROOT, "report")
    base_run_dir = _find_latest_run_dir(report_root=report_root)
    base_path = os.path.join(base_run_dir, "basis_pred_vs_etf.parquet")
    assert os.path.exists(base_path)

    # Load the basis join table and keep only the columns needed for stacking.
    base = pd.read_parquet(base_path, columns=["date", "datetime", "split", "basket_pred", "label_etf_10m", "model_name"])
    base = base.loc[base["split"].isin(["train", "test"])].copy()

    # Fit an alignment model per basket branch to avoid mixing distributions.
    aligned_pred_tables: list[pd.DataFrame] = []
    aligned_metrics: dict[str, dict] = {}
    align_models: dict[str, dict] = {}
    for branch_name, part in base.groupby("model_name", sort=True):
        # Fit the alignment model on the train split only.
        train = part.loc[part["split"] == "train"].copy()
        model = _fit_alignment_model(train=train)

        # Predict aligned values on both splits.
        x_all = part.loc[:, ["basket_pred"]].to_numpy(dtype=float)
        pred_aligned = model.predict(x_all).astype(float)
        model_name = f"aligned_{str(branch_name)}_to_etf"
        aligned_pred_tables.append(_build_pred_table(frame=part, model_name=model_name, pred=pred_aligned))

        # Store coefficients for the report.
        align_models[str(branch_name)] = {
            "intercept": float(model.intercept_),
            "slope": float(model.coef_[0]),
            "n_train": int(len(train)),
        }

    # Evaluate aligned predictions with the same metrics helper as pipeline.
    pred_table = pd.concat(aligned_pred_tables, axis=0, ignore_index=True)
    aligned_metrics = compute_metrics(pred_table=pred_table)

    # Write results into a new experiment run directory.
    day_tag = dt.datetime.now().strftime("%m%d")
    run_id = dt.datetime.now().strftime("exp1_target_alignment_%Y%m%d_%H%M%S")
    out_dir = os.path.join(report_root, day_tag, run_id)
    _ensure_dir(out_dir)

    # Serialize experiment outputs as YAML + concise markdown.
    payload = {
        "base_run_dir": str(base_run_dir),
        "alignment_models": align_models,
        "metrics": aligned_metrics,
    }
    with open(os.path.join(out_dir, "target_alignment_metrics.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, allow_unicode=True, sort_keys=False)

    # Compose a small markdown report for quick reading.
    lines: list[str] = []
    lines.append(f"# 实验 1: 目标函数对齐 (Target Alignment) ({run_id})")
    lines.append("")
    lines.append(f"- base_run_dir: `{base_run_dir}`")
    lines.append(f"- input: `basis_pred_vs_etf.parquet`")
    lines.append("")
    lines.append("## 对齐模型参数 (train)")
    lines.append("")
    lines.append("| branch | intercept | slope | n_train |")
    lines.append("| --- | --- | --- | --- |")
    for branch_name in sorted(align_models.keys()):
        row = align_models[branch_name]
        lines.append(f"| {branch_name} | {row['intercept']:.8f} | {row['slope']:.6f} | {int(row['n_train'])} |")
    lines.append("")
    lines.append("## 效果 (test)")
    lines.append("")
    lines.append("| model | ic | rank_ic | dir_acc | rmse | mae | n |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for model_name in sorted(aligned_metrics["overall"].keys()):
        row = aligned_metrics["overall"][model_name]["test"]
        lines.append(
            f"| {model_name} | {float(row['ic']):.6f} | {float(row['rank_ic']):.6f} | {float(row['direction_acc']):.6f} | "
            f"{float(row['rmse']):.6f} | {float(row['mae']):.6f} | {int(row['n'])} |"
        )
    lines.append("")
    with open(os.path.join(out_dir, "target_alignment_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[INFO] Done. out_dir={out_dir}")


if __name__ == "__main__":
    main()
