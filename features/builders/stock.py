from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from features.expression import compile_expression, default_env, evaluate_compiled
from features.registry import cached_load_registry


_FACTOR_REWRITE_RULES: dict[str, dict[str, int | str]] = {
    "alpha_019": {"rewrite_class": "prev_day_tail", "history_days": 1, "history_bars": 250},
    "alpha_032": {"rewrite_class": "prev_day_tail", "history_days": 1, "history_bars": 235},
    "alpha_039": {"rewrite_class": "prev_day_tail", "history_days": 1, "history_bars": 250},
    "alpha_052": {"rewrite_class": "prev_day_tail", "history_days": 1, "history_bars": 240},
    "alpha_071": {"rewrite_class": "multi_day_roll", "history_days": 2, "history_bars": 225},
    "alpha_096": {"rewrite_class": "operator_semantics", "history_days": 1, "history_bars": 101},
}


def _panel_to_frames(base_panel: pd.DataFrame, fields: list[str]) -> dict[str, pd.DataFrame]:
    """Convert a long-form base panel into wide (time x stock) frames."""

    # Build a stable multi-index grid so stack/unstack round-trips are exact.
    idx = base_panel.loc[:, ["datetime", "stock_code"]].copy()
    idx["datetime"] = pd.to_datetime(idx["datetime"]).astype("datetime64[us]")
    idx["stock_code"] = idx["stock_code"].astype(int)
    base = base_panel.copy()
    base["datetime"] = idx["datetime"].to_numpy()
    base["stock_code"] = idx["stock_code"].to_numpy()
    base = base.set_index(["datetime", "stock_code"]).sort_index()

    # Unstack each requested field into a wide frame.
    out: dict[str, pd.DataFrame] = {}
    for field in fields:
        out[str(field)] = base[str(field)].unstack("stock_code")
    return out


def _frames_to_series(frame: pd.DataFrame, name: str) -> pd.Series:
    """Convert a wide frame into a long series aligned on (datetime, stock_code)."""

    # Stack into a multi-index series so joins are index-stable.
    s = frame.stack(future_stack=True)
    s.name = str(name)
    return s


def _apply_xs_clip_and_rank(panel: pd.DataFrame, cols: list[str], lower_q: float, upper_q: float) -> pd.DataFrame:
    """Apply cross-sectional winsorize+rank normalization per datetime."""

    # Compute per-datetime quantile bounds for all columns in one grouped pass.
    bounds = panel.groupby("datetime", sort=False)[cols].quantile([float(lower_q), float(upper_q)]).reset_index()
    bounds = bounds.rename(columns={"level_1": "q"})
    lower = bounds.loc[bounds["q"] == float(lower_q)].drop(columns=["q"]).set_index("datetime")
    upper = bounds.loc[bounds["q"] == float(upper_q)].drop(columns=["q"]).set_index("datetime")
    lower = lower.rename(columns={c: f"{c}__lo" for c in cols})
    upper = upper.rename(columns={c: f"{c}__hi" for c in cols})

    # Merge bounds back and apply clip+rank column-by-column to match existing semantics.
    merged = panel.merge(lower.reset_index(), on="datetime", how="left")
    merged = merged.merge(upper.reset_index(), on="datetime", how="left")
    for col in cols:
        merged[col] = merged[col].astype(float).clip(lower=merged[f"{col}__lo"].astype(float), upper=merged[f"{col}__hi"].astype(float))
        merged[col] = merged.groupby("datetime", sort=False)[col].rank(pct=True)
    merged = merged.drop(columns=[f"{c}__lo" for c in cols] + [f"{c}__hi" for c in cols])
    return merged


@lru_cache(maxsize=None)
def _load_stock_factor_runtime(specs_root: str, factor_set_name: str) -> dict:
    """Load and cache stock-factor runtime metadata for one factor set."""

    # Load factor registry and resolve the selected factor set.
    registry = cached_load_registry(specs_root=specs_root)
    factor_set = registry.factor_sets[str(factor_set_name)]

    # Collect required raw input fields from factor specs.
    input_fields: list[str] = []
    compiled_items: list[tuple[str, object]] = []
    for name in factor_set.factors:
        spec = registry.factor_specs[str(name)]
        input_fields.extend([str(x) for x in spec.inputs])
        compiled_items.append((str(spec.name_en), compile_expression(formula=str(spec.formula))))

    # Force-load the core OHLCVA fields used by shared derived variables.
    essential_fields = ["Open", "High", "Low", "Close", "Vol", "Amount"]
    input_fields = sorted(set(list(input_fields) + essential_fields))

    # Return a plain runtime payload for repeated per-day builds.
    return {
        "factor_names": [str(name) for name in factor_set.factors],
        "input_fields": input_fields,
        "compiled_items": compiled_items,
        "invalid_after_first_invalid_factors": [str(name) for name in factor_set.invalid_after_first_invalid_factors],
        "xs_clip_rank_factors": [str(name) for name in factor_set.xs_clip_rank_factors],
    }


@lru_cache(maxsize=None)
def get_stock_factor_history_plan(specs_root: str, factor_set_name: str) -> dict:
    """Resolve extra history requirements for rewritten stock factors."""

    # Load the active factor names once from the cached runtime.
    runtime = _load_stock_factor_runtime(specs_root=specs_root, factor_set_name=factor_set_name)
    active_factor_names = [str(name) for name in runtime["factor_names"]]

    # Select the rewrite rules that are active in the current factor set.
    active_rules = {
        str(name): dict(_FACTOR_REWRITE_RULES[str(name)]) for name in active_factor_names if str(name) in _FACTOR_REWRITE_RULES
    }

    # Summarize the maximum cross-day history requirements for the caller.
    history_days = 0
    history_bars = 0
    if len(active_rules) > 0:
        history_days = max([int(rule["history_days"]) for rule in active_rules.values()])
        history_bars = max([int(rule["history_bars"]) for rule in active_rules.values()])
    return {"active_rules": active_rules, "history_days": int(history_days), "history_bars": int(history_bars)}


def build_stock_feature_panel_day(
    base_panel: pd.DataFrame,
    specs_root: str,
    factor_set_name: str,
    history_panel: pd.DataFrame | None,
) -> pd.DataFrame:
    """Build one-day stock feature panel from a base panel using a factor set."""

    # Merge the optional history rows so long-window factors can warm up across days.
    current = base_panel.copy()
    current["__is_current__"] = 1
    if history_panel is not None and len(history_panel) > 0:
        history = history_panel.copy()
        history["weight"] = np.nan
        history["label_stock_10m"] = np.nan
        history["__is_current__"] = 0
        eval_panel = pd.concat([history, current], axis=0, ignore_index=True, sort=False)
    else:
        eval_panel = current

    # Load cached runtime metadata for the selected factor set.
    runtime = _load_stock_factor_runtime(specs_root=specs_root, factor_set_name=factor_set_name)
    factor_names = list(runtime["factor_names"])
    input_fields = list(runtime["input_fields"])

    # Convert required base fields into wide frames and precompute derived env variables.
    frames = _panel_to_frames(base_panel=eval_panel, fields=input_fields + ["invalid_bar"])
    env = default_env()
    env.update({str(k).upper(): v for k, v in frames.items() if k != "invalid_bar"})

    # Precompute OPEN0/VWAP/HIGH_SO_FAR/LOW_SO_FAR for formula convenience.
    open_first = env["OPEN"].iloc[0].to_numpy(dtype=np.float64, copy=False)
    open0 = np.tile(open_first[None, :], (len(env["OPEN"].index), 1))
    env["OPEN0"] = pd.DataFrame(open0, index=env["OPEN"].index, columns=env["OPEN"].columns)
    cum_vol = env["VOL"].fillna(0.0).cumsum()
    cum_amt = env["AMOUNT"].fillna(0.0).cumsum()
    env["VWAP"] = cum_amt.div(cum_vol)
    env["HIGH_SO_FAR"] = env["HIGH"].cummax()
    env["LOW_SO_FAR"] = env["LOW"].cummin()

    # Provide alpha101-style lowercase aliases and common derived series.
    env["open"] = env["OPEN"]
    env["high"] = env["HIGH"]
    env["low"] = env["LOW"]
    env["close"] = env["CLOSE"]
    env["volume"] = env["VOL"]
    env["amount"] = env["AMOUNT"]
    env["vwap"] = env["VWAP"]
    env["returns"] = env["CLOSE"].div(env["CLOSE"].shift(1)).sub(1.0)

    # Precompute common ADV windows as rolling means of minute volume for alpha101 formulas.
    adv_windows = [5, 10, 15, 20, 30, 40, 50, 60, 81, 120, 150, 180]
    for win in adv_windows:
        env[f"adv{int(win)}"] = env["TS_MEAN"](env["VOL"], int(win))

    # Evaluate factor expressions into wide frames.
    factor_frames: dict[str, pd.DataFrame] = {}
    for name, compiled in runtime["compiled_items"]:
        factor_frames[str(name)] = evaluate_compiled(expr=compiled, env=env)

    # Apply invalid-after-first-invalid masking for cumulative-style factors.
    invalid = frames["invalid_bar"].astype(int)
    invalid_dates = pd.Index(pd.to_datetime(invalid.index).date)
    invalid_cum = invalid.groupby(invalid_dates, sort=False).cumsum().gt(0)
    for name in runtime["invalid_after_first_invalid_factors"]:
        if str(name) in factor_frames:
            factor_frames[str(name)] = factor_frames[str(name)].where(~invalid_cum)

    # Convert the factor frames into one long-form table and join once.
    factor_series = [_frames_to_series(frame=factor_frames[str(name)], name=str(name)) for name in factor_names]
    factor_table = pd.concat(factor_series, axis=1)
    base = eval_panel.copy()
    base["datetime"] = pd.to_datetime(base["datetime"]).astype("datetime64[us]")
    base = base.set_index(["datetime", "stock_code"]).sort_index()
    base = base.join(factor_table, how="left")

    # Compute time features in long form as non-normalized columns.
    base = base.reset_index()
    base["minute_of_day"] = base["MinuteIndex"].astype(int)
    base["is_open_30min"] = (base["MinuteIndex"].astype(int) < 30).astype(int)
    base["is_close_30min"] = (base["MinuteIndex"].astype(int) >= 211).astype(int)

    # Apply cross-sectional clip+rank postprocess for selected factors only.
    xs_cols = [c for c in runtime["xs_clip_rank_factors"] if c in base.columns]
    base = _apply_xs_clip_and_rank(panel=base, cols=xs_cols, lower_q=0.01, upper_q=0.99)

    # Trim back to the current day after all cross-day factor windows are computed.
    base = base.loc[base["__is_current__"].astype(int) == 1].copy()

    # Pack into the standardized panel schema used by the training pipeline.
    out_cols = [
        "date",
        "datetime",
        "stock_code",
        "weight",
        "label_stock_10m",
        "MinuteIndex",
    ] + factor_names + ["minute_of_day", "is_open_30min", "is_close_30min"]
    out = base.loc[:, out_cols].copy()
    return out
