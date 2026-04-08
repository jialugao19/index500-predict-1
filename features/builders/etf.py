from __future__ import annotations

from functools import lru_cache

import pandas as pd

from features.expression import compile_expression, default_env, evaluate_compiled
from features.registry import cached_load_registry


@lru_cache(maxsize=None)
def _load_etf_factor_runtime(specs_root: str, factor_set_name: str) -> list[tuple[str, object]]:
    """Load and cache ETF-factor runtime metadata for one factor set."""

    # Load factor registry and resolve the selected ETF factor set.
    registry = cached_load_registry(specs_root=specs_root)
    factor_set = registry.factor_sets[str(factor_set_name)]

    # Compile expressions once for repeated per-day builds.
    compiled_items: list[tuple[str, object]] = []
    for name in factor_set.factors:
        spec = registry.factor_specs[str(name)]
        compiled_items.append((str(spec.name_en), compile_expression(formula=str(spec.formula))))
    return compiled_items


def build_etf_features_day(
    etf_day: pd.DataFrame,
    specs_root: str,
    factor_set_name: str,
) -> pd.DataFrame:
    """Build one-day ETF feature frame from bars using a factor set."""

    # Load cached runtime metadata for the selected ETF factor set.
    compiled_items = _load_etf_factor_runtime(specs_root=specs_root, factor_set_name=factor_set_name)

    # Build a stable wide frame with a single column so the expression engine can reuse stock operators.
    day = etf_day.loc[:, ["DateTime", "Date", "Close", "Vol", "Amount"]].copy()
    day = day.rename(columns={"DateTime": "datetime", "Date": "date"})
    day["datetime"] = pd.to_datetime(day["datetime"]).astype("datetime64[us]")
    day = day.sort_values("datetime", ascending=True)
    day = day.set_index("datetime")
    close = day.loc[:, ["Close"]].rename(columns={"Close": 0})
    vol = day.loc[:, ["Vol"]].rename(columns={"Vol": 0})
    amt = day.loc[:, ["Amount"]].rename(columns={"Amount": 0})

    # Assemble the expression environment with uppercase input names.
    env = default_env()
    env["CLOSE"] = close
    env["VOL"] = vol
    env["AMOUNT"] = amt

    # Evaluate factor expressions into a dict of series so output stays long-form and simple.
    out = day.loc[:, ["date"]].copy()
    for name, compiled in compiled_items:
        frame = evaluate_compiled(expr=compiled, env=env)
        out[str(name)] = frame.iloc[:, 0].astype(float)

    # Emit a dataframe aligned with the original minute bars for joins in pipeline.
    out = out.reset_index().rename(columns={"datetime": "DateTime"})
    out = out.rename(columns={"date": "Date"})
    return out
