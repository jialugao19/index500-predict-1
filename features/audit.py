from __future__ import annotations

import ast
import math
import os
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
import yaml

from features.builders.stock import get_stock_factor_history_plan
from features.expression import compile_expression
from features.registry import load_registry
from stock_panel_loader import load_index_weights, load_or_build_stock_panel_day


_BARS_PER_DAY = 241
_ROLLING_ONE_ARG_OPS = {
    "ts_mean",
    "ts_std",
    "stddev",
    "ts_sum",
    "sum",
    "ts_min",
    "ts_max",
    "ts_rank",
    "ts_argmax",
    "ts_argmin",
    "ts_product",
    "decay_linear",
}
_ROLLING_TWO_ARG_OPS = {"ts_corr", "correlation", "ts_covariance", "covariance"}
_SHIFT_OPS = {"ts_delay", "delay", "ts_delta", "delta"}
_RETURN_OPS = {"ts_returns"}
_PASS_THROUGH_OPS = {
    "where",
    "rank",
    "scale",
    "sign",
    "signedpower",
    "signed_power",
    "signedsqrt",
    "signed_sqrt",
    "abs",
    "log",
    "sqrt",
    "pow",
    "max",
    "min",
    "maximum",
    "minimum",
    "safe_div",
}
_ADV_PATTERN = re.compile(r"^adv(\d+)$", re.IGNORECASE)


@dataclass(frozen=True)
class StaticFactorAudit:
    """Hold static audit metadata for one factor spec."""

    name_en: str
    scope: str
    tags: list[str]
    operators: list[str]
    numeric_risks: list[str]
    static_history_bars_required: int
    static_history_days_required: int
    rewrite_class: str
    rewrite_history_bars_required: int
    rewrite_history_days_required: int
    effective_history_bars_required: int
    effective_history_days_required: int


@dataclass(frozen=True)
class DynamicFactorAudit:
    """Hold dynamic audit metrics for one factor over sampled stock panels."""

    name_en: str
    total_count: int
    finite_count: int
    inf_count: int
    nan_count: int
    finite_ratio: float
    sample_total_count: int
    sample_finite_count: int
    sample_inf_count: int
    sample_nan_count: int
    sample_finite_ratio: float
    finite_mean: float | None
    finite_std: float | None
    gate_status: str


@dataclass(frozen=True)
class _AstAuditState:
    """Hold recursive AST audit state for one expression subtree."""

    history_bars_required: int
    operators: tuple[str, ...]
    numeric_risks: tuple[str, ...]


def _dedup_in_order(items: list[str]) -> list[str]:
    """Deduplicate strings while preserving first-seen order."""

    # Track first-seen items in one forward pass.
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        key = str(item)
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def _merge_ast_states(states: list[_AstAuditState]) -> _AstAuditState:
    """Merge multiple subtree states into one conservative parent state."""

    # Take the largest history requirement across all children.
    history_bars_required = 0
    if len(states) > 0:
        history_bars_required = max([int(state.history_bars_required) for state in states])

    # Union operators and numeric risks in stable order.
    operators = _dedup_in_order([name for state in states for name in state.operators])
    numeric_risks = _dedup_in_order([name for state in states for name in state.numeric_risks])
    return _AstAuditState(
        history_bars_required=int(history_bars_required),
        operators=tuple(operators),
        numeric_risks=tuple(numeric_risks),
    )


def _call_name(node: ast.AST) -> str:
    """Extract a lowercase function name from an AST call target."""

    # Resolve plain-name function calls directly.
    if isinstance(node, ast.Name):
        return str(node.id).lower()

    # Fail fast on unsupported call target shapes.
    raise TypeError(f"unsupported call target: {type(node).__name__}")


def _const_int(node: ast.AST) -> int:
    """Extract one runtime integer constant from an AST node."""

    # Parse plain numeric constants with runtime-consistent int coercion.
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return int(float(node.value))

    # Parse signed numeric constants with runtime-consistent int coercion.
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)) and isinstance(node.operand, ast.Constant):
        value = float(node.operand.value)
        if isinstance(node.op, ast.USub):
            value = -value
        return int(value)

    # Fail fast when the window argument is not a static constant.
    raise TypeError(f"window argument must be numeric constant, got {type(node).__name__}")


def _is_nonzero_numeric_constant(node: ast.AST) -> bool:
    """Check whether an AST node is a non-zero numeric constant."""

    # Accept plain numeric constants directly.
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value) != 0.0

    # Accept signed numeric constants after evaluating the unary sign.
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)) and isinstance(node.operand, ast.Constant):
        value = float(node.operand.value)
        if isinstance(node.op, ast.USub):
            value = -value
        return float(value) != 0.0

    # Treat any other expression as non-constant for risk purposes.
    return False


def _bars_to_extra_days(history_bars_required: int) -> int:
    """Convert required history bars into extra prior trading days."""

    # Return zero when one trading day of bars is enough.
    bars = int(history_bars_required)
    if bars <= _BARS_PER_DAY:
        return 0

    # Compute the number of full prior days needed beyond the current day.
    return int(math.ceil((bars - _BARS_PER_DAY) / float(_BARS_PER_DAY)))


def _analyze_ast(node: ast.AST) -> _AstAuditState:
    """Analyze an expression subtree for operators, risks, and history demand."""

    # Treat bare variable names as zero-lookback leaves except ADV aliases.
    if isinstance(node, ast.Name):
        adv_match = _ADV_PATTERN.match(str(node.id))
        if adv_match is not None:
            adv_window = int(adv_match.group(1))
            return _AstAuditState(
                history_bars_required=int(adv_window),
                operators=(str(node.id).lower(),),
                numeric_risks=tuple(),
            )
        return _AstAuditState(history_bars_required=0, operators=tuple(), numeric_risks=tuple())

    # Treat numeric constants as zero-lookback leaves.
    if isinstance(node, ast.Constant):
        return _AstAuditState(history_bars_required=0, operators=tuple(), numeric_risks=tuple())

    # Propagate unary expressions from their operand.
    if isinstance(node, ast.UnaryOp):
        return _analyze_ast(node.operand)

    # Merge binary expressions and flag direct division risk.
    if isinstance(node, ast.BinOp):
        left_state = _analyze_ast(node.left)
        right_state = _analyze_ast(node.right)
        merged = _merge_ast_states([left_state, right_state])
        if isinstance(node.op, ast.Div):
            risks = list(merged.numeric_risks)
            if not _is_nonzero_numeric_constant(node.right):
                risks = _dedup_in_order(risks + ["division_by_zero_risk"])
            return _AstAuditState(
                history_bars_required=int(merged.history_bars_required),
                operators=tuple(merged.operators),
                numeric_risks=tuple(risks),
            )
        return merged

    # Merge boolean expressions conservatively across all values.
    if isinstance(node, ast.BoolOp):
        child_states = [_analyze_ast(value) for value in node.values]
        return _merge_ast_states(child_states)

    # Merge comparison expressions conservatively across both sides.
    if isinstance(node, ast.Compare):
        child_states = [_analyze_ast(node.left)] + [_analyze_ast(comp) for comp in node.comparators]
        return _merge_ast_states(child_states)

    # Analyze function calls with operator-specific history rules.
    if isinstance(node, ast.Call):
        func_name = _call_name(node.func)
        arg_states = [_analyze_ast(arg) for arg in node.args]
        merged = _merge_ast_states(arg_states)

        # Estimate one-argument rolling operators from the first argument and window.
        if func_name in _ROLLING_ONE_ARG_OPS:
            window = _const_int(node.args[1])
            history_bars_required = int(arg_states[0].history_bars_required) + max(int(window) - 1, 0)
            operators = _dedup_in_order(list(merged.operators) + [func_name])
            return _AstAuditState(
                history_bars_required=int(history_bars_required),
                operators=tuple(operators),
                numeric_risks=tuple(merged.numeric_risks),
            )

        # Estimate rolling pair operators from both arguments and their window.
        if func_name in _ROLLING_TWO_ARG_OPS:
            window = _const_int(node.args[2])
            child_history = max([int(arg_states[0].history_bars_required), int(arg_states[1].history_bars_required)])
            history_bars_required = int(child_history) + max(int(window) - 1, 0)
            operators = _dedup_in_order(list(merged.operators) + [func_name])
            return _AstAuditState(
                history_bars_required=int(history_bars_required),
                operators=tuple(operators),
                numeric_risks=tuple(merged.numeric_risks),
            )

        # Estimate shift-like operators from the shifted child and lag length.
        if func_name in _SHIFT_OPS:
            periods = _const_int(node.args[1])
            history_bars_required = int(arg_states[0].history_bars_required) + max(int(periods), 0)
            operators = _dedup_in_order(list(merged.operators) + [func_name])
            return _AstAuditState(
                history_bars_required=int(history_bars_required),
                operators=tuple(operators),
                numeric_risks=tuple(merged.numeric_risks),
            )

        # Estimate return operators from the underlying price history and return lag.
        if func_name in _RETURN_OPS:
            periods = _const_int(node.args[1])
            history_bars_required = int(arg_states[0].history_bars_required) + max(int(periods), 0)
            operators = _dedup_in_order(list(merged.operators) + [func_name])
            return _AstAuditState(
                history_bars_required=int(history_bars_required),
                operators=tuple(operators),
                numeric_risks=tuple(merged.numeric_risks),
            )

        # Pass through non-windowed operators while preserving any operator-specific risks.
        if func_name in _PASS_THROUGH_OPS:
            numeric_risks = list(merged.numeric_risks)
            if func_name in {"max", "min", "maximum", "minimum"}:
                numeric_risks = _dedup_in_order(numeric_risks + ["nan_semantics_sensitive"])
            if func_name in {"log"}:
                numeric_risks = _dedup_in_order(numeric_risks + ["log_domain_risk"])
            if func_name in {"sqrt"}:
                numeric_risks = _dedup_in_order(numeric_risks + ["sqrt_domain_risk"])
            operators = _dedup_in_order(list(merged.operators) + [func_name])
            return _AstAuditState(
                history_bars_required=int(merged.history_bars_required),
                operators=tuple(operators),
                numeric_risks=tuple(numeric_risks),
            )

        # Record unknown operators explicitly so audit output exposes unsupported semantics.
        operators = _dedup_in_order(list(merged.operators) + [func_name])
        numeric_risks = _dedup_in_order(list(merged.numeric_risks) + [f"unknown_operator:{func_name}"])
        return _AstAuditState(
            history_bars_required=int(merged.history_bars_required),
            operators=tuple(operators),
            numeric_risks=tuple(numeric_risks),
        )

    # Fail fast on unsupported AST nodes so audit cannot silently under-estimate history.
    raise TypeError(f"unsupported AST node for audit: {type(node).__name__}")


def build_factor_static_audit(specs_root: str, factor_name: str, factor_set_name: str | None) -> StaticFactorAudit:
    """Build one static audit row for a factor spec."""

    # Load registry objects once and resolve the target factor spec.
    registry = load_registry(specs_root=specs_root)
    spec = registry.factor_specs[str(factor_name)]

    # Compile and analyze the factor AST under current runtime semantics.
    compiled = compile_expression(formula=str(spec.formula))
    ast_state = _analyze_ast(compiled.tree)

    # Pull any explicit runtime rewrite override from the stock history plan.
    rewrite_class = "same_day"
    rewrite_history_bars_required = 0
    rewrite_history_days_required = 0
    if factor_set_name is not None and "stock" in [str(tag) for tag in spec.tags]:
        history_plan = get_stock_factor_history_plan(specs_root=specs_root, factor_set_name=factor_set_name)
        rewrite_rule = history_plan["active_rules"].get(str(spec.name_en), {})
        if len(rewrite_rule) > 0:
            rewrite_class = str(rewrite_rule["rewrite_class"])
            rewrite_history_bars_required = int(rewrite_rule["history_bars"])
            rewrite_history_days_required = int(rewrite_rule["history_days"])

    # Compute static and effective day-level history requirements.
    static_history_bars_required = int(ast_state.history_bars_required)
    static_history_days_required = _bars_to_extra_days(history_bars_required=static_history_bars_required)
    effective_history_bars_required = max(int(static_history_bars_required), int(rewrite_history_bars_required))
    effective_history_days_required = max(int(static_history_days_required), int(rewrite_history_days_required))

    # Emit one immutable static audit row for manifest generation.
    return StaticFactorAudit(
        name_en=str(spec.name_en),
        scope=str(spec.tags[0]) if len(spec.tags) > 0 else "unknown",
        tags=[str(tag) for tag in spec.tags],
        operators=list(ast_state.operators),
        numeric_risks=list(ast_state.numeric_risks),
        static_history_bars_required=int(static_history_bars_required),
        static_history_days_required=int(static_history_days_required),
        rewrite_class=str(rewrite_class),
        rewrite_history_bars_required=int(rewrite_history_bars_required),
        rewrite_history_days_required=int(rewrite_history_days_required),
        effective_history_bars_required=int(effective_history_bars_required),
        effective_history_days_required=int(effective_history_days_required),
    )


def _history_day_bucket(history_days_required: int) -> str:
    """Map one day requirement into a compact grouping key."""

    # Separate same-day, one-day, and multi-day requirements.
    days = int(history_days_required)
    if days <= 0:
        return "day_0"
    if days == 1:
        return "day_1"
    return "day_2_plus"


def build_factor_set_static_audit(specs_root: str, factor_set_name: str) -> dict:
    """Build static audit output for one factor set."""

    # Load the registry and enumerate the selected factor set in configured order.
    registry = load_registry(specs_root=specs_root)
    factor_set = registry.factor_sets[str(factor_set_name)]
    rows = [
        build_factor_static_audit(specs_root=specs_root, factor_name=str(name), factor_set_name=factor_set_name)
        for name in factor_set.factors
    ]

    # Group factors by effective extra days for downstream cache planning.
    groups_by_history_days: dict[str, list[str]] = {"day_0": [], "day_1": [], "day_2_plus": []}
    groups_by_rewrite_class: dict[str, list[str]] = {}
    for row in rows:
        groups_by_history_days[_history_day_bucket(history_days_required=int(row.effective_history_days_required))].append(str(row.name_en))
        groups_by_rewrite_class.setdefault(str(row.rewrite_class), []).append(str(row.name_en))

    # Emit a YAML-friendly static audit payload.
    return {
        "factor_set_name": str(factor_set.name),
        "scope": str(factor_set.scope),
        "bars_per_day": int(_BARS_PER_DAY),
        "groups_by_history_days": groups_by_history_days,
        "groups_by_rewrite_class": groups_by_rewrite_class,
        "factors": [row.__dict__ for row in rows],
    }


def build_factor_library_static_audit(specs_root: str) -> dict:
    """Build static audit output for every factor spec in the library."""

    # Load the registry and audit every factor against the broadest compatible set when available.
    registry = load_registry(specs_root=specs_root)
    rows: list[StaticFactorAudit] = []
    for factor_name, spec in sorted(registry.factor_specs.items()):
        factor_set_name = None
        if "stock" in [str(tag) for tag in spec.tags] and "stock_all" in registry.factor_sets:
            factor_set_name = "stock_all"
        if "etf" in [str(tag) for tag in spec.tags] and "etf_all" in registry.factor_sets:
            factor_set_name = "etf_all"
        rows.append(build_factor_static_audit(specs_root=specs_root, factor_name=str(factor_name), factor_set_name=factor_set_name))

    # Group all factors by effective extra days for intake review.
    groups_by_history_days: dict[str, list[str]] = {"day_0": [], "day_1": [], "day_2_plus": []}
    for row in rows:
        groups_by_history_days[_history_day_bucket(history_days_required=int(row.effective_history_days_required))].append(str(row.name_en))

    # Emit a YAML-friendly library-wide audit payload.
    return {
        "bars_per_day": int(_BARS_PER_DAY),
        "groups_by_history_days": groups_by_history_days,
        "factors": [row.__dict__ for row in rows],
    }


def _empty_dynamic_state() -> dict:
    """Build one mutable accumulator for dynamic factor metrics."""

    # Initialize all streaming counters and moments explicitly.
    return {
        "total_count": 0,
        "finite_count": 0,
        "inf_count": 0,
        "nan_count": 0,
        "finite_sum": 0.0,
        "finite_sumsq": 0.0,
        "sample_total_count": 0,
        "sample_finite_count": 0,
        "sample_inf_count": 0,
        "sample_nan_count": 0,
    }


def _update_dynamic_state(state: dict, values: np.ndarray) -> None:
    """Update one dynamic accumulator from a float numpy vector."""

    # Count finite, inf, and nan values for coverage diagnostics.
    arr = values.astype(float, copy=False)
    finite_mask = np.isfinite(arr)
    inf_mask = np.isinf(arr)
    nan_mask = ~finite_mask & ~inf_mask
    finite_values = arr[finite_mask]

    # Update counters and streaming moments from this batch.
    state["total_count"] += int(arr.size)
    state["finite_count"] += int(finite_mask.sum())
    state["inf_count"] += int(inf_mask.sum())
    state["nan_count"] += int(nan_mask.sum())
    state["finite_sum"] += float(finite_values.sum()) if finite_values.size > 0 else 0.0
    state["finite_sumsq"] += float(np.square(finite_values).sum()) if finite_values.size > 0 else 0.0


def _finalize_dynamic_audit(name_en: str, state: dict) -> DynamicFactorAudit:
    """Finalize one dynamic accumulator into an immutable audit row."""

    # Compute finite ratios for both full data and sampled data.
    total_count = int(state["total_count"])
    finite_count = int(state["finite_count"])
    sample_total_count = int(state["sample_total_count"])
    sample_finite_count = int(state["sample_finite_count"])
    finite_ratio = float(finite_count) / float(total_count) if total_count > 0 else float("nan")
    sample_finite_ratio = float(sample_finite_count) / float(sample_total_count) if sample_total_count > 0 else float("nan")

    # Compute a population-standard-deviation estimate from streaming moments.
    finite_mean: float | None = None
    finite_std: float | None = None
    if finite_count > 0:
        finite_mean = float(state["finite_sum"]) / float(finite_count)
        variance = float(state["finite_sumsq"]) / float(finite_count) - float(finite_mean) ** 2
        finite_std = float(math.sqrt(max(float(variance), 0.0)))

    # Apply a simple pass/warn/fail gate for factor intake.
    gate_status = "pass"
    if finite_count == 0 or int(state["inf_count"]) > 0:
        gate_status = "fail"
    elif finite_std is None or float(finite_std) == 0.0:
        gate_status = "fail"
    elif sample_total_count > 0 and sample_finite_ratio < 0.8:
        gate_status = "warn"

    # Emit one immutable dynamic audit row.
    return DynamicFactorAudit(
        name_en=str(name_en),
        total_count=int(total_count),
        finite_count=int(finite_count),
        inf_count=int(state["inf_count"]),
        nan_count=int(state["nan_count"]),
        finite_ratio=float(finite_ratio),
        sample_total_count=int(sample_total_count),
        sample_finite_count=int(sample_finite_count),
        sample_inf_count=int(state["sample_inf_count"]),
        sample_nan_count=int(state["sample_nan_count"]),
        sample_finite_ratio=float(sample_finite_ratio),
        finite_mean=finite_mean,
        finite_std=finite_std,
        gate_status=str(gate_status),
    )


def audit_stock_factor_set_dynamic(
    specs_root: str,
    factor_set_name: str,
    dates: list[int],
    weight_path: str,
    stock1m_root: str,
    horizon_minutes: int,
    base_cache_root: str,
    feature_cache_root: str,
    sample_mod: int,
) -> dict:
    """Run dynamic stock-factor audit over a list of cached or freshly built trading days."""

    # Load static metadata and initialize one accumulator per factor.
    registry = load_registry(specs_root=specs_root)
    factor_names = [str(name) for name in registry.factor_sets[str(factor_set_name)].factors]
    weights = load_index_weights(weight_path=weight_path)
    states = {str(name): _empty_dynamic_state() for name in factor_names}

    # Iterate requested dates and update full-data plus sampled-data metrics.
    for date in dates:
        day = load_or_build_stock_panel_day(
            date=int(date),
            weights=weights,
            stock1m_root=stock1m_root,
            horizon_minutes=horizon_minutes,
            base_cache_root=base_cache_root,
            feature_cache_root=feature_cache_root,
            specs_root=specs_root,
            factor_set_name=factor_set_name,
        )
        key = (
            day["date"].astype(np.int64) * 1_000_003
            + day["stock_code"].astype(np.int64) * 10_007
            + day["MinuteIndex"].astype(np.int64) * 10_009
        )
        sample_mask = ((key % int(sample_mod)).astype(int) == 0).to_numpy(dtype=bool)

        # Update every factor accumulator from both the full and sampled slices.
        for factor_name in factor_names:
            full_values = day[str(factor_name)].astype(float).to_numpy(dtype=float)
            sample_values = full_values[sample_mask]
            _update_dynamic_state(state=states[str(factor_name)], values=full_values)
            states[str(factor_name)]["sample_total_count"] += int(sample_values.size)
            states[str(factor_name)]["sample_finite_count"] += int(np.isfinite(sample_values).sum())
            states[str(factor_name)]["sample_inf_count"] += int(np.isinf(sample_values).sum())
            states[str(factor_name)]["sample_nan_count"] += int((~np.isfinite(sample_values) & ~np.isinf(sample_values)).sum())

    # Finalize all per-factor rows and summarize gate counts.
    rows = [_finalize_dynamic_audit(name_en=str(name), state=states[str(name)]) for name in factor_names]
    status_counts: dict[str, int] = {}
    for row in rows:
        status_counts[str(row.gate_status)] = int(status_counts.get(str(row.gate_status), 0) + 1)

    # Emit a YAML-friendly dynamic audit payload.
    return {
        "factor_set_name": str(factor_set_name),
        "dates": [int(date) for date in dates],
        "sample_mod": int(sample_mod),
        "status_counts": status_counts,
        "factors": [row.__dict__ for row in rows],
    }


def build_stock_factor_audit_report(
    specs_root: str,
    factor_set_name: str,
    dates: list[int],
    weight_path: str,
    stock1m_root: str,
    horizon_minutes: int,
    base_cache_root: str,
    feature_cache_root: str,
    sample_mod: int,
) -> dict:
    """Build one combined stock-factor audit report with static and dynamic sections."""

    # Build the static section first so history groups are explicit.
    static_audit = build_factor_set_static_audit(specs_root=specs_root, factor_set_name=factor_set_name)

    # Build the dynamic section next so gate results sit next to the static plan.
    dynamic_audit = audit_stock_factor_set_dynamic(
        specs_root=specs_root,
        factor_set_name=factor_set_name,
        dates=dates,
        weight_path=weight_path,
        stock1m_root=stock1m_root,
        horizon_minutes=horizon_minutes,
        base_cache_root=base_cache_root,
        feature_cache_root=feature_cache_root,
        sample_mod=sample_mod,
    )

    # Merge per-factor static and dynamic rows by factor name.
    dynamic_rows = {str(row["name_en"]): row for row in dynamic_audit["factors"]}
    merged_rows: list[dict] = []
    for static_row in static_audit["factors"]:
        merged = dict(static_row)
        merged.update(dynamic_rows[str(static_row["name_en"])])
        merged_rows.append(merged)

    # Emit the combined YAML-friendly report payload.
    return {
        "factor_set_name": str(factor_set_name),
        "scope": str(static_audit["scope"]),
        "bars_per_day": int(_BARS_PER_DAY),
        "groups_by_history_days": static_audit["groups_by_history_days"],
        "groups_by_rewrite_class": static_audit["groups_by_rewrite_class"],
        "dynamic_status_counts": dynamic_audit["status_counts"],
        "dates": [int(date) for date in dates],
        "sample_mod": int(sample_mod),
        "factors": merged_rows,
    }


def write_factor_audit_yaml(path: str, payload: dict) -> None:
    """Write one factor audit payload as YAML."""

    # Ensure the parent directory exists before writing.
    parent = os.path.dirname(path)
    if len(parent) > 0:
        os.makedirs(parent, exist_ok=True)

    # Serialize the payload with stable key order preserved from construction.
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, allow_unicode=True, sort_keys=False)
