from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CompiledExpression:
    """Hold a compiled AST for a factor expression string."""

    source: str
    tree: ast.AST


def _rewrite_ternary(expr: str) -> str:
    """Rewrite C-style `cond ? a : b` into `WHERE(cond, a, b)` for nested expressions."""

    # Strip outer whitespace to keep scanning deterministic.
    s = expr.strip()

    # Iteratively rewrite the innermost ternary so nested cases behave correctly.
    while True:
        # Scan all '?' positions and track their parenthesis depth.
        depth = 0
        qs: list[tuple[int, int]] = []
        for i, ch in enumerate(s):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            elif ch == "?":
                qs.append((i, depth))

        # Stop when no ternary operators remain.
        if not qs:
            return s

        # Select the deepest (most nested) '?' and prefer the rightmost on ties.
        max_depth = max(d for _, d in qs)
        q = max(i for i, d in qs if d == max_depth)
        depth_q = max_depth

        # Find the matching ':' for the chosen '?' at the same parenthesis depth.
        depth = depth_q
        nested = 0
        colon = -1
        for i in range(q + 1, len(s)):
            ch = s[i]
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            elif ch == "?" and depth == depth_q:
                nested += 1
            elif ch == ":" and depth == depth_q:
                if nested == 0:
                    colon = i
                    break
                nested -= 1
        if colon < 0:
            raise ValueError(f"malformed ternary expression, missing ':' in: {expr}")

        # Find the start boundary of the ternary expression within the current nesting.
        start = 0
        depth = depth_q
        for i in range(q - 1, -1, -1):
            ch = s[i]
            if ch == "(":
                depth -= 1
            elif ch == ")":
                depth += 1
            if depth < depth_q:
                start = i + 1
                break
            if depth == depth_q and ch == ",":
                start = i + 1
                break

        # Find the end boundary of the ternary expression within the current nesting.
        end = len(s)
        depth = depth_q
        for i in range(colon + 1, len(s)):
            ch = s[i]
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth < depth_q:
                    end = i
                    break
            if depth == depth_q and ch == ",":
                end = i
                break

        # Replace the identified ternary span with a WHERE call.
        cond = s[start:q].strip()
        a = s[q + 1 : colon].strip()
        b = s[colon + 1 : end].strip()
        replaced = f"WHERE({cond}, {a}, {b})"
        s = s[:start] + replaced + s[end:]


def _rewrite_infix_ops(expr: str) -> str:
    """Rewrite non-Python infix tokens (||, &&, ^) into Python equivalents."""

    # Normalize common boolean operators used by the alpha DSL.
    s = expr.replace("||", "|").replace("&&", "&")

    # Normalize exponentiation from '^' into Python '**'.
    return s.replace("^", "**")


def compile_expression(formula: str) -> CompiledExpression:
    """Compile a formula string into a Python AST with ternaries rewritten."""

    # Normalize non-Python infix operators before any ternary rewriting.
    cleaned = _rewrite_infix_ops(formula)

    # Normalize ternary operators into a function call so AST evaluation is simple.
    rewritten = _rewrite_ternary(cleaned)

    # Parse into an expression AST.
    tree = ast.parse(rewritten, mode="eval")
    return CompiledExpression(source=rewritten, tree=tree.body)


def _align_frames(a: pd.DataFrame, b: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Align two (time x instrument) frames on the shared index and columns."""

    # Align columns first so per-instrument operations are consistent.
    aa, bb = a.align(b, join="inner", axis=1)

    # Align index next so per-time operations are consistent.
    aa, bb = aa.align(bb, join="inner", axis=0)
    return aa, bb


def _binary_op(op: str, left: Any, right: Any) -> Any:
    """Apply a binary operator with explicit pandas alignment rules."""

    # Align DataFrame/DataFrame inputs on common grid to avoid silent mis-broadcasting.
    if isinstance(left, pd.DataFrame) and isinstance(right, pd.DataFrame):
        l2, r2 = _align_frames(left, right)
        if op == "add":
            return l2.add(r2)
        if op == "sub":
            return l2.sub(r2)
        if op == "mul":
            return l2.mul(r2)
        if op == "truediv":
            return l2.div(r2)
        if op == "pow":
            return l2.pow(r2)
        if op == "lt":
            return l2.lt(r2)
        if op == "le":
            return l2.le(r2)
        if op == "gt":
            return l2.gt(r2)
        if op == "ge":
            return l2.ge(r2)
        if op == "eq":
            return l2.eq(r2)
        if op == "ne":
            return l2.ne(r2)
        if op == "and":
            return l2.astype(bool) & r2.astype(bool)
        if op == "or":
            return l2.astype(bool) | r2.astype(bool)
        raise ValueError(f"unsupported df/df op: {op}")

    # Fall back to python numeric semantics for scalar-like operands.
    if op == "add":
        return left + right
    if op == "sub":
        return left - right
    if op == "mul":
        return left * right
    if op == "truediv":
        return left / right
    if op == "pow":
        return left**right
    if op == "lt":
        return left < right
    if op == "le":
        return left <= right
    if op == "gt":
        return left > right
    if op == "ge":
        return left >= right
    if op == "eq":
        return left == right
    if op == "ne":
        return left != right
    if op == "and":
        return left & right
    if op == "or":
        return left | right
    raise ValueError(f"unsupported op: {op}")


def _where(cond: Any, a: Any, b: Any) -> Any:
    """Compute WHERE(cond, a, b) as an elementwise conditional."""

    # Require a DataFrame condition so shape is explicit and auditable.
    if not isinstance(cond, pd.DataFrame):
        raise TypeError("WHERE expects a DataFrame boolean condition in this task.")

    # Broadcast scalar a/b into frames so output shape matches cond.
    aa = a
    bb = b
    if not isinstance(aa, pd.DataFrame):
        aa = pd.DataFrame(float(aa), index=cond.index, columns=cond.columns)
    if not isinstance(bb, pd.DataFrame):
        bb = pd.DataFrame(float(bb), index=cond.index, columns=cond.columns)

    # Align frames and apply pandas where semantics.
    c2, aa2 = _align_frames(cond.astype(bool), aa)
    c2, bb2 = _align_frames(c2, bb)
    return aa2.where(c2, other=bb2)


class _Evaluator(ast.NodeVisitor):
    """Evaluate a restricted expression AST against an environment mapping."""

    def __init__(self, env: dict[str, Any]):
        # Store the evaluation environment in a plain dict for fast lookup.
        self._env = env

    def visit_Name(self, node: ast.Name) -> Any:  # noqa: N802
        # Resolve identifiers from the environment only.
        if node.id not in self._env:
            raise NameError(f"unknown name: {node.id}")
        return self._env[node.id]

    def visit_Constant(self, node: ast.Constant) -> Any:  # noqa: N802
        # Permit only numeric constants as factor expressions should be numeric.
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise TypeError(f"unsupported constant type: {type(node.value).__name__}")

    def visit_UnaryOp(self, node: ast.UnaryOp) -> Any:  # noqa: N802
        # Evaluate unary operations such as -X.
        v = self.visit(node.operand)
        if isinstance(node.op, ast.USub):
            return -v
        if isinstance(node.op, ast.UAdd):
            return +v
        raise TypeError(f"unsupported unary op: {type(node.op).__name__}")

    def visit_BinOp(self, node: ast.BinOp) -> Any:  # noqa: N802
        # Evaluate both sides and apply the requested binary operator.
        left = self.visit(node.left)
        right = self.visit(node.right)
        if isinstance(node.op, ast.Add):
            return _binary_op("add", left, right)
        if isinstance(node.op, ast.Sub):
            return _binary_op("sub", left, right)
        if isinstance(node.op, ast.Mult):
            return _binary_op("mul", left, right)
        if isinstance(node.op, ast.Div):
            return _binary_op("truediv", left, right)
        if isinstance(node.op, ast.Pow):
            return _binary_op("pow", left, right)
        if isinstance(node.op, ast.BitAnd):
            return _binary_op("and", left, right)
        if isinstance(node.op, ast.BitOr):
            return _binary_op("or", left, right)
        raise TypeError(f"unsupported bin op: {type(node.op).__name__}")

    def visit_Compare(self, node: ast.Compare) -> Any:  # noqa: N802
        # Evaluate chained comparisons left-to-right and AND them together.
        left = self.visit(node.left)
        out = None
        for op, comp in zip(node.ops, node.comparators, strict=True):
            right = self.visit(comp)
            if isinstance(op, ast.Lt):
                v = _binary_op("lt", left, right)
            elif isinstance(op, ast.LtE):
                v = _binary_op("le", left, right)
            elif isinstance(op, ast.Gt):
                v = _binary_op("gt", left, right)
            elif isinstance(op, ast.GtE):
                v = _binary_op("ge", left, right)
            elif isinstance(op, ast.Eq):
                v = _binary_op("eq", left, right)
            elif isinstance(op, ast.NotEq):
                v = _binary_op("ne", left, right)
            else:
                raise TypeError(f"unsupported compare op: {type(op).__name__}")

            out = v if out is None else (out & v)
            left = right
        return out

    def visit_Call(self, node: ast.Call) -> Any:  # noqa: N802
        # Resolve the function from the environment.
        fn = self.visit(node.func)
        if not callable(fn):
            raise TypeError("call target is not callable")

        # Evaluate positional and keyword arguments.
        args = [self.visit(a) for a in node.args]
        kwargs = {str(k.arg).lower(): self.visit(k.value) for k in node.keywords}

        # Dispatch WHERE specially so it can accept DataFrame conditions.
        if fn is _where:
            return _where(*args)

        # Call the function with normalized keyword casing.
        return fn(*args, **kwargs)

    def generic_visit(self, node: ast.AST) -> Any:
        # Fail fast on any unsupported AST nodes to keep evaluation safe and auditable.
        raise TypeError(f"unsupported expression node: {type(node).__name__}")


def evaluate_compiled(expr: CompiledExpression, env: dict[str, Any]) -> Any:
    """Evaluate a compiled expression using the provided environment mapping."""

    # Run the restricted evaluator against the AST body.
    return _Evaluator(env=env).visit(expr.tree)


def default_env() -> dict[str, Any]:
    """Build the default operator environment for minute-level factor evaluation."""

    # Import operators lazily so this module stays lightweight on import.
    from features.operators.cs import (  # noqa: WPS433
        indneutralize,
        log,
        maximum,
        minimum,
        rank,
        safe_div,
        scale,
        sign,
        signed_power,
        signed_sqrt,
    )
    from features.operators.ts import (  # noqa: WPS433
        ts_corr,
        ts_covariance,
        ts_decay_linear,
        ts_delay,
        ts_delta,
        ts_product,
        ts_rank,
        ts_argmax,
        ts_argmin,
        ts_max,
        ts_mean,
        ts_min,
        ts_returns,
        ts_std,
        ts_sum,
    )

    # Build the environment mapping with operator aliases matching the factor DSL.
    return {
        "WHERE": _where,
        "ABS": np.abs,
        "abs": np.abs,
        "Abs": np.abs,
        "SIN": np.sin,
        "SQRT": np.sqrt,
        "POW": np.power,
        "LOG": log,
        "log": log,
        "Log": log,
        "TS_RETURNS": ts_returns,
        "TS_DELAY": ts_delay,
        "DELAY": ts_delay,
        "delay": ts_delay,
        "TS_MEAN": ts_mean,
        "TS_STD": ts_std,
        "STDDEV": ts_std,
        "stddev": ts_std,
        "TS_SUM": ts_sum,
        "SUM": ts_sum,
        "sum": ts_sum,
        "Sum": ts_sum,
        "TS_MIN": ts_min,
        "TS_MAX": ts_max,
        "ts_min": ts_min,
        "ts_max": ts_max,
        "MIN": minimum,
        "min": minimum,
        "MAX": maximum,
        "max": maximum,
        "SAFE_DIV": safe_div,
        "safe_div": safe_div,
        "TS_CORR": ts_corr,
        "CORRELATION": ts_corr,
        "correlation": ts_corr,
        "Correlation": ts_corr,
        "TS_COVARIANCE": ts_covariance,
        "COVARIANCE": ts_covariance,
        "covariance": ts_covariance,
        "Covariance": ts_covariance,
        "DELTA": ts_delta,
        "delta": ts_delta,
        "Delta": ts_delta,
        "TS_PRODUCT": ts_product,
        "PRODUCT": ts_product,
        "product": ts_product,
        "Product": ts_product,
        "DECAY_LINEAR": ts_decay_linear,
        "decay_linear": ts_decay_linear,
        "Decay_Linear": ts_decay_linear,
        "TS_RANK": ts_rank,
        "ts_rank": ts_rank,
        "Ts_Rank": ts_rank,
        "TS_ARGMAX": ts_argmax,
        "ts_argmax": ts_argmax,
        "Ts_ArgMax": ts_argmax,
        "TS_ARGMIN": ts_argmin,
        "ts_argmin": ts_argmin,
        "Ts_ArgMin": ts_argmin,
        "RANK": rank,
        "rank": rank,
        "Rank": rank,
        "SCALE": scale,
        "scale": scale,
        "Scale": scale,
        "SIGN": sign,
        "sign": sign,
        "Sign": sign,
        "INDNEUTRALIZE": indneutralize,
        "indneutralize": indneutralize,
        "SIGNED_SQRT": signed_sqrt,
        "SIGNED_POWER": signed_power,
        "SignedPower": signed_power,
        "signedpower": signed_power,
        "signed_power": signed_power,
    }
