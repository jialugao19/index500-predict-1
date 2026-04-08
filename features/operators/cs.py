from __future__ import annotations

import numpy as np
import pandas as pd

from features.operators.utils import as_float_frame


def log(x: pd.DataFrame) -> pd.DataFrame:
    """Compute LOG(X) elementwise as natural log with inf mapped to NaN."""

    # Normalize input so existing inf values become NaN before transform.
    y = as_float_frame(x)

    # Apply natural logarithm; non-positive values yield NaN/inf which we normalize next.
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.log(y.to_numpy(dtype=np.float64, copy=False))
    return as_float_frame(pd.DataFrame(out, index=y.index, columns=y.columns))


def sign(x: pd.DataFrame) -> pd.DataFrame:
    """Compute SIGN(X) elementwise as -1/0/1 on float64 inputs."""

    # Normalize inputs so sign is applied on a stable float64 array.
    y = as_float_frame(x)

    # Apply numpy sign elementwise and preserve the DataFrame shape.
    out = np.sign(y.to_numpy(dtype=np.float64, copy=False))
    return pd.DataFrame(out, index=y.index, columns=y.columns)


def rank(x: pd.DataFrame) -> pd.DataFrame:
    """Compute RANK(X) as cross-sectional percentile rank at each timestamp."""

    # Normalize inputs so rank is computed over float values consistently.
    y = as_float_frame(x)

    # Compute per-row percentile ranks across instruments (columns).
    return y.rank(axis=1, pct=True)


def scale(x: pd.DataFrame, a: float = 1.0) -> pd.DataFrame:
    """Compute SCALE(X, a) so sum(abs(x)) per row equals a."""

    # Normalize inputs so abs/sum behave consistently on float64.
    y = as_float_frame(x)

    # Compute row-wise scale factors and rescale with an explicit zero-denominator guard.
    target = float(a)
    denom = y.abs().sum(axis=1)
    factor = target / denom.where(denom.ne(0.0))
    return y.mul(factor, axis=0)


def indneutralize(x: pd.DataFrame, g: pd.Series) -> pd.DataFrame:
    """Compute INDNEUTRALIZE(X, g) by demeaning within each group for every timestamp."""

    # Normalize inputs and align group labels to the current column universe.
    y = as_float_frame(x)
    groups = g.reindex(y.columns)

    # Demean each row within group buckets using a groupby on the transposed frame.
    out = y.T.groupby(groups, sort=False).transform(lambda df: df.sub(df.mean(axis=0))).T
    return as_float_frame(out)


def safe_div(a: pd.DataFrame | float, b: pd.DataFrame | float) -> pd.DataFrame:
    """Compute SAFE_DIV(A, B) with zero or non-finite denominators mapped to NaN."""

    # Broadcast scalar inputs into a DataFrame so output shape is explicit.
    aa = a
    bb = b
    if not isinstance(aa, pd.DataFrame) and isinstance(bb, pd.DataFrame):
        aa = pd.DataFrame(float(aa), index=bb.index, columns=bb.columns)
    if not isinstance(bb, pd.DataFrame) and isinstance(aa, pd.DataFrame):
        bb = pd.DataFrame(float(bb), index=aa.index, columns=aa.columns)

    # Align frames and compute a nan-preserving guarded division.
    if isinstance(aa, pd.DataFrame) and isinstance(bb, pd.DataFrame):
        aa2, bb2 = aa.align(bb, join="inner", axis=1)
        aa2, bb2 = aa2.align(bb2, join="inner", axis=0)
        num = aa2.to_numpy(dtype=np.float64, copy=False)
        den = bb2.to_numpy(dtype=np.float64, copy=False)
        out = np.full_like(num, np.nan)
        ok = np.isfinite(num) & np.isfinite(den) & (den != 0.0)
        np.divide(num, den, out=out, where=ok)
        return pd.DataFrame(out, index=aa2.index, columns=aa2.columns)

    # Fail fast when neither input is a DataFrame, as specs should return frames.
    raise TypeError("SAFE_DIV expects at least one DataFrame input in this task.")


def minimum(a: pd.DataFrame | float, b: pd.DataFrame | float) -> pd.DataFrame:
    """Compute MIN(A, B) elementwise with DataFrame/scalar broadcasting."""

    # Broadcast scalar inputs into a DataFrame so output shape is explicit.
    aa = a
    bb = b
    if not isinstance(aa, pd.DataFrame) and isinstance(bb, pd.DataFrame):
        aa = pd.DataFrame(float(aa), index=bb.index, columns=bb.columns)
    if not isinstance(bb, pd.DataFrame) and isinstance(aa, pd.DataFrame):
        bb = pd.DataFrame(float(bb), index=aa.index, columns=aa.columns)

    # Align frames and compute elementwise minimum with nan-aware semantics.
    if isinstance(aa, pd.DataFrame) and isinstance(bb, pd.DataFrame):
        aa2, bb2 = aa.align(bb, join="inner", axis=1)
        aa2, bb2 = aa2.align(bb2, join="inner", axis=0)
        out = np.fmin(aa2.to_numpy(dtype=np.float64, copy=False), bb2.to_numpy(dtype=np.float64, copy=False))
        return pd.DataFrame(out, index=aa2.index, columns=aa2.columns)

    # Fail fast when neither input is a DataFrame, as specs should return frames.
    raise TypeError("MIN expects at least one DataFrame input in this task.")


def maximum(a: pd.DataFrame | float, b: pd.DataFrame | float) -> pd.DataFrame:
    """Compute MAX(A, B) elementwise with DataFrame/scalar broadcasting."""

    # Broadcast scalar inputs into a DataFrame so output shape is explicit.
    aa = a
    bb = b
    if not isinstance(aa, pd.DataFrame) and isinstance(bb, pd.DataFrame):
        aa = pd.DataFrame(float(aa), index=bb.index, columns=bb.columns)
    if not isinstance(bb, pd.DataFrame) and isinstance(aa, pd.DataFrame):
        bb = pd.DataFrame(float(bb), index=aa.index, columns=aa.columns)

    # Align frames and compute elementwise maximum with nan-aware semantics.
    if isinstance(aa, pd.DataFrame) and isinstance(bb, pd.DataFrame):
        aa2, bb2 = aa.align(bb, join="inner", axis=1)
        aa2, bb2 = aa2.align(bb2, join="inner", axis=0)
        out = np.fmax(aa2.to_numpy(dtype=np.float64, copy=False), bb2.to_numpy(dtype=np.float64, copy=False))
        return pd.DataFrame(out, index=aa2.index, columns=aa2.columns)

    # Fail fast when neither input is a DataFrame, as specs should return frames.
    raise TypeError("MAX expects at least one DataFrame input in this task.")


def signed_sqrt(x: pd.DataFrame) -> pd.DataFrame:
    """Compute SIGNED_SQRT(X) = sign(X) * sqrt(abs(X))."""

    # Normalize input to float64 so sign/sqrt behave consistently.
    y = as_float_frame(x)

    # Apply the sign-preserving square-root transform.
    arr = y.to_numpy(dtype=np.float64, copy=False)
    out = np.sign(arr) * np.sqrt(np.abs(arr))
    return pd.DataFrame(out, index=y.index, columns=y.columns)


def signed_power(x: pd.DataFrame, power: float | pd.DataFrame) -> pd.DataFrame:
    """Compute SIGNED_POWER(X, p) = sign(X) * abs(X)**p (p can be scalar or frame)."""

    # Normalize input to float64 so sign/abs/power are stable and NaN-preserving.
    y = as_float_frame(x)

    # Align a DataFrame exponent to X so elementwise semantics are explicit.
    p = power
    if isinstance(p, pd.DataFrame):
        pp = as_float_frame(p)
        yy, pp = y.align(pp, join="inner", axis=1)
        yy, pp = yy.align(pp, join="inner", axis=0)
        arr = yy.to_numpy(dtype=np.float64, copy=False)
        par = pp.to_numpy(dtype=np.float64, copy=False)
        out = np.sign(arr) * (np.abs(arr) ** par)
        return pd.DataFrame(out, index=yy.index, columns=yy.columns)

    # Apply scalar exponent sign-preserving power transform as used by several factor expressions.
    p_float = float(p)
    arr = y.to_numpy(dtype=np.float64, copy=False)
    out = np.sign(arr) * (np.abs(arr) ** p_float)
    return pd.DataFrame(out, index=y.index, columns=y.columns)
