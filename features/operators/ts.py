from __future__ import annotations

import numpy as np
import pandas as pd

from features.operators.utils import as_float_frame


def _sliding_window_view_time(arr: np.ndarray, window: int) -> np.ndarray:
    """Create a sliding window view along the time axis for (T, N) arrays."""

    # Normalize window into an int and validate it is positive.
    win = int(window)
    if win <= 0:
        raise ValueError(f"window must be positive, got {window}")

    # Build a zero-copy window view and move the window axis to the middle: (T-win+1, win, N).
    windows = np.lib.stride_tricks.sliding_window_view(arr, window_shape=win, axis=0)
    return np.moveaxis(windows, -1, 1)


def ts_returns(close: pd.DataFrame, window: int, mode: int) -> pd.DataFrame:
    """Compute TS_RETURNS(CLOSE, window, MODE=1) as close/close.shift(window) - 1."""

    # Enforce the supported return mode so wiring mistakes fail fast.
    if int(mode) != 1:
        raise ValueError(f"ts_returns expects mode=1, got {mode}")

    # Normalize inputs so inf never contaminates arithmetic.
    y = as_float_frame(close)

    # Compute simple returns over the requested lookback window.
    return y.div(y.shift(int(window))).sub(1.0)


def ts_delay(x: pd.DataFrame, periods: int) -> pd.DataFrame:
    """Compute TS_DELAY(X, periods) along the time axis for each instrument."""

    # Shift values down by the given number of periods so t uses t-periods.
    return as_float_frame(x).shift(int(periods))


def ts_delta(x: pd.DataFrame, periods: int) -> pd.DataFrame:
    """Compute DELTA(X, periods) = X - TS_DELAY(X, periods)."""

    # Normalize inputs so subtraction behaves consistently on float64.
    y = as_float_frame(x)

    # Compute the difference against the shifted series.
    return y.sub(ts_delay(y, periods=int(periods)))


def ts_mean(x: pd.DataFrame, window: int) -> pd.DataFrame:
    """Compute TS_MEAN(X, window) with a strict full-finite window requirement."""

    # Normalize inputs and compute rolling mean with a full window.
    y = as_float_frame(x)
    m = y.rolling(int(window)).mean()

    # Enforce a strict finite window so any NaN inside the window yields NaN output.
    ok = y.rolling(int(window)).count().eq(int(window))
    return m.where(ok)


def ts_std(x: pd.DataFrame, window: int) -> pd.DataFrame:
    """Compute TS_STD(X, window) (ddof=1) with a strict full-finite window requirement."""

    # Normalize inputs and compute rolling std with a full window.
    y = as_float_frame(x)
    s = y.rolling(int(window)).std(ddof=1)

    # Enforce a strict finite window so any NaN inside the window yields NaN output.
    ok = y.rolling(int(window)).count().eq(int(window))
    return s.where(ok)


def ts_max(x: pd.DataFrame, window: int) -> pd.DataFrame:
    """Compute TS_MAX(X, window) with a strict full-finite window requirement."""

    # Normalize inputs and compute rolling max with a full window.
    y = as_float_frame(x)
    mx = y.rolling(int(window)).max()

    # Enforce a strict finite window so any NaN inside the window yields NaN output.
    ok = y.rolling(int(window)).count().eq(int(window))
    return mx.where(ok)


def ts_min(x: pd.DataFrame, window: int) -> pd.DataFrame:
    """Compute TS_MIN(X, window) with a strict full-finite window requirement."""

    # Normalize inputs and compute rolling min with a full window.
    y = as_float_frame(x)
    mn = y.rolling(int(window)).min()

    # Enforce a strict finite window so any NaN inside the window yields NaN output.
    ok = y.rolling(int(window)).count().eq(int(window))
    return mn.where(ok)


def ts_sum(x: pd.DataFrame, window: int) -> pd.DataFrame:
    """Compute TS_SUM(X, window) with a strict full-finite window requirement."""

    # Normalize inputs and compute rolling sum with a full window.
    y = as_float_frame(x)
    s = y.rolling(int(window)).sum()

    # Enforce a strict finite window so any NaN inside the window yields NaN output.
    ok = y.rolling(int(window)).count().eq(int(window))
    return s.where(ok)


def ts_product(x: pd.DataFrame, window: int) -> pd.DataFrame:
    """Compute TS_PRODUCT(X, window) with a strict full-finite window requirement."""

    # Normalize inputs and convert into a contiguous float64 array.
    y = as_float_frame(x)
    arr = y.to_numpy(dtype=np.float64, copy=False)

    # Build a sliding window view and compute strict finite masks.
    win = int(window)
    windows = _sliding_window_view_time(arr, window=win)
    ok = np.isfinite(windows).all(axis=1)

    # Compute window products and write them back into a full-length output array.
    prod = np.prod(windows, axis=1)
    out = np.full_like(arr, np.nan)
    out[win - 1 :, :] = np.where(ok, prod, np.nan)
    return pd.DataFrame(out, index=y.index, columns=y.columns)


def ts_covariance(x: pd.DataFrame, y: pd.DataFrame, window: int) -> pd.DataFrame:
    """Compute TS_COVARIANCE(X, Y, window) as strict-window rolling covariance (ddof=0)."""

    # Align inputs and normalize to float so pairwise semantics are explicit.
    xx = as_float_frame(x)
    yy = as_float_frame(y)
    xx, yy = xx.align(yy, join="inner", axis=1)
    xx, yy = xx.align(yy, join="inner", axis=0)

    # Build a strict pairwise-valid mask so any NaN in X or Y invalidates the whole window.
    ok = xx.notna() & yy.notna()
    xx = xx.where(ok)
    yy = yy.where(ok)

    # Compute strict rolling moments needed for covariance.
    win = int(window)
    mx = xx.rolling(win, min_periods=win).mean()
    my = yy.rolling(win, min_periods=win).mean()
    mxy = xx.mul(yy).rolling(win, min_periods=win).mean()

    # Convert raw moments into covariance and preserve NaNs by construction.
    return mxy.sub(mx.mul(my))


def ts_corr(x: pd.DataFrame, y: pd.DataFrame, window: int) -> pd.DataFrame:
    """Compute TS_CORR(X, Y, window) as strict-window rolling Pearson correlation."""

    # Compute strict covariance and strict rolling standard deviations.
    cov = ts_covariance(x, y, window=int(window))
    sx = ts_std(x, window=int(window))
    sy = ts_std(y, window=int(window))

    # Convert into correlation with an explicit non-zero std requirement.
    denom = sx.mul(sy)
    return cov.div(denom.where(denom.ne(0.0)))


def ts_rank(x: pd.DataFrame, window: int) -> pd.DataFrame:
    """Compute TS_RANK(X, window) as the percentile rank of the latest value in each window."""

    # Normalize inputs and convert into a contiguous float64 array.
    y = as_float_frame(x)
    arr = y.to_numpy(dtype=np.float64, copy=False)

    # Build a sliding window view and compute strict finite masks.
    win = int(window)
    windows = _sliding_window_view_time(arr, window=win)
    ok = np.isfinite(windows).all(axis=1)

    # Compute the percentile rank of the last element with average-tie handling.
    last = windows[:, -1, :]
    less = (windows < last[:, None, :]).sum(axis=1, dtype=np.int64)
    equal = (windows == last[:, None, :]).sum(axis=1, dtype=np.int64)
    rank_pos = less.astype(np.float64) + (equal.astype(np.float64) + 1.0) / 2.0
    pct = rank_pos / float(win)

    # Write the rolling result back into the full-length output array.
    out = np.full_like(arr, np.nan)
    out[win - 1 :, :] = np.where(ok, pct, np.nan)
    return pd.DataFrame(out, index=y.index, columns=y.columns)


def ts_argmax(x: pd.DataFrame, window: int) -> pd.DataFrame:
    """Compute TS_ARGMAX(X, window) as the argmax position inside each rolling window."""

    # Normalize inputs and convert into a contiguous float64 array.
    y = as_float_frame(x)
    arr = y.to_numpy(dtype=np.float64, copy=False)

    # Build a sliding window view and compute strict finite masks.
    win = int(window)
    windows = _sliding_window_view_time(arr, window=win)
    ok = np.isfinite(windows).all(axis=1)

    # Compute argmax positions along the window axis and write into the full output.
    arg = np.argmax(windows, axis=1).astype(np.float64)
    out = np.full_like(arr, np.nan)
    out[win - 1 :, :] = np.where(ok, arg, np.nan)
    return pd.DataFrame(out, index=y.index, columns=y.columns)


def ts_argmin(x: pd.DataFrame, window: int) -> pd.DataFrame:
    """Compute TS_ARGMIN(X, window) as the argmin position inside each rolling window."""

    # Normalize inputs and convert into a contiguous float64 array.
    y = as_float_frame(x)
    arr = y.to_numpy(dtype=np.float64, copy=False)

    # Build a sliding window view and compute strict finite masks.
    win = int(window)
    windows = _sliding_window_view_time(arr, window=win)
    ok = np.isfinite(windows).all(axis=1)

    # Compute argmin positions along the window axis and write into the full output.
    arg = np.argmin(windows, axis=1).astype(np.float64)
    out = np.full_like(arr, np.nan)
    out[win - 1 :, :] = np.where(ok, arg, np.nan)
    return pd.DataFrame(out, index=y.index, columns=y.columns)


def ts_decay_linear(x: pd.DataFrame, window: int) -> pd.DataFrame:
    """Compute DECAY_LINEAR(X, window) as a linearly-weighted moving average."""

    # Normalize inputs and convert into a contiguous float64 array.
    y = as_float_frame(x)
    arr = y.to_numpy(dtype=np.float64, copy=False)

    # Build a sliding window view and compute strict finite masks.
    win = int(window)
    windows = _sliding_window_view_time(arr, window=win)
    ok = np.isfinite(windows).all(axis=1)

    # Build normalized linear weights and apply them via a vectorized weighted sum.
    w = np.arange(win, 0, -1, dtype=np.float64)
    w = w / float(w.sum())
    weighted = (windows * w[None, :, None]).sum(axis=1)

    # Write the rolling result back into the full-length output array.
    out = np.full_like(arr, np.nan)
    out[win - 1 :, :] = np.where(ok, weighted, np.nan)
    return pd.DataFrame(out, index=y.index, columns=y.columns)
