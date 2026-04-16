from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FrameZScoreStats:
    """Hold per-column z-score statistics for a feature frame."""

    columns: list[str]
    mean: pd.Series
    std: pd.Series


@dataclass(frozen=True)
class SeriesZScoreStats:
    """Hold z-score statistics for a scalar target series."""

    name: str
    mean: float
    std: float


def fit_frame_zscore_stats(frame: pd.DataFrame, columns: list[str]) -> FrameZScoreStats:
    """Fit per-column z-score statistics from a dataframe."""

    # Slice the training columns as float data.
    data = frame.loc[:, columns].astype(float)

    # Estimate mean and standard deviation from the training slice.
    mean = data.mean(axis=0, skipna=True)
    std = data.std(axis=0, skipna=True, ddof=0)

    # Require finite training statistics for every column.
    assert bool(np.isfinite(mean.to_numpy(dtype=float)).all())
    assert bool(np.isfinite(std.to_numpy(dtype=float)).all())
    assert bool((std.to_numpy(dtype=float) > 0.0).all())

    # Return immutable stats for downstream transforms.
    return FrameZScoreStats(columns=list(columns), mean=mean, std=std)


def transform_frame_zscore(frame: pd.DataFrame, stats: FrameZScoreStats) -> pd.DataFrame:
    """Apply fitted z-score statistics to a dataframe."""

    # Copy the input frame to keep the transform side-effect free.
    out = frame.copy()

    # Cast standardized columns to float to avoid incompatible-dtype assignments.
    out = out.astype({name: float for name in stats.columns})

    # Standardize the requested columns with aligned pandas broadcasting.
    out.loc[:, stats.columns] = (out.loc[:, stats.columns] - stats.mean).div(stats.std, axis=1)
    return out


def fit_series_zscore_stats(series: pd.Series, name: str) -> SeriesZScoreStats:
    """Fit z-score statistics from a target series."""

    # Convert the training target into a float vector.
    values = series.astype(float).to_numpy(dtype=float)
    values = values[np.isfinite(values)]

    # Estimate scalar mean and standard deviation.
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=0))

    # Require a finite non-zero training scale.
    assert np.isfinite(mean)
    assert np.isfinite(std)
    assert float(std) > 0.0

    # Return immutable scalar stats.
    return SeriesZScoreStats(name=str(name), mean=mean, std=std)


def transform_series_zscore(series: pd.Series, stats: SeriesZScoreStats) -> pd.Series:
    """Apply fitted z-score statistics to a target series."""

    # Standardize the series into model space.
    return (series.astype(float) - float(stats.mean)) / float(stats.std)


def inverse_series_zscore(values: np.ndarray, stats: SeriesZScoreStats) -> np.ndarray:
    """Map standardized model outputs back to raw target units."""

    # Undo the training-target standardization.
    return values.astype(float) * float(stats.std) + float(stats.mean)
