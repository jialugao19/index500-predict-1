from __future__ import annotations

import numpy as np
import pandas as pd


def as_float_frame(x: pd.DataFrame) -> pd.DataFrame:
    """Normalize a matrix into float64 with inf treated as NaN."""

    # Convert to float64 so downstream numpy ops are stable and consistent.
    y = x.astype(np.float64)

    # Replace inf with NaN to keep strict-window logic correct.
    return y.replace([np.inf, -np.inf], np.nan)


def align_on_common_index_and_columns(frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Align multiple (time x instrument) frames on the common index and columns."""

    # Compute the common time index across all inputs.
    common_index = None
    for v in frames.values():
        common_index = v.index if common_index is None else common_index.intersection(v.index)

    # Compute the common instrument columns across all inputs.
    common_cols = None
    for v in frames.values():
        common_cols = v.columns if common_cols is None else common_cols.intersection(v.columns)

    # Reindex each frame to the common grid to guarantee exact broadcasting semantics.
    return {k: v.reindex(index=common_index, columns=common_cols) for k, v in frames.items()}

