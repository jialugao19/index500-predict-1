from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from features.spec import FactorSetSpec, FactorSpec, load_factor_set_spec, load_factor_spec


@dataclass(frozen=True)
class FactorRegistry:
    """Hold all factor specs and factor sets loaded from `features/specs/`."""

    factor_specs: dict[str, FactorSpec]
    factor_sets: dict[str, FactorSetSpec]


def _dedup_in_order(items: list[str]) -> list[str]:
    """Deduplicate a string list while preserving order."""

    # Collect unseen items in first-seen order.
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if str(item) in seen:
            continue
        seen.add(str(item))
        out.append(str(item))
    return out


def _build_union_factor_set(
    factor_sets: dict[str, FactorSetSpec],
    name: str,
    scope: str,
    source_names: list[str],
    xs_clip_rank_factors: list[str],
) -> FactorSetSpec:
    """Build a synthetic factor set by unioning multiple source sets."""

    # Collect source specs in the requested order.
    source_specs = [factor_sets[str(source_name)] for source_name in source_names]
    assert all([str(spec.scope) == str(scope) for spec in source_specs])

    # Union factor names and invalid-after names in first-seen order.
    factors = _dedup_in_order([factor for spec in source_specs for factor in spec.factors])
    invalid_after = _dedup_in_order(
        [factor for spec in source_specs for factor in spec.invalid_after_first_invalid_factors]
    )
    xs_clip_rank = _dedup_in_order([factor for spec in source_specs for factor in spec.xs_clip_rank_factors])
    xs_clip_rank = _dedup_in_order(xs_clip_rank + [str(factor) for factor in xs_clip_rank_factors])

    # Emit a synthetic factor set with explicit preprocessing choices.
    return FactorSetSpec(
        name=str(name),
        scope=str(scope),
        factors=factors,
        xs_clip_rank_factors=[str(factor) for factor in xs_clip_rank],
        invalid_after_first_invalid_factors=invalid_after,
    )


def load_registry(specs_root: str) -> FactorRegistry:
    """Load all factor specs and factor set specs from a specs root directory."""

    # Locate the factor and factor-set directories under specs_root.
    root = Path(specs_root)
    factors_dir = root / "factors"
    sets_dir = root / "sets"

    # Load factor specs from YAML files into a name-keyed dict.
    factor_specs: dict[str, FactorSpec] = {}
    for path in sorted(factors_dir.glob("*.yaml")):
        spec = load_factor_spec(path=path)
        factor_specs[str(spec.name_en)] = spec

    # Load factor sets from YAML files into a name-keyed dict.
    factor_sets: dict[str, FactorSetSpec] = {}
    for path in sorted(sets_dir.glob("*.yaml")):
        spec = load_factor_set_spec(path=path)
        factor_sets[str(spec.name)] = spec

    # Require at least one spec so configuration errors fail fast.
    if len(factor_specs) == 0:
        raise FileNotFoundError(os.path.join(str(factors_dir), "*.yaml"))
    if len(factor_sets) == 0:
        raise FileNotFoundError(os.path.join(str(sets_dir), "*.yaml"))

    # Build future-proof synthetic union sets for the active pipeline.
    if ("stock_default" in factor_sets) and ("stock_alpha101" in factor_sets):
        factor_sets["stock_all"] = _build_union_factor_set(
            factor_sets=factor_sets,
            name="stock_all",
            scope="stock",
            source_names=["stock_default", "stock_alpha101"],
            xs_clip_rank_factors=[],
        )
    if "etf_default" in factor_sets:
        factor_sets["etf_all"] = _build_union_factor_set(
            factor_sets=factor_sets,
            name="etf_all",
            scope="etf",
            source_names=["etf_default"],
            xs_clip_rank_factors=[],
        )

    return FactorRegistry(factor_specs=factor_specs, factor_sets=factor_sets)


@lru_cache(maxsize=None)
def cached_load_registry(specs_root: str) -> FactorRegistry:
    """Load registry once per specs_root for fast repeated per-day builds."""

    # Delegate to the plain loader and cache results by specs_root.
    return load_registry(specs_root=specs_root)
