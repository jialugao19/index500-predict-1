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

    return FactorRegistry(factor_specs=factor_specs, factor_sets=factor_sets)


@lru_cache(maxsize=None)
def cached_load_registry(specs_root: str) -> FactorRegistry:
    """Load registry once per specs_root for fast repeated per-day builds."""

    # Delegate to the plain loader and cache results by specs_root.
    return load_registry(specs_root=specs_root)
