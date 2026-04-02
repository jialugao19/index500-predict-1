from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class FactorSpec:
    """Hold a factor spec for minute-level feature engineering."""

    name_en: str
    name_zh: str
    formula: str
    inputs: list[str]
    tags: list[str]


@dataclass(frozen=True)
class FactorSetSpec:
    """Hold a factor set specification for a specific scope."""

    name: str
    scope: str
    factors: list[str]
    xs_clip_rank_factors: list[str]
    invalid_after_first_invalid_factors: list[str]


def _load_yaml(path: Path) -> dict:
    """Load a YAML file into a python dict."""

    # Read YAML text from disk to keep encoding explicit and stable.
    raw = path.read_text(encoding="utf-8")

    # Parse YAML into a dict so dataclass construction is explicit downstream.
    return yaml.safe_load(raw)


def load_factor_spec(path: Path) -> FactorSpec:
    """Load a factor spec YAML file into a FactorSpec."""

    # Parse the YAML file.
    spec = _load_yaml(path)

    # Construct FactorSpec so downstream access stays explicit.
    return FactorSpec(**spec)


def load_factor_set_spec(path: Path) -> FactorSetSpec:
    """Load a factor set YAML file into a FactorSetSpec."""

    # Parse the YAML file.
    spec = _load_yaml(path)

    # Construct FactorSetSpec so builder wiring stays explicit.
    return FactorSetSpec(**spec)

