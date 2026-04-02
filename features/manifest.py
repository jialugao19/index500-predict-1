from __future__ import annotations

import hashlib
from dataclasses import asdict

from features.registry import FactorRegistry


def build_factor_set_manifest(registry: FactorRegistry, factor_set_name: str) -> dict:
    """Build a YAML-friendly manifest dict for a factor set."""

    # Resolve the factor set from the registry.
    factor_set = registry.factor_sets[str(factor_set_name)]

    # Serialize a stable representation to compute a content hash.
    payload = asdict(factor_set)
    text = str(payload).encode("utf-8")
    digest = hashlib.sha256(text).hexdigest()

    # Emit a compact manifest structure for reporting and provenance.
    return {
        "factor_set": payload,
        "factor_set_sha256": str(digest),
        "factors": [asdict(registry.factor_specs[name]) for name in factor_set.factors],
    }

