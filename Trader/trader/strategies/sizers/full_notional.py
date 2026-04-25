from __future__ import annotations


def normalize_params(params: dict[str, object]) -> dict[str, object]:
    if params:
        raise ValueError("full_notional does not accept params")
    return {}


def compute_fraction(params: dict[str, object]) -> float:
    """Full notional always deploys 100% of available cash."""
    return 1.0
