from __future__ import annotations


DEFAULT_PARAMS: dict[str, float] = {"fraction": 0.25}


def normalize_params(params: dict[str, object]) -> dict[str, float]:
    merged = {**DEFAULT_PARAMS, **params}
    fraction = float(merged["fraction"])
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fixed_fraction.fraction must be in (0.0, 1.0]")
    return {"fraction": fraction}


def compute_fraction(params: dict[str, object]) -> float:
    """Return the cash fraction [0, 1] to deploy on each entry."""
    return float(params.get("fraction", DEFAULT_PARAMS["fraction"]))
