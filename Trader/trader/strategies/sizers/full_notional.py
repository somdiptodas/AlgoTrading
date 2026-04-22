from __future__ import annotations


def normalize_params(params: dict[str, object]) -> dict[str, object]:
    if params:
        raise ValueError("full_notional does not accept params")
    return {}
