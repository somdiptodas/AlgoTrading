from __future__ import annotations


def normalize_params(params: dict[str, object]) -> dict[str, str]:
    merged = {"session": "regular", **params}
    session = str(merged["session"])
    if session != "regular":
        raise ValueError("Only regular-session filtering is supported in v1")
    return {"session": session}
