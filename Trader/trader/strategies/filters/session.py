from __future__ import annotations

from typing import Sequence

from trader.data.models import MarketBar


def normalize_params(params: dict[str, object]) -> dict[str, str]:
    merged = {"session": "regular", **params}
    session = str(merged["session"])
    if session != "regular":
        raise ValueError("Only regular-session filtering is supported in v1")
    return {"session": session}


def generate_mask(
    history_bars: Sequence[MarketBar],
    test_bars: Sequence[MarketBar],
    params: dict[str, object],
) -> list[bool]:
    return [bar.is_regular_session for bar in test_bars]
