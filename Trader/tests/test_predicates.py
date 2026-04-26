from __future__ import annotations

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from trader.data.models import MarketBar
from trader.strategies.decisions import SignalVote
from trader.strategies.predicates import PREDICATES, PredicateHandler, PredicateRegistry


NEW_YORK = ZoneInfo("America/New_York")


def _bar(minute: int, close: float, *, volume: float = 1_000.0, vwap: float | None = 100.0) -> MarketBar:
    timestamp = (datetime(2026, 1, 5, 9, 30, tzinfo=NEW_YORK) + timedelta(minutes=minute)).astimezone(timezone.utc)
    return MarketBar(
        timestamp_ms=int(timestamp.timestamp() * 1000),
        timestamp_utc=timestamp.isoformat(),
        open=close,
        high=close + 0.25,
        low=close - 0.25,
        close=close,
        volume=volume,
        vwap=vwap,
    )


def test_atomic_predicate_registry_is_separate_from_strategy_registry() -> None:
    assert PREDICATES.names() == ()


def test_atomic_predicate_registry_normalizes_and_generates_votes() -> None:
    def normalize(params: dict[str, object]) -> dict[str, object]:
        return {"threshold": float(params.get("threshold", 100.0))}

    def generate(
        history_bars: tuple[MarketBar, ...],
        test_bars: tuple[MarketBar, ...],
        params: dict[str, object],
    ) -> list[SignalVote]:
        threshold = float(params["threshold"])
        return [
            SignalVote("example", bar.close > threshold, f"close {bar.close:.2f} > {threshold:.2f}")
            for bar in test_bars
        ]

    registry = PredicateRegistry(
        {
            "example": PredicateHandler(
                normalize_params=normalize,
                required_history=lambda params: 0,
                generate_votes=generate,
            )
        }
    )

    assert registry.names() == ("example",)
    assert registry.validate_params("example", {"threshold": "100.50"}) == {"threshold": 100.5}
    assert registry.required_history("example", {"threshold": 100.5}) == 0
    assert registry.generate_votes("example", (), (_bar(0, 100.0), _bar(1, 101.0)), {"threshold": 100.5}) == [
        SignalVote("example", False, "close 100.00 > 100.50"),
        SignalVote("example", True, "close 101.00 > 100.50"),
    ]


def test_atomic_predicate_registry_rejects_unknown_or_invalid_handlers() -> None:
    registry = PredicateRegistry()
    with pytest.raises(ValueError, match="Unknown atomic predicate"):
        registry.validate_params("missing", {})

    bad_length = PredicateRegistry(
        {
            "bad": PredicateHandler(
                normalize_params=lambda params: {},
                required_history=lambda params: 0,
                generate_votes=lambda history, test, params: [],
            )
        }
    )
    with pytest.raises(ValueError, match="produced 0 votes for 1 test bars"):
        bad_length.generate_votes("bad", (), (_bar(0, 100.0),), {})
