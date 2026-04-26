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
    assert "rsi_below" in PREDICATES.names()
    assert "ema_cross" not in PREDICATES.names()
    assert "breakout" not in PREDICATES.names()


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


def test_rsi_below_predicate_votes_when_rsi_is_under_threshold() -> None:
    history = (_bar(0, 100.0), _bar(1, 101.0))
    test = (_bar(2, 90.0), _bar(3, 89.0))

    assert PREDICATES.validate_params("rsi_below", {"length": 2, "threshold": "30"}) == {
        "length": 2,
        "threshold": 30.0,
    }
    assert PREDICATES.required_history("rsi_below", {"length": 2, "threshold": 30.0}) == 3
    votes = PREDICATES.generate_votes("rsi_below", history, test, {"length": 2, "threshold": 30.0})

    assert [vote.passed for vote in votes] == [True, True]
    assert votes[0].detail == "RSI 8.33 < 30.00"


def test_rsi_above_predicate_votes_when_rsi_is_over_threshold() -> None:
    history = (_bar(0, 100.0), _bar(1, 99.0))
    test = (_bar(2, 110.0), _bar(3, 111.0))

    assert PREDICATES.validate_params("rsi_above", {"length": 2, "threshold": "70"}) == {
        "length": 2,
        "threshold": 70.0,
    }
    votes = PREDICATES.generate_votes("rsi_above", history, test, {"length": 2, "threshold": 70.0})

    assert [vote.passed for vote in votes] == [True, True]
    assert votes[0].detail == "RSI 91.67 > 70.00"


def test_ema_trend_up_predicate_votes_when_fast_ema_is_above_slow_ema() -> None:
    history = (_bar(0, 100.0), _bar(1, 101.0), _bar(2, 102.0))
    test = (_bar(3, 110.0),)

    assert PREDICATES.validate_params("ema_trend_up", {"fast": 2, "slow": 3, "buffer_bps": "0"}) == {
        "fast": 2,
        "slow": 3,
        "buffer_bps": 0.0,
    }
    assert PREDICATES.required_history("ema_trend_up", {"fast": 2, "slow": 3}) == 3
    votes = PREDICATES.generate_votes("ema_trend_up", history, test, {"fast": 2, "slow": 3})

    assert votes[0].passed is True
    assert votes[0].detail.startswith("fast EMA ")
    assert " > slow EMA " in votes[0].detail


def test_ema_trend_down_predicate_votes_when_fast_ema_is_below_slow_ema() -> None:
    history = (_bar(0, 110.0), _bar(1, 109.0), _bar(2, 108.0))
    test = (_bar(3, 100.0),)

    votes = PREDICATES.generate_votes("ema_trend_down", history, test, {"fast": 2, "slow": 3})

    assert votes[0].passed is True
    assert votes[0].detail.startswith("fast EMA ")
    assert " < slow EMA " in votes[0].detail


def test_breakout_up_predicate_votes_when_close_exceeds_prior_high() -> None:
    history = (_bar(0, 100.0), _bar(1, 101.0), _bar(2, 102.0))
    test = (_bar(3, 103.0),)

    assert PREDICATES.validate_params("breakout_up", {"window": 3, "buffer_bps": "0"}) == {
        "window": 3,
        "buffer_bps": 0.0,
    }
    assert PREDICATES.required_history("breakout_up", {"window": 3}) == 3
    votes = PREDICATES.generate_votes("breakout_up", history, test, {"window": 3})

    assert votes == [SignalVote("breakout_up", True, "close 103.00 > prior high 102.25")]


def test_breakout_failed_predicate_votes_when_close_breaks_prior_low() -> None:
    history = (_bar(0, 103.0), _bar(1, 102.0), _bar(2, 101.0))
    test = (_bar(3, 100.0),)

    votes = PREDICATES.generate_votes("breakout_failed", history, test, {"window": 3})

    assert votes == [SignalVote("breakout_failed", True, "close 100.00 < prior low 100.75")]


def test_vwap_distance_predicate_votes_when_price_is_far_enough_from_vwap() -> None:
    test = (_bar(0, 99.50, vwap=100.0),)

    assert PREDICATES.validate_params("vwap_distance", {"side": "below", "min_bps": "25"}) == {
        "side": "below",
        "min_bps": 25.0,
        "max_bps": 100_000.0,
    }
    assert PREDICATES.required_history("vwap_distance", {"side": "below"}) == 0
    votes = PREDICATES.generate_votes("vwap_distance", (), test, {"side": "below", "min_bps": 25.0})

    assert votes == [
        SignalVote("vwap_distance", True, "VWAP distance 50.00 bps below in [25.00, 100000.00]")
    ]


def test_vwap_reclaimed_predicate_votes_when_price_is_back_above_vwap() -> None:
    test = (_bar(0, 100.10, vwap=100.0),)

    assert PREDICATES.validate_params("vwap_reclaimed", {"min_bps": "5"}) == {"min_bps": 5.0}
    votes = PREDICATES.generate_votes("vwap_reclaimed", (), test, {"min_bps": 5.0})

    assert votes == [SignalVote("vwap_reclaimed", True, "close 100.10 >= VWAP reclaim 100.05")]
