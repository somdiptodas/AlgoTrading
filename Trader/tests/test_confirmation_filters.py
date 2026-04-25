from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from trader.data.models import MarketBar
from trader.strategies.registry import REGISTRY
from trader.strategies.spec import FilterSpec, SignalSpec, StrategySpec


NEW_YORK = ZoneInfo("America/New_York")


def _bar(index: int, close: float, *, vwap: float | None = 100.0) -> MarketBar:
    timestamp = (datetime(2026, 1, 5, 9, 30, tzinfo=NEW_YORK) + timedelta(minutes=index)).astimezone(timezone.utc)
    return MarketBar(
        timestamp_ms=int(timestamp.timestamp() * 1000),
        timestamp_utc=timestamp.isoformat(),
        open=close,
        high=close + 0.25,
        low=close - 0.25,
        close=close,
        volume=1_000.0,
        vwap=vwap,
    )


def _vwap_filter(side: str, min_deviation_bps: float = 25.0) -> FilterSpec:
    return FilterSpec("vwap_distance", {"side": side, "min_deviation_bps": min_deviation_bps})


def test_vwap_distance_filter_allows_and_blocks_below_side() -> None:
    spec = StrategySpec(name="below_vwap", filters=(_vwap_filter("below", 25.0),))
    bars = (_bar(0, 99.70), _bar(1, 99.90))

    regime = REGISTRY.generate_regime(spec, tuple(), bars)

    assert regime == [False, False]
    mask = REGISTRY.filter_handlers["vwap_distance"]["generate_mask"](
        tuple(),
        bars,
        REGISTRY.validate_spec(spec).filters[0].params,
    )
    assert mask == [True, False]


def test_vwap_distance_filter_allows_and_blocks_above_side() -> None:
    spec = StrategySpec(name="above_vwap", filters=(_vwap_filter("above", 25.0),))
    bars = (_bar(0, 100.30), _bar(1, 100.10))
    params = REGISTRY.validate_spec(spec).filters[0].params

    assert REGISTRY.filter_handlers["vwap_distance"]["generate_mask"](tuple(), bars, params) == [True, False]


def test_vwap_distance_filter_missing_vwap_blocks() -> None:
    spec = StrategySpec(name="missing_vwap", filters=(_vwap_filter("below", 25.0),))
    params = REGISTRY.validate_spec(spec).filters[0].params

    assert REGISTRY.filter_handlers["vwap_distance"]["generate_mask"](tuple(), (_bar(0, 99.0, vwap=None),), params) == [
        False
    ]


@pytest.mark.parametrize(
    ("params", "message"),
    (
        ({"side": "near"}, "side"),
        ({"min_deviation_bps": -1.0}, "min_deviation_bps"),
        ({"min_deviation_bps": 20.0, "max_deviation_bps": 10.0}, "max_deviation_bps"),
        ({"min_deviation_bps": float("nan")}, "min_deviation_bps must be finite"),
    ),
)
def test_vwap_distance_filter_rejects_invalid_params(params: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        REGISTRY.validate_spec(StrategySpec(name="bad_vwap_distance", filters=(FilterSpec("vwap_distance", params),)))


def test_vwap_distance_filter_does_not_peek_forward() -> None:
    spec = StrategySpec(name="no_peek", filters=(_vwap_filter("below", 25.0),))
    params = REGISTRY.validate_spec(spec).filters[0].params
    bars = (_bar(0, 99.70), _bar(1, 99.70), _bar(2, 99.70))
    altered = (bars[0], replace(bars[1], close=120.0), replace(bars[2], close=120.0))

    assert REGISTRY.filter_handlers["vwap_distance"]["generate_mask"](tuple(), bars, params)[0] == REGISTRY.filter_handlers[
        "vwap_distance"
    ]["generate_mask"](tuple(), altered, params)[0]


def test_rsi_reversion_can_be_confirmed_by_vwap_distance() -> None:
    base = StrategySpec(
        name="rsi_base",
        signal=SignalSpec(
            "rsi_reversion",
            {"rsi_length": 2, "oversold_threshold": 30.0, "overbought_threshold": 70.0},
        ),
    )
    confirmed = StrategySpec(
        name="rsi_confirmed",
        signal=base.signal,
        filters=(_vwap_filter("below", 25.0),),
    )
    blocked = StrategySpec(
        name="rsi_blocked",
        signal=base.signal,
        filters=(_vwap_filter("below", 25.0),),
    )
    bars = (_bar(0, 100.0), _bar(1, 99.0), _bar(2, 98.0), _bar(3, 97.0))
    blocked_bars = tuple(replace(bar, vwap=bar.close) for bar in bars)

    assert REGISTRY.generate_regime(base, tuple(), bars)[2] is True
    assert REGISTRY.generate_regime(confirmed, tuple(), bars)[2] is True
    assert REGISTRY.generate_regime(blocked, tuple(), blocked_bars)[2] is False
