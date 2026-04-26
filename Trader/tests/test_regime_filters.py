from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from trader.data.models import MarketBar
from trader.execution.engine import run_long_only_engine
from trader.research.planner import DeterministicPlanner
from trader.strategies.filters.regime import generate_reporting_regime_labels
from trader.strategies.registry import REGISTRY
from trader.strategies.spec import FilterSpec, SignalSpec, StrategySpec


NEW_YORK = ZoneInfo("America/New_York")


def _bar(day: int, minute: int, close: float, *, volume: float = 1_000.0, vwap: float | None = 100.0) -> MarketBar:
    timestamp = (
        datetime(2026, 1, 5, 9, 30, tzinfo=NEW_YORK)
        + timedelta(days=day, minutes=minute)
    ).astimezone(timezone.utc)
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


def _vwap_spec(filters: tuple[FilterSpec, ...] = ()) -> StrategySpec:
    return StrategySpec(
        name="vwap_filtered",
        signal=SignalSpec(
            "vwap_deviation",
            {"entry_deviation_bps": 25.0, "exit_deviation_bps": 0.0, "max_hold_bars": 10},
        ),
        filters=filters,
    )


def test_regime_filters_validate_and_contribute_required_history() -> None:
    spec = REGISTRY.validate_spec(
        _vwap_spec(
            (
                FilterSpec(
                    "intraday_volatility",
                    {"lookback_bars": 3, "percentile_window": 7, "min_percentile": 50.0},
                ),
            )
        )
    )

    assert spec.filters[0].params == {
        "lookback_bars": 3,
        "percentile_window": 7,
        "min_percentile": 50.0,
        "max_percentile": 100.0,
    }
    assert REGISTRY.required_history(spec) == 10


@pytest.mark.parametrize(
    ("filter_spec", "message"),
    (
        (FilterSpec("unknown", {}), "Unknown filter handler"),
        (FilterSpec("intraday_volatility", {"lookback_bars": 1}), "lookback_bars"),
        (FilterSpec("intraday_volatility", {"min_percentile": float("nan")}), "percentile bounds"),
        (FilterSpec("prior_day_range", {"max_range_bps": -1.0}), "max_range_bps"),
        (FilterSpec("relative_volume", {"lookback_bars": 0}), "lookback_bars"),
        (FilterSpec("relative_volume", {"min_ratio": 2.0, "max_ratio": 1.0}), "max_ratio"),
        (FilterSpec("day_type", {"mode": "sideways"}), "mode"),
        (FilterSpec("day_type", {"min_efficiency": 0.2, "max_efficiency": 0.4}), "efficiency"),
    ),
)
def test_regime_filter_validation_rejects_bad_params(filter_spec: FilterSpec, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        REGISTRY.validate_spec(_vwap_spec((filter_spec,)))


def test_regime_filters_stack_as_logical_and() -> None:
    bars = tuple(_bar(0, minute, 99.50) for minute in range(5))
    allow_filter = FilterSpec("relative_volume", {"lookback_bars": 1, "min_ratio": 0.0})
    block_filter = FilterSpec("day_type", {"mode": "trend", "min_bars": 30})

    assert REGISTRY.generate_regime(_vwap_spec((allow_filter,)), tuple(), bars)[1] is True
    assert REGISTRY.generate_regime(_vwap_spec((allow_filter, block_filter)), tuple(), bars) == [False] * 5


def test_relative_volume_filter_changes_execution() -> None:
    bars = tuple(_bar(0, minute, 99.50, volume=1_000.0) for minute in range(5))
    unfiltered_regime = REGISTRY.generate_regime(_vwap_spec(), tuple(), bars)
    blocked_regime = REGISTRY.generate_regime(
        _vwap_spec((FilterSpec("relative_volume", {"lookback_bars": 1, "min_ratio": 2.0}),)),
        tuple(),
        bars,
    )

    assert len(run_long_only_engine(bars, unfiltered_regime, _vwap_spec().exec_config).trades) == 1
    assert run_long_only_engine(bars, blocked_regime, _vwap_spec().exec_config).trades == tuple()


def test_intraday_volatility_filter_does_not_peek_forward() -> None:
    history = tuple(_bar(0, index, 100.0 + index) for index in range(6))
    test = tuple(_bar(0, 6 + index, 106.0 + index) for index in range(3))
    altered = (test[0], replace(test[1], close=150.0), replace(test[2], close=50.0))
    spec = _vwap_spec(
        (
            FilterSpec(
                "intraday_volatility",
                {"lookback_bars": 2, "percentile_window": 3, "min_percentile": 0.0, "max_percentile": 100.0},
            ),
        )
    )

    assert REGISTRY.generate_regime(spec, history, test)[0] == REGISTRY.generate_regime(spec, history, altered)[0]


def test_prior_day_range_filter_uses_only_completed_prior_sessions() -> None:
    history = tuple(_bar(0, minute, 100.0 + minute) for minute in range(4))
    test = tuple(_bar(1, minute, 99.50) for minute in range(3))
    altered = tuple(replace(bar, high=200.0, low=50.0) for bar in test)
    spec = _vwap_spec((FilterSpec("prior_day_range", {"min_range_bps": 10.0}),))

    assert REGISTRY.generate_regime(spec, history, test) == REGISTRY.generate_regime(spec, history, altered)


def test_relative_volume_filter_does_not_peek_forward() -> None:
    test = tuple(_bar(0, index, 99.50, volume=1_000.0) for index in range(4))
    altered = (test[0], test[1], replace(test[2], volume=10_000.0), replace(test[3], volume=10_000.0))
    spec = _vwap_spec((FilterSpec("relative_volume", {"lookback_bars": 1, "min_ratio": 0.0}),))

    assert REGISTRY.generate_regime(spec, tuple(), test)[1] == REGISTRY.generate_regime(spec, tuple(), altered)[1]


def test_day_type_filter_does_not_peek_forward() -> None:
    test = tuple(_bar(0, index, 100.0 + index) for index in range(5))
    altered = (test[0], test[1], replace(test[2], close=50.0), replace(test[3], close=50.0), replace(test[4], close=50.0))
    spec = _vwap_spec((FilterSpec("day_type", {"mode": "trend", "min_bars": 2, "trend_bps": 0.0}),))

    assert REGISTRY.generate_regime(spec, tuple(), test)[1] == REGISTRY.generate_regime(spec, tuple(), altered)[1]


def test_reporting_regime_labels_do_not_peek_forward() -> None:
    history = tuple(_bar(0, index, 100.0 + index) for index in range(4))
    test = tuple(_bar(0, 4 + index, 104.0 + index) for index in range(4))
    altered = (
        test[0],
        test[1],
        replace(test[2], close=50.0, low=49.0),
        replace(test[3], close=150.0, high=151.0),
    )

    labels = generate_reporting_regime_labels(
        history,
        test,
        volatility_lookback_bars=2,
        volatility_percentile_window=3,
    )
    altered_labels = generate_reporting_regime_labels(
        history,
        altered,
        volatility_lookback_bars=2,
        volatility_percentile_window=3,
    )

    assert labels[:2] == altered_labels[:2]


def test_planner_emits_optional_regime_filter_variants() -> None:
    planned = DeterministicPlanner(REGISTRY).plan(batch_size=40, allowed_signal_families=("vwap_deviation",))

    assert any(item.spec.filters == tuple() for item in planned)
    assert any(item.spec.filters for item in planned)
    assert all(REGISTRY.validate_spec(item.spec) for item in planned)


def test_planner_default_budget_includes_each_regime_filter_family() -> None:
    planned = DeterministicPlanner(REGISTRY).plan(
        batch_size=72,
        allowed_signal_families=("ema_cross", "breakout", "rsi_reversion", "vwap_deviation"),
    )

    assert {item.spec.filters[0].name for item in planned if item.spec.filters} >= {
        "intraday_volatility",
        "prior_day_range",
        "relative_volume",
        "day_type",
    }
