from __future__ import annotations

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from trader.data.models import MarketBar
from trader.evaluation.metrics import calculate_metrics
from trader.evaluation.splits import build_walk_forward_folds
from trader.execution.engine import BacktestResult
from trader.execution.fills import Trade
from trader.strategies.registry import REGISTRY
from trader.strategies.spec import SignalSpec, StrategySpec


NEW_YORK = ZoneInfo("America/New_York")


def _market_bars(count: int) -> tuple[MarketBar, ...]:
    start = datetime(2026, 1, 5, 9, 30, tzinfo=NEW_YORK)
    bars = []
    for index in range(count):
        timestamp = start + timedelta(minutes=index)
        timestamp_utc = timestamp.astimezone(timezone.utc)
        price = 100.0 + (index * 0.1)
        bars.append(
            MarketBar(
                timestamp_ms=int(timestamp_utc.timestamp() * 1000),
                timestamp_utc=timestamp_utc.isoformat(),
                open=price,
                high=price + 0.25,
                low=price - 0.25,
                close=price,
                volume=1_000.0,
            )
        )
    return tuple(bars)


def test_walk_forward_splits_are_ordered_and_non_overlapping() -> None:
    bars = _market_bars(240)
    split_plan_id, folds = build_walk_forward_folds(bars, num_folds=3, min_train_bars=120, embargo_bars=2)
    assert split_plan_id
    assert len(folds) == 3
    for fold in folds:
        assert fold.train_end_idx < fold.test_start_idx
        assert fold.test_start_idx - fold.train_end_idx - 1 >= 2


def test_metrics_on_known_trade_series() -> None:
    bars = _market_bars(3)
    result = BacktestResult(
        bars=bars,
        trades=(
            Trade("a", "b", 100.0, 101.0, 10, 2, 10.0, 1.0, "signal_flip"),
            Trade("b", "c", 101.0, 100.5, 10, 1, -5.0, -0.5, "signal_flip"),
        ),
        equity_curve=(100_000.0, 100_010.0, 100_005.0),
        initial_cash=100_000.0,
        final_cash=100_005.0,
    )
    metrics = calculate_metrics(result)
    assert metrics["trade_count"] == 2.0
    assert metrics["win_rate_pct"] == 50.0
    assert metrics["max_drawdown_pct"] >= 0.0


def test_strategy_hash_is_stable() -> None:
    spec = StrategySpec(
        name="ema_test",
        signal=SignalSpec("ema_cross", {"fast_length": 20, "slow_length": 80, "signal_buffer_bps": 0.0}),
    )
    validated = REGISTRY.validate_spec(spec)
    assert validated.spec_hash() == REGISTRY.validate_spec(spec).spec_hash()
    renamed = StrategySpec(
        name="ema_test_renamed",
        signal=SignalSpec("ema_cross", {"fast_length": 20, "slow_length": 80, "signal_buffer_bps": 0.0}),
    )
    assert validated.spec_hash() == REGISTRY.validate_spec(renamed).spec_hash()


def test_invalid_breakout_spec_rejected() -> None:
    spec = StrategySpec(
        name="bad_breakout",
        signal=SignalSpec("breakout", {"entry_window": 10, "exit_window": 20, "buffer_bps": 0.0}),
    )
    with pytest.raises(ValueError):
        REGISTRY.validate_spec(spec)
