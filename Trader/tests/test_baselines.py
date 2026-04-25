from __future__ import annotations

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from trader.data.models import MarketBar
from trader.evaluation.baselines import evaluate_baselines
from trader.execution.engine import BacktestResult
from trader.execution.fills import Trade
from trader.strategies.spec import ExecConfig


NEW_YORK = ZoneInfo("America/New_York")


def _bars() -> tuple[MarketBar, ...]:
    bars = []
    sessions = (datetime(2026, 1, 5, 9, 30, tzinfo=NEW_YORK), datetime(2026, 1, 6, 9, 30, tzinfo=NEW_YORK))
    prices = (100.0, 101.0, 102.0, 103.0, 102.0, 104.0)
    for session_start, session_prices in zip(sessions, (prices[:3], prices[3:])):
        for index, price in enumerate(session_prices):
            timestamp = (session_start + timedelta(minutes=index)).astimezone(timezone.utc)
            bars.append(
                MarketBar(
                    timestamp_ms=int(timestamp.timestamp() * 1000),
                    timestamp_utc=timestamp.isoformat(),
                    open=price,
                    high=price + 0.25,
                    low=price - 0.25,
                    close=price,
                    volume=1_000.0,
                )
            )
    return tuple(bars)


def _single_session_bars(count: int) -> tuple[MarketBar, ...]:
    return _priced_single_session_bars(tuple(100.0 + index for index in range(count)))


def _priced_single_session_bars(prices: tuple[float, ...]) -> tuple[MarketBar, ...]:
    bars = []
    session_start = datetime(2026, 1, 5, 9, 30, tzinfo=NEW_YORK)
    for index, price in enumerate(prices):
        timestamp = (session_start + timedelta(minutes=index)).astimezone(timezone.utc)
        bars.append(
            MarketBar(
                timestamp_ms=int(timestamp.timestamp() * 1000),
                timestamp_utc=timestamp.isoformat(),
                open=price,
                high=price + 0.25,
                low=price - 0.25,
                close=price,
                volume=1_000.0,
            )
        )
    return tuple(bars)


def _reference_result(bars: tuple[MarketBar, ...]) -> BacktestResult:
    return BacktestResult(
        bars=bars,
        trades=(
            Trade(
                bars[0].timestamp_utc,
                bars[1].timestamp_utc,
                100.0,
                101.0,
                10,
                2,
                10.0,
                1.0,
                "signal_flip",
            ),
            Trade(
                bars[3].timestamp_utc,
                bars[3].timestamp_utc,
                103.0,
                103.0,
                10,
                1,
                0.0,
                0.0,
                "signal_flip",
            ),
        ),
        equity_curve=(100_000.0, 100_010.0, 100_010.0, 100_010.0, 100_010.0, 100_010.0),
        initial_cash=100_000.0,
        final_cash=100_010.0,
    )


def _same_session_reference(bars: tuple[MarketBar, ...]) -> BacktestResult:
    return BacktestResult(
        bars=bars,
        trades=(
            Trade(bars[0].timestamp_utc, bars[1].timestamp_utc, 100.0, 101.0, 10, 2, 10.0, 1.0, "signal_flip"),
            Trade(bars[2].timestamp_utc, bars[3].timestamp_utc, 102.0, 103.0, 10, 2, 10.0, 1.0, "signal_flip"),
        ),
        equity_curve=(100_000.0, 100_010.0, 100_010.0, 100_020.0),
        initial_cash=100_000.0,
        final_cash=100_020.0,
    )


def test_intraday_baselines_are_present_and_session_bounded() -> None:
    bars = _bars()
    reference = _reference_result(bars)
    baselines = evaluate_baselines(
        bars,
        ExecConfig(initial_cash=100_000.0),
        strategy_trades=reference.trades,
        seed_material="test-seed",
    )

    assert set(baselines) == {
        "always_flat",
        "buy_and_hold",
        "regular_session_open_to_close_long",
        "session_long_flat_at_close",
        "randomized_entry_same_exposure",
    }
    assert baselines["regular_session_open_to_close_long"]["trade_count"] == 2.0
    assert baselines["regular_session_open_to_close_long"]["exposure_pct"] == 100.0
    assert baselines["session_long_flat_at_close"]["trade_count"] == 2.0


def test_randomized_entry_baseline_is_reproducible_and_matches_reference_exposure() -> None:
    bars = _bars()
    reference = _reference_result(bars)

    first = evaluate_baselines(
        bars,
        ExecConfig(initial_cash=100_000.0),
        strategy_trades=reference.trades,
        seed_material="test-seed",
    )["randomized_entry_same_exposure"]
    second = evaluate_baselines(
        bars,
        ExecConfig(initial_cash=100_000.0),
        strategy_trades=reference.trades,
        seed_material="test-seed",
    )["randomized_entry_same_exposure"]

    assert first == second
    assert first["trade_count"] == 2.0
    assert first["exposure_pct"] == 50.0


def test_randomized_entry_baseline_preserves_same_session_trade_count_and_exposure() -> None:
    bars = _single_session_bars(4)
    reference = _same_session_reference(bars)

    baseline = evaluate_baselines(
        bars,
        ExecConfig(initial_cash=100_000.0),
        strategy_trades=reference.trades,
        seed_material="collision-seed",
    )["randomized_entry_same_exposure"]

    assert baseline["trade_count"] == 2.0
    assert baseline["exposure_pct"] == 100.0


def test_randomized_entry_baseline_uses_execution_costs() -> None:
    bars = _bars()
    reference = _reference_result(bars)
    free = evaluate_baselines(
        bars,
        ExecConfig(initial_cash=100_000.0, commission_per_order=0.0, slippage_bps=0.0),
        strategy_trades=reference.trades,
        seed_material="cost-seed",
    )["randomized_entry_same_exposure"]
    costed = evaluate_baselines(
        bars,
        ExecConfig(initial_cash=100_000.0, commission_per_order=5.0, slippage_bps=100.0),
        strategy_trades=reference.trades,
        seed_material="cost-seed",
    )["randomized_entry_same_exposure"]

    assert costed["final_cash"] < free["final_cash"]


def test_randomized_entry_baseline_retries_unaffordable_windows() -> None:
    bars = _priced_single_session_bars((100.0, 1_000_000.0, 1_000_001.0))
    reference = BacktestResult(
        bars=bars,
        trades=(
            Trade(bars[0].timestamp_utc, bars[0].timestamp_utc, 100.0, 100.0, 10, 1, 0.0, 0.0, "signal_flip"),
        ),
        equity_curve=(100_000.0, 100_000.0, 100_000.0),
        initial_cash=100_000.0,
        final_cash=100_000.0,
    )

    baseline = evaluate_baselines(
        bars,
        ExecConfig(initial_cash=100_000.0),
        strategy_trades=reference.trades,
        seed_material="unaffordable-seed",
    )["randomized_entry_same_exposure"]

    assert baseline["trade_count"] == 1.0
    assert baseline["exposure_pct"] == pytest.approx(100.0 / 3.0)
