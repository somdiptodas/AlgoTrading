from __future__ import annotations

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from trader.data.models import MarketBar
from trader.execution.engine import run_long_only_decision_engine
from trader.execution.fills import enter_long
from trader.strategies.decisions import RuleDecision, SignalVote, TradeDecision
from trader.strategies.spec import ExecConfig


NEW_YORK = ZoneInfo("America/New_York")


def _bar(index: int, price: float = 100.0) -> MarketBar:
    timestamp = (datetime(2026, 1, 5, 9, 30, tzinfo=NEW_YORK) + timedelta(minutes=index)).astimezone(timezone.utc)
    return MarketBar(
        timestamp_ms=int(timestamp.timestamp() * 1000),
        timestamp_utc=timestamp.isoformat(),
        open=price,
        high=price + 0.25,
        low=price - 0.25,
        close=price,
        volume=1_000.0,
    )


def _ohlc_bar(index: int, *, open_price: float, high: float, low: float, close: float) -> MarketBar:
    timestamp = (datetime(2026, 1, 5, 9, 30, tzinfo=NEW_YORK) + timedelta(minutes=index)).astimezone(timezone.utc)
    return MarketBar(
        timestamp_ms=int(timestamp.timestamp() * 1000),
        timestamp_utc=timestamp.isoformat(),
        open=open_price,
        high=high,
        low=low,
        close=close,
        volume=1_000.0,
    )


def _decision(entry: bool, exit: bool) -> TradeDecision:
    return TradeDecision(
        entry=RuleDecision(entry, "entry_passed" if entry else "entry_failed"),
        exit=RuleDecision(exit, "exit_passed" if exit else "exit_failed"),
    )


def test_decision_engine_requires_one_decision_per_bar() -> None:
    with pytest.raises(ValueError, match="bars and decisions_by_bar must have identical lengths"):
        run_long_only_decision_engine(
            (_bar(0), _bar(1)),
            (_decision(True, False),),
            ExecConfig(initial_cash=100_000.0),
        )


def test_decision_engine_enters_from_entry_rule_while_flat() -> None:
    bars = tuple(_bar(index, 100.0) for index in range(4))
    decisions = (
        _decision(True, False),
        _decision(False, False),
        _decision(False, False),
        _decision(False, False),
    )

    result = run_long_only_decision_engine(bars, decisions, ExecConfig(initial_cash=100_000.0, slippage_bps=0.0))

    assert len(result.trades) == 1
    assert result.trades[0].entry_timestamp_utc == bars[1].timestamp_utc
    assert result.trades[0].entry_reason == "entry_passed"


def test_decision_engine_exits_from_exit_rule_while_long() -> None:
    bars = tuple(_bar(index, 100.0) for index in range(4))
    decisions = (
        _decision(True, False),
        _decision(False, True),
        _decision(False, False),
        _decision(False, False),
    )

    result = run_long_only_decision_engine(bars, decisions, ExecConfig(initial_cash=100_000.0, slippage_bps=0.0))

    assert len(result.trades) == 1
    assert result.trades[0].entry_timestamp_utc == bars[1].timestamp_utc
    assert result.trades[0].exit_timestamp_utc == bars[2].timestamp_utc
    assert result.trades[0].exit_reason == "exit_passed"


def test_decision_engine_preserves_stop_loss_exit() -> None:
    bars = (
        _ohlc_bar(0, open_price=100.0, high=100.0, low=100.0, close=100.0),
        _ohlc_bar(1, open_price=100.0, high=101.0, low=99.5, close=100.5),
        _ohlc_bar(2, open_price=98.0, high=99.0, low=97.0, close=98.5),
        _ohlc_bar(3, open_price=102.0, high=102.0, low=102.0, close=102.0),
    )
    decisions = (
        _decision(True, False),
        _decision(False, True),
        _decision(False, False),
        _decision(False, False),
    )

    result = run_long_only_decision_engine(
        bars,
        decisions,
        ExecConfig(initial_cash=100_000.0, slippage_bps=0.0, stop_loss_bps=100.0),
    )

    assert len(result.trades) == 1
    assert result.trades[0].exit_reason == "stop_loss"
    assert result.trades[0].exit_price == pytest.approx(98.0)


def test_decision_engine_preserves_session_close_exit() -> None:
    bars = tuple(_bar(index, 100.0) for index in range(4))
    decisions = (
        _decision(True, False),
        _decision(False, False),
        _decision(False, False),
        _decision(False, False),
    )

    result = run_long_only_decision_engine(bars, decisions, ExecConfig(initial_cash=100_000.0, slippage_bps=0.0))

    assert len(result.trades) == 1
    assert result.trades[0].exit_timestamp_utc == bars[-1].timestamp_utc
    assert result.trades[0].exit_reason == "session_close"


def test_decision_engine_preserves_final_bar_exit() -> None:
    bars = tuple(_bar(index, 100.0) for index in range(4))
    decisions = (
        _decision(True, False),
        _decision(False, False),
        _decision(False, False),
        _decision(False, False),
    )

    result = run_long_only_decision_engine(
        bars,
        decisions,
        ExecConfig(initial_cash=100_000.0, slippage_bps=0.0, flat_at_close=False),
    )

    assert len(result.trades) == 1
    assert result.trades[0].exit_timestamp_utc == bars[-1].timestamp_utc
    assert result.trades[0].exit_reason == "final_bar"


def test_decision_engine_preserves_one_open_position_behavior() -> None:
    bars = tuple(_bar(index, 100.0) for index in range(5))
    decisions = tuple(_decision(True, False) for _ in bars)

    result = run_long_only_decision_engine(bars, decisions, ExecConfig(initial_cash=100_000.0, slippage_bps=0.0))

    assert len(result.trades) == 1
    assert result.trades[0].entry_timestamp_utc == bars[1].timestamp_utc
    assert result.trades[0].exit_timestamp_utc == bars[-1].timestamp_utc


def test_enter_long_stores_entry_decision_on_position() -> None:
    entry_rule = RuleDecision(True, "entry_vote_passed", (SignalVote("entry_a", True, "ok"),))

    _, position = enter_long(
        100_000.0,
        _bar(0, 100.0),
        ExecConfig(initial_cash=100_000.0, slippage_bps=0.0),
        reason=entry_rule.reason,
        entry_rule=entry_rule,
    )

    assert position is not None
    assert position.entry_rule == entry_rule
