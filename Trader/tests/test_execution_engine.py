from __future__ import annotations

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from trader.data.models import MarketBar
from trader.execution.engine import run_long_only_decision_engine
from trader.strategies.decisions import RuleDecision, TradeDecision
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
