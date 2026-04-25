from __future__ import annotations

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from trader.data.models import MarketBar
from trader.evaluation.metrics import calculate_metrics
from trader.evaluation.runner import EvaluationRunner
from trader.execution.engine import BacktestResult, run_long_only_engine
from trader.execution.fills import Trade, enter_long, exit_long
from trader.strategies.registry import REGISTRY
from trader.strategies.spec import ExecConfig, SignalSpec, StrategySpec


NEW_YORK = ZoneInfo("America/New_York")


def _bar(index: int, price: float = 100.0, volume: float = 1_000.0) -> MarketBar:
    timestamp = (datetime(2026, 1, 5, 9, 30, tzinfo=NEW_YORK) + timedelta(minutes=index)).astimezone(timezone.utc)
    return MarketBar(
        timestamp_ms=int(timestamp.timestamp() * 1000),
        timestamp_utc=timestamp.isoformat(),
        open=price,
        high=price + 0.25,
        low=price - 0.25,
        close=price,
        volume=volume,
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


def test_fill_costs_apply_per_share_commission_spread_and_notional_cap() -> None:
    config = ExecConfig(
        initial_cash=1_000.0,
        commission_per_order=1.0,
        commission_per_share=0.10,
        slippage_bps=0.0,
        spread_bps=10.0,
        max_position_notional=500.0,
    )

    cash, position = enter_long(1_000.0, _bar(0, 100.0), config)
    assert position is not None
    assert position.shares == 4
    assert position.entry_price == pytest.approx(100.05)
    assert position.entry_commission == pytest.approx(1.40)

    position.bars_held = 1
    cash, trade = exit_long(cash, position, _bar(1, 101.0), config, "test", fill_at_close=True)

    assert cash == pytest.approx(1_000.798)
    assert trade.cost_cash == pytest.approx(3.202)


def test_stop_loss_exits_at_threshold_when_bar_low_crosses_stop() -> None:
    config = ExecConfig(initial_cash=100_000.0, slippage_bps=0.0, stop_loss_bps=100.0)
    bars = (
        _ohlc_bar(0, open_price=100.0, high=100.0, low=100.0, close=100.0),
        _ohlc_bar(1, open_price=100.0, high=101.0, low=99.5, close=100.5),
        _ohlc_bar(2, open_price=101.0, high=101.0, low=98.5, close=100.0),
    )

    result = run_long_only_engine(bars, (True, True, True), config)

    assert len(result.trades) == 1
    assert result.trades[0].exit_reason == "stop_loss"
    assert result.trades[0].exit_price == pytest.approx(99.0)


def test_stop_loss_gap_through_exits_at_bar_open() -> None:
    config = ExecConfig(initial_cash=100_000.0, slippage_bps=0.0, stop_loss_bps=100.0)
    bars = (
        _ohlc_bar(0, open_price=100.0, high=100.0, low=100.0, close=100.0),
        _ohlc_bar(1, open_price=100.0, high=101.0, low=99.5, close=100.5),
        _ohlc_bar(2, open_price=98.0, high=99.0, low=97.0, close=98.5),
    )

    result = run_long_only_engine(bars, (True, True, True), config)

    assert len(result.trades) == 1
    assert result.trades[0].exit_reason == "stop_loss"
    assert result.trades[0].exit_price == pytest.approx(98.0)


def test_stop_loss_preempts_pending_signal_exit_on_gap_through() -> None:
    config = ExecConfig(initial_cash=100_000.0, slippage_bps=0.0, stop_loss_bps=100.0)
    bars = (
        _ohlc_bar(0, open_price=100.0, high=100.0, low=100.0, close=100.0),
        _ohlc_bar(1, open_price=100.0, high=101.0, low=99.5, close=100.5),
        _ohlc_bar(2, open_price=98.0, high=99.0, low=97.0, close=98.5),
        _ohlc_bar(3, open_price=102.0, high=102.0, low=102.0, close=102.0),
    )

    result = run_long_only_engine(bars, (True, False, False, False), config)

    assert len(result.trades) == 1
    assert result.trades[0].exit_reason == "stop_loss"
    assert result.trades[0].exit_price == pytest.approx(98.0)


def test_cost_drag_metrics_use_trade_cost_cash() -> None:
    result = BacktestResult(
        bars=(_bar(0),),
        trades=(Trade("a", "b", 100.0, 100.0, 1, 1, 0.0, 0.0, "test", cost_cash=12.5),),
        equity_curve=(100_000.0,),
        initial_cash=100_000.0,
        final_cash=100_000.0,
    )

    metrics = calculate_metrics(result)

    assert metrics["cost_drag_cash"] == 12.5
    assert metrics["cost_drag_pct"] == pytest.approx(0.0125)


def test_cost_scenario_metrics_are_reported() -> None:
    config = ExecConfig(initial_cash=100_000.0, commission_per_order=1.0, slippage_bps=1.0, spread_bps=2.0)
    spec = REGISTRY.validate_spec(
        StrategySpec(
            name="cost_scenario",
            signal=SignalSpec("ema_cross", {"fast_length": 2, "slow_length": 3, "signal_buffer_bps": 0.0}),
            exec_config=config,
        )
    )
    bars = (_bar(0, 100.0), _bar(1, 101.0), _bar(2, 102.0))
    regime = (False, True, True)
    result = run_long_only_engine(bars, regime, config)
    metrics = calculate_metrics(result)

    scenario_metrics = EvaluationRunner.__new__(EvaluationRunner)._cost_scenario_metrics(
        spec,
        bars,
        regime,
        1.0,
        metrics,
    )

    assert scenario_metrics["cost_drag_return_pct"] > 0.0
    assert scenario_metrics["cost_scenario_slippage_plus_2bps_delta_pct"] < 0.0
    assert scenario_metrics["cost_scenario_spread_plus_2bps_delta_pct"] < 0.0


def test_registry_rejects_invalid_cost_config_values() -> None:
    with pytest.raises(ValueError, match="commission_per_share"):
        REGISTRY.validate_spec(StrategySpec(name="bad_commission", exec_config=ExecConfig(commission_per_share=-0.01)))
    with pytest.raises(ValueError, match="spread_bps"):
        REGISTRY.validate_spec(StrategySpec(name="bad_spread", exec_config=ExecConfig(spread_bps=-1.0)))
    with pytest.raises(ValueError, match="max_position_notional"):
        REGISTRY.validate_spec(StrategySpec(name="bad_notional", exec_config=ExecConfig(max_position_notional=0.0)))
    with pytest.raises(ValueError, match="stop_loss_bps"):
        REGISTRY.validate_spec(StrategySpec(name="bad_stop_loss", exec_config=ExecConfig(stop_loss_bps=0.0)))
