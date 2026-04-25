from __future__ import annotations

from trader.evaluation.robustness import _monthly_strategy_pnl_breakdown, assess_robustness
from trader.execution.engine import BacktestResult
from trader.execution.fills import Trade
from trader.strategies.spec import SignalSpec, StrategySpec


class _NoNeighborRegistry:
    def neighbors(self, spec: StrategySpec) -> tuple[StrategySpec, ...]:
        return tuple()


def _trade(exit_timestamp_utc: str, pnl_cash: float) -> Trade:
    return Trade(
        entry_timestamp_utc="2026-01-01T14:30:00+00:00",
        exit_timestamp_utc=exit_timestamp_utc,
        entry_price=100.0,
        exit_price=101.0,
        shares=1,
        bars_held=1,
        pnl_cash=pnl_cash,
        pnl_pct=pnl_cash,
        exit_reason="signal_flip",
    )


def _backtest(trades: tuple[Trade, ...]) -> BacktestResult:
    final_cash = 100_000.0 + sum(trade.pnl_cash for trade in trades)
    return BacktestResult(
        bars=tuple(),
        trades=trades,
        equity_curve=(100_000.0, final_cash),
        initial_cash=100_000.0,
        final_cash=final_cash,
    )


def _checks_for(trades: tuple[Trade, ...]) -> dict[str, float | bool]:
    result = assess_robustness(
        spec=StrategySpec(
            name="robustness_test",
            signal=SignalSpec("ema_cross", {"fast_length": 20, "slow_length": 80, "signal_buffer_bps": 0.0}),
        ),
        aggregate_metrics={
            "return_pct": 1.0,
            "sharpe_like": 1.0,
            "max_drawdown_pct": 1.0,
        },
        fold_metrics=({"return_pct": 1.0}, {"return_pct": 1.0}),
        fold_backtests=(_backtest(trades),),
        registry=_NoNeighborRegistry(),
        neighbor_metric_fn=lambda spec: {},
    )
    return result.checks


def test_monthly_strategy_pnl_uses_trade_exit_local_month() -> None:
    backtest = _backtest(
        (
            _trade("2026-03-01T04:30:00+00:00", 10.0),
            _trade("2026-03-02T15:00:00+00:00", 20.0),
        )
    )

    assert _monthly_strategy_pnl_breakdown((backtest,)) == {
        "2026-02": 10.0,
        "2026-03": 20.0,
    }


def test_regime_pass_allows_inclusive_eighty_percent_positive_pnl_concentration() -> None:
    checks = _checks_for(
        (
            _trade("2026-01-15T20:00:00+00:00", 80.0),
            _trade("2026-02-15T20:00:00+00:00", 20.0),
        )
    )

    assert checks["positive_monthly_pnl_concentration_pct"] == 80.0
    assert checks["regime_pass"] is True


def test_regime_pass_fails_when_positive_pnl_is_too_concentrated() -> None:
    checks = _checks_for(
        (
            _trade("2026-01-15T20:00:00+00:00", 81.0),
            _trade("2026-02-15T20:00:00+00:00", 19.0),
        )
    )

    assert checks["positive_monthly_pnl_concentration_pct"] == 81.0
    assert checks["regime_pass"] is False


def test_regime_pass_fails_when_loss_pnl_is_too_concentrated() -> None:
    checks = _checks_for(
        (
            _trade("2026-01-15T20:00:00+00:00", -100.0),
            _trade("2026-02-15T20:00:00+00:00", 10.0),
            _trade("2026-03-15T20:00:00+00:00", 10.0),
        )
    )

    assert checks["positive_monthly_pnl_concentration_pct"] == 50.0
    assert checks["loss_monthly_pnl_concentration_pct"] == 100.0
    assert checks["regime_pass"] is False


def test_regime_pass_fails_without_positive_monthly_pnl() -> None:
    checks = _checks_for(
        (
            _trade("2026-01-15T20:00:00+00:00", 0.0),
            _trade("2026-02-15T20:00:00+00:00", -1.0),
        )
    )

    assert checks["positive_monthly_pnl_present"] is False
    assert checks["regime_pass"] is False
