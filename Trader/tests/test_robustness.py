from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from trader.data.models import MarketBar
from trader.data.view import DataSlice
from trader.evaluation.metrics import annualized_sharpe_for_backtests
from trader.evaluation.robustness import RobustnessResult, _monthly_strategy_pnl_breakdown, assess_robustness
from trader.evaluation.runner import EvaluationPreview, EvaluationRunner, FoldResult
from trader.evaluation.splits import Fold
from trader.execution.engine import BacktestResult
from trader.execution.fills import Trade
from trader.strategies.registry import REGISTRY
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


def _bars(count: int) -> tuple[MarketBar, ...]:
    start = datetime(2026, 1, 1, 14, 30, tzinfo=timezone.utc)
    bars = []
    for index in range(count):
        timestamp = start + timedelta(minutes=index)
        bars.append(
            MarketBar(
                timestamp_ms=int(timestamp.timestamp() * 1000),
                timestamp_utc=timestamp.isoformat(),
                open=100.0,
                high=101.0,
                low=99.0,
                close=100.0,
                volume=1_000.0,
            )
        )
    return tuple(bars)


def _backtest(trades: tuple[Trade, ...], bar_count: int = 1) -> BacktestResult:
    final_cash = 100_000.0 + sum(trade.pnl_cash for trade in trades)
    return BacktestResult(
        bars=_bars(bar_count),
        trades=trades,
        equity_curve=(100_000.0, final_cash),
        initial_cash=100_000.0,
        final_cash=final_cash,
    )


def _fold(fold_id: str) -> Fold:
    return Fold(
        fold_id=fold_id,
        train_start_idx=0,
        train_end_idx=0,
        test_start_idx=0,
        test_end_idx=0,
        embargo_bars=0,
        train_start_utc="2026-01-01T14:30:00+00:00",
        train_end_utc="2026-01-01T14:30:00+00:00",
        test_start_utc="2026-01-01T14:31:00+00:00",
        test_end_utc="2026-01-01T14:31:00+00:00",
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


def test_evaluate_preview_aggregates_neighbor_metrics_across_all_folds(monkeypatch) -> None:
    runner = EvaluationRunner.__new__(EvaluationRunner)
    runner.registry = REGISTRY
    runner._fold_result_cache = {}
    folds = (_fold("fold_1"), _fold("fold_2"), _fold("fold_3"))
    fold_returns = {"fold_1": 1.0, "fold_2": 3.0, "fold_3": 5.0}
    fold_bar_counts = {"fold_1": 10, "fold_2": 10, "fold_3": 30}
    neighbor_fold_ids: list[str] = []
    captured_neighbor_metrics: dict[str, float] = {}

    spec = StrategySpec(
        name="ema_test",
        signal=SignalSpec("ema_cross", {"fast_length": 20, "slow_length": 80, "signal_buffer_bps": 0.0}),
    )
    preview = EvaluationPreview(
        spec=REGISTRY.validate_spec(spec),
        data_slice=DataSlice(
            bars=tuple(),
            snapshot_id="snapshot",
            first_timestamp_utc=None,
            last_timestamp_utc=None,
        ),
        split_plan_id="split-plan",
        folds=folds,
        required_history=80,
        cost_model_id="cost-model",
        evaluation_key="evaluation-key",
    )

    def fake_evaluate_fold(fold_spec: StrategySpec, fold: Fold, bars) -> FoldResult:
        if fold_spec.name.startswith("ema_cross_"):
            neighbor_fold_ids.append(fold.fold_id)
            return_pct = fold_returns[fold.fold_id]
        else:
            return_pct = 2.0
        return FoldResult(
            fold_id=fold.fold_id,
            train_start_utc=fold.train_start_utc,
            train_end_utc=fold.train_end_utc,
            test_start_utc=fold.test_start_utc,
            test_end_utc=fold.test_end_utc,
            metrics={
                "return_pct": return_pct,
                "sharpe_like": return_pct / 10.0,
                "max_drawdown_pct": 1.0,
            },
            baseline_metrics={},
            baseline_deltas={},
            warnings=tuple(),
            backtest=_backtest(tuple(), bar_count=fold_bar_counts[fold.fold_id]),
        )

    def fake_assess_robustness(*, neighbor_metric_fn, registry, spec, **kwargs) -> RobustnessResult:
        neighbor_spec = registry.neighbors(spec)[0]
        captured_neighbor_metrics.update(neighbor_metric_fn(neighbor_spec))
        return RobustnessResult(
            checks={
                "fold_consistency_pass": True,
                "regime_pass": True,
                "neighborhood_pass": True,
                "drawdown_pass": True,
            },
            passed=True,
        )

    monkeypatch.setattr(runner, "_evaluate_fold", fake_evaluate_fold)
    monkeypatch.setattr("trader.evaluation.runner.assess_robustness", fake_assess_robustness)

    runner.evaluate_preview(preview)

    assert neighbor_fold_ids == ["fold_1", "fold_2", "fold_3"]
    assert captured_neighbor_metrics["return_pct"] == 3.8
    assert captured_neighbor_metrics["sharpe_like"] == 0.38


def test_evaluate_preview_annualized_sharpe_uses_combined_fold_returns(monkeypatch) -> None:
    runner = EvaluationRunner.__new__(EvaluationRunner)
    folds = (_fold("fold_1"), _fold("fold_2"))
    starting_equity = 100_000.0
    fold_backtests = {
        "fold_1": BacktestResult(
            bars=_bars(2),
            trades=tuple(),
            equity_curve=(starting_equity * 1.001, starting_equity * 1.001 * 0.9995),
            initial_cash=starting_equity,
            final_cash=starting_equity * 1.001 * 0.9995,
        ),
        "fold_2": BacktestResult(
            bars=_bars(1),
            trades=tuple(),
            equity_curve=(starting_equity * 1.0015,),
            initial_cash=starting_equity,
            final_cash=starting_equity * 1.0015,
        ),
    }
    preview = EvaluationPreview(
        spec=REGISTRY.validate_spec(
            StrategySpec(
                name="ema_test",
                signal=SignalSpec("ema_cross", {"fast_length": 20, "slow_length": 80, "signal_buffer_bps": 0.0}),
            )
        ),
        data_slice=DataSlice(
            bars=tuple(),
            snapshot_id="snapshot",
            first_timestamp_utc=None,
            last_timestamp_utc=None,
        ),
        split_plan_id="split-plan",
        folds=folds,
        required_history=80,
        cost_model_id="cost-model",
        evaluation_key="evaluation-key",
    )

    def fake_evaluate_fold(fold_spec: StrategySpec, fold: Fold, bars) -> FoldResult:
        backtest = fold_backtests[fold.fold_id]
        return FoldResult(
            fold_id=fold.fold_id,
            train_start_utc=fold.train_start_utc,
            train_end_utc=fold.train_end_utc,
            test_start_utc=fold.test_start_utc,
            test_end_utc=fold.test_end_utc,
            metrics={
                "return_pct": 1.0,
                "annualized_sharpe": -999.0,
                "sharpe_like": 0.1,
                "max_drawdown_pct": 1.0,
            },
            baseline_metrics={},
            baseline_deltas={},
            warnings=tuple(),
            backtest=backtest,
        )

    monkeypatch.setattr(runner, "_evaluate_fold", fake_evaluate_fold)

    _, aggregate_metrics = runner._evaluate_preview_folds(preview.spec, preview)

    assert aggregate_metrics["annualized_sharpe"] == pytest.approx(
        annualized_sharpe_for_backtests(tuple(fold_backtests.values()))
    )
