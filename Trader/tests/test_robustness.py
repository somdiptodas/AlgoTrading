from __future__ import annotations

from dataclasses import replace
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


class _AlwaysLongRegistry:
    def __init__(self) -> None:
        self.generated_calls: list[tuple[tuple[MarketBar, ...], tuple[MarketBar, ...]]] = []

    def generate_regime(
        self,
        spec: StrategySpec,
        train_bars: tuple[MarketBar, ...],
        test_bars: tuple[MarketBar, ...],
    ) -> tuple[bool, ...]:
        self.generated_calls.append((train_bars, test_bars))
        return tuple(True for _ in test_bars)

    def compute_sizing_fraction(self, spec: StrategySpec) -> float:
        return 1.0


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


def _indexed_fold(fold_id: str, train_start: int, train_end: int, test_start: int, test_end: int) -> Fold:
    return Fold(
        fold_id=fold_id,
        train_start_idx=train_start,
        train_end_idx=train_end,
        test_start_idx=test_start,
        test_end_idx=test_end,
        embargo_bars=0,
        train_start_utc=f"train-start-{fold_id}",
        train_end_utc=f"train-end-{fold_id}",
        test_start_utc=f"test-start-{fold_id}",
        test_end_utc=f"test-end-{fold_id}",
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

    def fake_evaluate_fold(fold_spec: StrategySpec, fold: Fold, bars, **kwargs) -> FoldResult:
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

    monkeypatch.setattr(
        runner,
        "_evaluate_stage_a",
        lambda preview: (
            fake_evaluate_fold(preview.spec, preview.folds[-1], preview.data_slice.bars),
            {
                "return_pct": 1.0,
                "trade_count": 10.0,
                "max_drawdown_pct": 1.0,
                "exposure_pct": 10.0,
            },
        ),
    )
    monkeypatch.setattr(runner, "_evaluate_fold", fake_evaluate_fold)
    monkeypatch.setattr("trader.evaluation.runner.assess_robustness", fake_assess_robustness)

    runner.evaluate_preview(preview)

    assert neighbor_fold_ids == ["fold_1", "fold_2", "fold_3"]
    assert captured_neighbor_metrics["return_pct"] == 3.8
    assert captured_neighbor_metrics["sharpe_like"] == 0.38
    assert {"stage_a", "stage_b", "robustness_neighbors"} <= set(runner.phase_timings)


def test_evaluate_preview_stage_a_rejects_before_stage_b_and_robustness(monkeypatch) -> None:
    runner = EvaluationRunner.__new__(EvaluationRunner)
    folds = (_fold("fold_1"), _fold("fold_2"))
    spec = REGISTRY.validate_spec(
        StrategySpec(
            name="ema_test",
            signal=SignalSpec("ema_cross", {"fast_length": 20, "slow_length": 80, "signal_buffer_bps": 0.0}),
        )
    )
    preview = EvaluationPreview(
        spec=spec,
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
    stage_a_fold = FoldResult(
        fold_id="fold_2",
        train_start_utc=folds[-1].train_start_utc,
        train_end_utc=folds[-1].train_end_utc,
        test_start_utc=folds[-1].test_start_utc,
        test_end_utc=folds[-1].test_end_utc,
        metrics={
            "return_pct": 0.0,
            "trade_count": 10.0,
            "max_drawdown_pct": 1.0,
            "exposure_pct": 10.0,
        },
        baseline_metrics={},
        baseline_deltas={},
        warnings=tuple(),
        backtest=_backtest(tuple(), bar_count=10),
    )

    monkeypatch.setattr(runner, "_evaluate_stage_a", lambda preview: (stage_a_fold, dict(stage_a_fold.metrics)))
    monkeypatch.setattr(
        runner,
        "_evaluate_preview_folds",
        lambda spec, preview: pytest.fail("stage B should not run for Stage-A rejects"),
    )
    monkeypatch.setattr(
        "trader.evaluation.runner.assess_robustness",
        lambda **kwargs: pytest.fail("robustness should not run for Stage-A rejects"),
    )

    result = runner.evaluate_preview(preview)

    assert result.promotion_stage == "exploratory"
    assert result.fold_results == (stage_a_fold,)
    assert result.aggregate_metrics["stage_a_pass"] == 0.0
    assert result.robustness_checks == {
        "stage_a_pass": False,
        "stage_a_reject_non_positive_return": True,
    }
    assert "stage_a" in runner.phase_timings
    assert "stage_b" not in runner.phase_timings
    assert "robustness_neighbors" not in runner.phase_timings


def test_evaluate_stage_a_preview_returns_same_rejection_shape(monkeypatch) -> None:
    runner = EvaluationRunner.__new__(EvaluationRunner)
    folds = (_fold("fold_1"),)
    spec = REGISTRY.validate_spec(
        StrategySpec(
            name="ema_test",
            signal=SignalSpec("ema_cross", {"fast_length": 20, "slow_length": 80, "signal_buffer_bps": 0.0}),
        )
    )
    preview = EvaluationPreview(
        spec=spec,
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
    stage_a_fold = FoldResult(
        fold_id="fold_1",
        train_start_utc=folds[-1].train_start_utc,
        train_end_utc=folds[-1].train_end_utc,
        test_start_utc=folds[-1].test_start_utc,
        test_end_utc=folds[-1].test_end_utc,
        metrics={
            "return_pct": 0.0,
            "trade_count": 10.0,
            "max_drawdown_pct": 1.0,
            "exposure_pct": 10.0,
        },
        baseline_metrics={},
        baseline_deltas={},
        warnings=tuple(),
        backtest=_backtest(tuple(), bar_count=10),
    )
    monkeypatch.setattr(runner, "_evaluate_stage_a", lambda preview: (stage_a_fold, dict(stage_a_fold.metrics)))

    result = runner.evaluate_stage_a_preview(preview)

    assert result is not None
    assert result.fold_results == (stage_a_fold,)
    assert result.robustness_checks == {
        "stage_a_pass": False,
        "stage_a_reject_non_positive_return": True,
    }


def test_evaluate_preview_runs_stage_b_when_stage_a_passes(monkeypatch) -> None:
    runner = EvaluationRunner.__new__(EvaluationRunner)
    runner.registry = REGISTRY
    folds = (_fold("fold_1"),)
    spec = REGISTRY.validate_spec(
        StrategySpec(
            name="ema_test",
            signal=SignalSpec("ema_cross", {"fast_length": 20, "slow_length": 80, "signal_buffer_bps": 0.0}),
        )
    )
    preview = EvaluationPreview(
        spec=spec,
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
    stage_a_fold = FoldResult(
        fold_id="fold_1",
        train_start_utc=folds[-1].train_start_utc,
        train_end_utc=folds[-1].train_end_utc,
        test_start_utc=folds[-1].test_start_utc,
        test_end_utc=folds[-1].test_end_utc,
        metrics={
            "return_pct": 1.0,
            "trade_count": 10.0,
            "max_drawdown_pct": 1.0,
            "exposure_pct": 10.0,
        },
        baseline_metrics={},
        baseline_deltas={},
        warnings=tuple(),
        backtest=_backtest(tuple(), bar_count=10),
    )
    stage_b_metrics = {
        "return_pct": 2.0,
        "annualized_sharpe": 1.0,
        "sharpe_like": 0.5,
        "max_drawdown_pct": 1.0,
        "trade_count": 10.0,
        "delta_buy_and_hold_return_pct": 1.0,
        "delta_exposure_adjusted_buy_and_hold_pct": 1.0,
    }
    stage_b_fold = replace(stage_a_fold, metrics=stage_b_metrics)

    monkeypatch.setattr(runner, "_evaluate_stage_a", lambda preview: (stage_a_fold, dict(stage_a_fold.metrics)))
    monkeypatch.setattr(runner, "_evaluate_preview_folds", lambda spec, preview: ((stage_b_fold,), stage_b_metrics))
    monkeypatch.setattr(runner, "_add_cost_scenario_metrics", lambda spec, fold, bars, fold_result, **kwargs: fold_result)
    monkeypatch.setattr(
        "trader.evaluation.runner.assess_robustness",
        lambda **kwargs: RobustnessResult(
            checks={
                "fold_consistency_pass": True,
                "regime_pass": True,
                "neighborhood_pass": True,
                "drawdown_pass": True,
            },
            passed=True,
        ),
    )

    result = runner.evaluate_preview(preview)

    assert result.aggregate_metrics["return_pct"] == 2.0
    assert result.aggregate_metrics["stage_a_pass"] == 1.0
    assert result.robustness_checks["neighborhood_pass"] is True


def test_evaluate_preview_can_skip_stage_a_prescreen(monkeypatch) -> None:
    runner = EvaluationRunner.__new__(EvaluationRunner)
    runner.registry = REGISTRY
    folds = (_fold("fold_1"),)
    spec = REGISTRY.validate_spec(
        StrategySpec(
            name="ema_test",
            signal=SignalSpec("ema_cross", {"fast_length": 20, "slow_length": 80, "signal_buffer_bps": 0.0}),
        )
    )
    preview = EvaluationPreview(
        spec=spec,
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
    metrics = {
        "return_pct": 2.0,
        "annualized_sharpe": 1.0,
        "sharpe_like": 0.5,
        "max_drawdown_pct": 1.0,
        "trade_count": 10.0,
        "delta_buy_and_hold_return_pct": 1.0,
        "delta_exposure_adjusted_buy_and_hold_pct": 1.0,
    }
    fold = FoldResult(
        fold_id="fold_1",
        train_start_utc=folds[-1].train_start_utc,
        train_end_utc=folds[-1].train_end_utc,
        test_start_utc=folds[-1].test_start_utc,
        test_end_utc=folds[-1].test_end_utc,
        metrics=metrics,
        baseline_metrics={},
        baseline_deltas={},
        warnings=tuple(),
        backtest=_backtest(tuple(), bar_count=10),
    )
    monkeypatch.setattr(runner, "_evaluate_stage_a", lambda preview: pytest.fail("Stage A should be skipped"))
    monkeypatch.setattr(runner, "_evaluate_preview_folds", lambda spec, preview: ((fold,), metrics))
    monkeypatch.setattr(runner, "_add_cost_scenario_metrics", lambda spec, fold, bars, fold_result, **kwargs: fold_result)
    monkeypatch.setattr(
        "trader.evaluation.runner.assess_robustness",
        lambda **kwargs: RobustnessResult(
            checks={
                "fold_consistency_pass": True,
                "regime_pass": True,
                "neighborhood_pass": True,
                "drawdown_pass": True,
            },
            passed=True,
        ),
    )

    result = runner.evaluate_preview(preview, run_stage_a=False)

    assert result.aggregate_metrics["return_pct"] == 2.0
    assert "stage_a_pass" not in result.aggregate_metrics


def test_evaluate_preview_skips_stage_a_when_robustness_is_disabled(monkeypatch) -> None:
    runner = EvaluationRunner.__new__(EvaluationRunner)
    folds = (_fold("fold_1"),)
    spec = REGISTRY.validate_spec(
        StrategySpec(
            name="ema_test",
            signal=SignalSpec("ema_cross", {"fast_length": 20, "slow_length": 80, "signal_buffer_bps": 0.0}),
        )
    )
    preview = EvaluationPreview(
        spec=spec,
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
    stage_b_metrics = {
        "return_pct": 2.0,
        "annualized_sharpe": 1.0,
        "sharpe_like": 0.5,
        "max_drawdown_pct": 1.0,
        "trade_count": 10.0,
        "delta_buy_and_hold_return_pct": 1.0,
        "delta_exposure_adjusted_buy_and_hold_pct": 1.0,
    }

    monkeypatch.setattr(runner, "_evaluate_stage_a", lambda preview: pytest.fail("Stage A should be skipped"))
    monkeypatch.setattr(
        runner,
        "_evaluate_preview_folds",
        lambda spec, preview: (
            (
                FoldResult(
                    fold_id="fold_1",
                    train_start_utc=folds[-1].train_start_utc,
                    train_end_utc=folds[-1].train_end_utc,
                    test_start_utc=folds[-1].test_start_utc,
                    test_end_utc=folds[-1].test_end_utc,
                    metrics=stage_b_metrics,
                    baseline_metrics={},
                    baseline_deltas={},
                    warnings=tuple(),
                    backtest=_backtest(tuple(), bar_count=10),
                ),
            ),
            stage_b_metrics,
        ),
    )

    result = runner.evaluate_preview(preview, include_robustness=False)

    assert result.aggregate_metrics["return_pct"] == 2.0
    assert "stage_a_pass" not in result.aggregate_metrics
    assert result.robustness_checks == {}


def test_stage_a_uses_most_recent_fold_without_baselines_or_cost_stress(monkeypatch) -> None:
    runner = EvaluationRunner.__new__(EvaluationRunner)
    registry = _AlwaysLongRegistry()
    runner.registry = registry
    spec = REGISTRY.validate_spec(
        StrategySpec(
            name="ema_test",
            signal=SignalSpec("ema_cross", {"fast_length": 20, "slow_length": 80, "signal_buffer_bps": 0.0}),
        )
    )
    preview = EvaluationPreview(
        spec=spec,
        data_slice=DataSlice(
            bars=_bars(6),
            snapshot_id="snapshot",
            first_timestamp_utc=None,
            last_timestamp_utc=None,
        ),
        split_plan_id="split-plan",
        folds=(
            _indexed_fold("fold_1", 0, 1, 2, 3),
            _indexed_fold("fold_2", 0, 3, 4, 5),
        ),
        required_history=80,
        cost_model_id="cost-model",
        evaluation_key="evaluation-key",
    )

    monkeypatch.setattr(
        runner,
        "_cost_scenario_metrics",
        lambda *args, **kwargs: pytest.fail("Stage A should not run cost stress"),
    )
    monkeypatch.setattr(
        "trader.evaluation.runner.evaluate_baselines",
        lambda *args, **kwargs: pytest.fail("Stage A should not run baselines"),
    )

    stage_a = runner._evaluate_stage_a(preview)

    assert stage_a is not None
    fold_result, metrics = stage_a
    assert fold_result.fold_id == "fold_2"
    assert [bar.timestamp_utc for bar in fold_result.backtest.bars] == [
        preview.data_slice.bars[4].timestamp_utc,
        preview.data_slice.bars[5].timestamp_utc,
    ]
    assert len(registry.generated_calls) == 1
    train_bars, test_bars = registry.generated_calls[0]
    assert train_bars == preview.data_slice.bars[0:4]
    assert test_bars == preview.data_slice.bars[4:6]
    assert fold_result.baseline_metrics == {}
    assert fold_result.baseline_deltas == {}
    assert "return_pct" in metrics


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

    def fake_evaluate_fold(fold_spec: StrategySpec, fold: Fold, bars, **kwargs) -> FoldResult:
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


def test_aggregate_fold_results_compounds_regime_returns() -> None:
    runner = EvaluationRunner.__new__(EvaluationRunner)
    folds = (
        FoldResult(
            fold_id="fold_1",
            train_start_utc="",
            train_end_utc="",
            test_start_utc="",
            test_end_utc="",
            metrics={
                "return_pct": 1.0,
                "annualized_sharpe": 0.0,
                "information_ratio_vs_buy_and_hold": 0.0,
                "p_value_vs_random_entry": 1.0,
                "regime_trend_return_pct": 10.0,
                "regime_trend_day_count": 1.0,
            },
            baseline_metrics={},
            baseline_deltas={},
            warnings=tuple(),
            backtest=_backtest(tuple(), bar_count=2),
        ),
        FoldResult(
            fold_id="fold_2",
            train_start_utc="",
            train_end_utc="",
            test_start_utc="",
            test_end_utc="",
            metrics={
                "return_pct": 1.0,
                "annualized_sharpe": 0.0,
                "information_ratio_vs_buy_and_hold": 0.0,
                "p_value_vs_random_entry": 1.0,
                "regime_trend_return_pct": -10.0,
                "regime_trend_day_count": 2.0,
            },
            baseline_metrics={},
            baseline_deltas={},
            warnings=tuple(),
            backtest=_backtest(tuple(), bar_count=2),
        ),
    )

    aggregate = runner._aggregate_fold_results(folds)

    assert aggregate["regime_trend_return_pct"] == pytest.approx(-1.0)
    assert aggregate["regime_trend_day_count"] == 3.0
