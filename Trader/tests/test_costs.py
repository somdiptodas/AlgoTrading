from __future__ import annotations

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from trader.data.models import MarketBar
from trader.data.view import DataSlice
from trader.evaluation.metrics import calculate_metrics
from trader.evaluation.robustness import RobustnessResult
from trader.evaluation.runner import EvaluationPreview, EvaluationRunner, ExperimentResult, FoldResult
from trader.evaluation.splits import Fold
from trader.execution.engine import BacktestResult, run_long_only_engine
from trader.execution.fills import Trade, enter_long, exit_long
from trader.reporting.report import render_experiment_report
from trader.strategies.registry import REGISTRY
from trader.strategies.spec import ExecConfig, SignalSpec, StrategySpec


NEW_YORK = ZoneInfo("America/New_York")


class _AlwaysLongRegistry:
    def generate_regime(
        self,
        spec: StrategySpec,
        train_bars: tuple[MarketBar, ...],
        test_bars: tuple[MarketBar, ...],
    ) -> tuple[bool, ...]:
        return tuple(True for _ in test_bars)

    def compute_sizing_fraction(self, spec: StrategySpec) -> float:
        return 1.0

    def neighbors(self, spec: StrategySpec) -> tuple[StrategySpec, ...]:
        return tuple()


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


def _fold() -> Fold:
    return Fold(
        fold_id="fold_1",
        train_start_idx=0,
        train_end_idx=1,
        test_start_idx=2,
        test_end_idx=3,
        embargo_bars=0,
        train_start_utc=_bar(0).timestamp_utc,
        train_end_utc=_bar(1).timestamp_utc,
        test_start_utc=_bar(2).timestamp_utc,
        test_end_utc=_bar(3).timestamp_utc,
    )


def _preview(spec: StrategySpec, bars: tuple[MarketBar, ...], fold: Fold) -> EvaluationPreview:
    return EvaluationPreview(
        spec=spec,
        data_slice=DataSlice(
            bars=bars,
            snapshot_id="snapshot",
            first_timestamp_utc=bars[0].timestamp_utc,
            last_timestamp_utc=bars[-1].timestamp_utc,
        ),
        split_plan_id="split",
        folds=(fold,),
        required_history=3,
        cost_model_id=spec.exec_config.cost_model_id(),
        evaluation_key="key",
    )


def _fold_result(fold: Fold, metrics: dict[str, float], bars: tuple[MarketBar, ...]) -> FoldResult:
    return FoldResult(
        fold_id=fold.fold_id,
        train_start_utc=fold.train_start_utc,
        train_end_utc=fold.train_end_utc,
        test_start_utc=fold.test_start_utc,
        test_end_utc=fold.test_end_utc,
        metrics=metrics,
        baseline_metrics={},
        baseline_deltas={},
        warnings=tuple(),
        backtest=BacktestResult(
            bars=bars[fold.test_start_idx : fold.test_end_idx + 1],
            trades=tuple(),
            equity_curve=(100_000.0, 101_000.0),
            initial_cash=100_000.0,
            final_cash=101_000.0,
        ),
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


def test_entry_session_window_first_30m_blocks_later_entries() -> None:
    config = ExecConfig(initial_cash=100_000.0, entry_session_window="first_30m")
    bars = tuple(_bar(index, 100.0) for index in range(35))
    regime = tuple(index == 31 for index in range(35))

    result = run_long_only_engine(bars, regime, config)

    assert result.trades == tuple()


def test_entry_session_window_first_30m_allows_early_entries() -> None:
    config = ExecConfig(initial_cash=100_000.0, entry_session_window="first_30m")
    bars = tuple(_bar(index, 100.0) for index in range(5))
    regime = tuple(index == 0 for index in range(5))

    result = run_long_only_engine(bars, regime, config)

    assert len(result.trades) == 1
    assert result.trades[0].entry_timestamp_utc == bars[1].timestamp_utc


def test_entry_session_window_first_30m_uses_fill_bar_boundary() -> None:
    config = ExecConfig(initial_cash=100_000.0, entry_session_window="first_30m")
    bars = tuple(_bar(index, 100.0) for index in range(32))
    regime = tuple(index == 29 for index in range(32))

    result = run_long_only_engine(bars, regime, config)

    assert result.trades == tuple()


def test_entry_session_window_last_30m_allows_late_entries() -> None:
    config = ExecConfig(initial_cash=100_000.0, entry_session_window="last_30m")
    bars = tuple(_bar(index, 100.0) for index in range(365))
    regime = tuple(index == 359 for index in range(365))

    result = run_long_only_engine(bars, regime, config)

    assert len(result.trades) == 1
    assert result.trades[0].entry_timestamp_utc == bars[360].timestamp_utc


def test_entry_session_window_last_30m_blocks_earlier_entries() -> None:
    config = ExecConfig(initial_cash=100_000.0, entry_session_window="last_30m")
    bars = tuple(_bar(index, 100.0) for index in range(365))
    regime = tuple(index == 358 for index in range(365))

    result = run_long_only_engine(bars, regime, config)

    assert result.trades == tuple()


def test_entry_session_window_avoid_midday_blocks_midday_entries() -> None:
    config = ExecConfig(initial_cash=100_000.0, entry_session_window="avoid_midday")
    bars = tuple(_bar(index, 100.0) for index in range(200))
    regime = tuple(index == 150 for index in range(200))

    result = run_long_only_engine(bars, regime, config)

    assert result.trades == tuple()


def test_entry_session_window_avoid_midday_allows_morning_entries() -> None:
    config = ExecConfig(initial_cash=100_000.0, entry_session_window="avoid_midday")
    bars = tuple(_bar(index, 100.0) for index in range(20))
    regime = tuple(index == 0 for index in range(20))

    result = run_long_only_engine(bars, regime, config)

    assert len(result.trades) == 1
    assert result.trades[0].entry_timestamp_utc == bars[1].timestamp_utc


def test_no_new_entry_cutoff_blocks_entries_before_close() -> None:
    config = ExecConfig(initial_cash=100_000.0, no_new_entry_minutes_before_close=30)
    bars = tuple(_bar(index, 100.0) for index in range(365))
    regime = tuple(index == 360 for index in range(365))

    result = run_long_only_engine(bars, regime, config)

    assert result.trades == tuple()


def test_no_new_entry_cutoff_allows_entries_before_cutoff() -> None:
    config = ExecConfig(initial_cash=100_000.0, no_new_entry_minutes_before_close=30)
    bars = tuple(_bar(index, 100.0) for index in range(365))
    regime = tuple(index == 328 for index in range(365))

    result = run_long_only_engine(bars, regime, config)

    assert len(result.trades) == 1
    assert result.trades[0].entry_timestamp_utc == bars[329].timestamp_utc


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


def test_evaluate_fold_does_not_run_cost_stress(monkeypatch) -> None:
    runner = EvaluationRunner.__new__(EvaluationRunner)
    runner.registry = REGISTRY
    runner._fold_result_cache = {}
    bars = tuple(_bar(index, 100.0 + index) for index in range(8))
    fold = Fold(
        fold_id="fold_1",
        train_start_idx=0,
        train_end_idx=3,
        test_start_idx=4,
        test_end_idx=7,
        embargo_bars=0,
        train_start_utc=bars[0].timestamp_utc,
        train_end_utc=bars[3].timestamp_utc,
        test_start_utc=bars[4].timestamp_utc,
        test_end_utc=bars[7].timestamp_utc,
    )
    spec = REGISTRY.validate_spec(
        StrategySpec(
            name="base_only_cost",
            signal=SignalSpec("ema_cross", {"fast_length": 2, "slow_length": 3, "signal_buffer_bps": 0.0}),
        )
    )

    monkeypatch.setattr(
        runner,
        "_cost_scenario_metrics",
        lambda *args, **kwargs: pytest.fail("_evaluate_fold should not run cost stress"),
    )

    result = runner._evaluate_fold(spec, fold, bars)

    assert "cost_scenario_slippage_plus_2bps_return_pct" not in result.metrics
    assert "cost_drag_return_pct" not in result.metrics


def test_evaluate_preview_adds_cost_stress_after_base_gates_pass(monkeypatch) -> None:
    runner = EvaluationRunner.__new__(EvaluationRunner)
    runner.registry = _AlwaysLongRegistry()
    bars = tuple(_bar(index, 100.0 + index) for index in range(4))
    fold = _fold()
    spec = REGISTRY.validate_spec(
        StrategySpec(
            name="cost_after_gate",
            signal=SignalSpec("ema_cross", {"fast_length": 2, "slow_length": 3, "signal_buffer_bps": 0.0}),
        )
    )
    metrics = {
        "return_pct": 1.0,
        "annualized_sharpe": 1.0,
        "sharpe_like": 0.5,
        "max_drawdown_pct": 1.0,
        "trade_count": 10.0,
        "delta_buy_and_hold_return_pct": 0.25,
        "delta_exposure_adjusted_buy_and_hold_pct": 0.25,
        "information_ratio_vs_buy_and_hold": 0.6,
        "p_value_vs_random_entry": 0.05,
    }
    fold_result = _fold_result(fold, metrics, bars)

    monkeypatch.setattr(runner, "_evaluate_stage_a", lambda preview: (fold_result, {"return_pct": 1.0, "trade_count": 10.0, "max_drawdown_pct": 1.0, "exposure_pct": 10.0}))
    monkeypatch.setattr(runner, "_evaluate_preview_folds", lambda spec, preview: ((fold_result,), metrics))
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
    monkeypatch.setattr(
        runner,
        "_cost_scenario_metrics",
        lambda *args, **kwargs: {"cost_drag_return_pct": 0.1},
    )

    result = runner.evaluate_preview(_preview(spec, bars, fold))

    assert result.promotion_stage == "candidate"
    assert result.fold_results[0].metrics["cost_drag_return_pct"] == 0.1
    assert result.aggregate_metrics["cost_drag_return_pct"] == 0.1


def test_evaluate_preview_skips_cost_stress_for_exploratory_results(monkeypatch) -> None:
    runner = EvaluationRunner.__new__(EvaluationRunner)
    runner.registry = _AlwaysLongRegistry()
    bars = tuple(_bar(index, 100.0 + index) for index in range(4))
    fold = _fold()
    spec = REGISTRY.validate_spec(
        StrategySpec(
            name="cost_skip_exploratory",
            signal=SignalSpec("ema_cross", {"fast_length": 2, "slow_length": 3, "signal_buffer_bps": 0.0}),
        )
    )
    metrics = {
        "return_pct": 1.0,
        "annualized_sharpe": 1.0,
        "sharpe_like": 0.5,
        "max_drawdown_pct": 1.0,
        "trade_count": 10.0,
        "delta_buy_and_hold_return_pct": 0.25,
        "delta_exposure_adjusted_buy_and_hold_pct": 0.25,
    }
    fold_result = _fold_result(fold, metrics, bars)

    monkeypatch.setattr(runner, "_evaluate_stage_a", lambda preview: (fold_result, {"return_pct": 1.0, "trade_count": 10.0, "max_drawdown_pct": 1.0, "exposure_pct": 10.0}))
    monkeypatch.setattr(runner, "_evaluate_preview_folds", lambda spec, preview: ((fold_result,), metrics))
    monkeypatch.setattr("trader.evaluation.runner.promotion_stage", lambda aggregate, checks: "exploratory")
    monkeypatch.setattr(
        runner,
        "_cost_scenario_metrics",
        lambda *args, **kwargs: pytest.fail("exploratory results should skip cost stress"),
    )

    result = runner.evaluate_preview(_preview(spec, bars, fold))

    assert result.promotion_stage == "exploratory"
    assert "cost_drag_return_pct" not in result.aggregate_metrics


def test_evaluate_preview_skips_cost_stress_when_robustness_is_disabled(monkeypatch) -> None:
    runner = EvaluationRunner.__new__(EvaluationRunner)
    bars = tuple(_bar(index, 100.0 + index) for index in range(4))
    fold = _fold()
    spec = REGISTRY.validate_spec(
        StrategySpec(
            name="cost_skip_no_robustness",
            signal=SignalSpec("ema_cross", {"fast_length": 2, "slow_length": 3, "signal_buffer_bps": 0.0}),
        )
    )
    metrics = {
        "return_pct": 1.0,
        "annualized_sharpe": 1.0,
        "sharpe_like": 0.5,
        "max_drawdown_pct": 1.0,
        "trade_count": 10.0,
        "delta_buy_and_hold_return_pct": 1.0,
        "delta_exposure_adjusted_buy_and_hold_pct": 1.0,
    }
    fold_result = _fold_result(fold, metrics, bars)

    monkeypatch.setattr(runner, "_evaluate_stage_a", lambda preview: pytest.fail("Stage A should be skipped"))
    monkeypatch.setattr(runner, "_evaluate_preview_folds", lambda spec, preview: ((fold_result,), metrics))
    monkeypatch.setattr(
        runner,
        "_cost_scenario_metrics",
        lambda *args, **kwargs: pytest.fail("include_robustness=False should skip cost stress"),
    )

    result = runner.evaluate_preview(_preview(spec, bars, fold), include_robustness=False)

    assert result.promotion_stage == "exploratory"
    assert "cost_drag_return_pct" not in result.aggregate_metrics


def test_report_omits_fold_cost_drag_when_cost_stress_is_skipped() -> None:
    bars = tuple(_bar(index, 100.0 + index) for index in range(4))
    fold = _fold()
    spec = REGISTRY.validate_spec(
        StrategySpec(
            name="cost_report",
            signal=SignalSpec("ema_cross", {"fast_length": 2, "slow_length": 3, "signal_buffer_bps": 0.0}),
        )
    )
    metrics = {
        "return_pct": 1.0,
        "sharpe_like": 0.5,
        "max_drawdown_pct": 1.0,
        "trade_count": 10.0,
    }
    result = ExperimentResult(
        experiment_id="cost_report",
        status="completed",
        spec=spec,
        spec_hash=spec.spec_hash(),
        data_snapshot_id="snapshot",
        split_plan_id="split",
        cost_model_id=spec.exec_config.cost_model_id(),
        aggregate_metrics=metrics,
        fold_results=(_fold_result(fold, metrics, bars),),
        robustness_checks={},
        promotion_stage="exploratory",
    )

    report = render_experiment_report(result)

    assert "cost_drag=" not in report


def test_registry_rejects_invalid_cost_config_values() -> None:
    with pytest.raises(ValueError, match="commission_per_share"):
        REGISTRY.validate_spec(StrategySpec(name="bad_commission", exec_config=ExecConfig(commission_per_share=-0.01)))
    with pytest.raises(ValueError, match="spread_bps"):
        REGISTRY.validate_spec(StrategySpec(name="bad_spread", exec_config=ExecConfig(spread_bps=-1.0)))
    with pytest.raises(ValueError, match="max_position_notional"):
        REGISTRY.validate_spec(StrategySpec(name="bad_notional", exec_config=ExecConfig(max_position_notional=0.0)))
    with pytest.raises(ValueError, match="stop_loss_bps"):
        REGISTRY.validate_spec(StrategySpec(name="bad_stop_loss", exec_config=ExecConfig(stop_loss_bps=0.0)))
    with pytest.raises(ValueError, match="entry_session_window"):
        REGISTRY.validate_spec(StrategySpec(name="bad_entry_window", exec_config=ExecConfig(entry_session_window="lunch")))
    with pytest.raises(ValueError, match="no_new_entry_minutes_before_close"):
        REGISTRY.validate_spec(
            StrategySpec(name="bad_entry_cutoff", exec_config=ExecConfig(no_new_entry_minutes_before_close=-1))
        )
    with pytest.raises(ValueError, match="no_new_entry_minutes_before_close"):
        REGISTRY.validate_spec(
            StrategySpec(name="too_large_entry_cutoff", exec_config=ExecConfig(no_new_entry_minutes_before_close=391))
        )
