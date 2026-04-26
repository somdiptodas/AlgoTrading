from __future__ import annotations

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from trader.data.models import MarketBar
from trader.evaluation.runner import EvaluationRunner, FoldResult, _aggregate_random_entry_p_value
from trader.evaluation.splits import Fold
from trader.evaluation.baselines import baseline_deltas, evaluate_baselines, randomized_entry_same_exposure
from trader.execution.engine import BacktestResult
from trader.execution.fills import Trade
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


def _fold(bars: tuple[MarketBar, ...]) -> Fold:
    return Fold(
        fold_id="fold_1",
        train_start_idx=0,
        train_end_idx=1,
        test_start_idx=2,
        test_end_idx=len(bars) - 1,
        embargo_bars=0,
        train_start_utc=bars[0].timestamp_utc,
        train_end_utc=bars[1].timestamp_utc,
        test_start_utc=bars[2].timestamp_utc,
        test_end_utc=bars[-1].timestamp_utc,
    )


def _baseline_metrics(return_pct: float = 0.0) -> dict[str, float]:
    return {
        "return_pct": return_pct,
        "annualized_sharpe": 0.0,
        "sharpe_like": 0.0,
        "max_drawdown_pct": 0.0,
        "exposure_pct": 0.0,
        "trade_count": 0.0,
        "win_rate_pct": 0.0,
        "profit_factor": 0.0,
        "avg_trade_pct": 0.0,
        "final_cash": 100_000.0,
    }


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


def test_randomized_entry_reports_deterministic_bootstrap_p_value() -> None:
    bars = _priced_single_session_bars((100.0, 99.0, 98.0, 97.0, 96.0, 95.0))
    reference = _same_session_reference(bars)

    first = randomized_entry_same_exposure(
        bars,
        ExecConfig(initial_cash=100_000.0),
        reference.trades,
        seed_material="bootstrap-seed",
        strategy_return_pct=((reference.final_cash / reference.initial_cash) - 1.0) * 100.0,
        resample_count=20,
    )
    second = randomized_entry_same_exposure(
        bars,
        ExecConfig(initial_cash=100_000.0),
        reference.trades,
        seed_material="bootstrap-seed",
        strategy_return_pct=((reference.final_cash / reference.initial_cash) - 1.0) * 100.0,
        resample_count=20,
    )

    assert first == second
    assert first["bootstrap_sample_count"] == 20.0
    assert 0.0 <= first["p_value_vs_random_entry"] <= 1.0


def test_baseline_deltas_include_random_entry_p_value() -> None:
    deltas = baseline_deltas(
        {"return_pct": 1.0, "annualized_sharpe": 1.0, "sharpe_like": 0.5, "exposure_pct": 10.0},
        {
            "randomized_entry_same_exposure": {
                "return_pct": 0.5,
                "annualized_sharpe": 0.1,
                "sharpe_like": 0.2,
                "p_value_vs_random_entry": 0.08,
            }
        },
    )

    assert deltas["p_value_vs_random_entry"] == 0.08


def test_aggregate_random_entry_p_value_keeps_zero_sample_fold_weight() -> None:
    bars = _single_session_bars(4)
    fold_with_samples = FoldResult(
        fold_id="fold_1",
        train_start_utc=bars[0].timestamp_utc,
        train_end_utc=bars[0].timestamp_utc,
        test_start_utc=bars[0].timestamp_utc,
        test_end_utc=bars[-1].timestamp_utc,
        metrics={"return_pct": 1.0},
        baseline_metrics={},
        baseline_deltas={},
        warnings=tuple(),
        backtest=BacktestResult(
            bars=bars,
            trades=tuple(),
            equity_curve=tuple(100_000.0 for _ in bars),
            initial_cash=100_000.0,
            final_cash=100_000.0,
        ),
        random_entry_return_samples=(4.0, 4.0),
    )
    fold_without_samples = FoldResult(
        fold_id="fold_2",
        train_start_utc=bars[0].timestamp_utc,
        train_end_utc=bars[0].timestamp_utc,
        test_start_utc=bars[0].timestamp_utc,
        test_end_utc=bars[-1].timestamp_utc,
        metrics={"return_pct": 0.0},
        baseline_metrics={},
        baseline_deltas={},
        warnings=tuple(),
        backtest=BacktestResult(
            bars=bars,
            trades=tuple(),
            equity_curve=tuple(100_000.0 for _ in bars),
            initial_cash=100_000.0,
            final_cash=100_000.0,
        ),
    )

    assert _aggregate_random_entry_p_value(3.0, (fold_with_samples, fold_without_samples)) == pytest.approx(1 / 3)


def test_runner_caches_fixed_fold_baselines_but_not_randomized_entry(monkeypatch) -> None:
    bars = _bars()
    fold = _fold(bars)
    runner = EvaluationRunner.__new__(EvaluationRunner)
    runner.registry = _AlwaysLongRegistry()
    runner._fold_result_cache = {}
    runner._baseline_cache = {}
    calls = {
        "always_flat": 0,
        "buy_and_hold": 0,
        "regular_session_open_to_close_long": 0,
        "session_long_flat_at_close": 0,
        "randomized_entry_same_exposure": 0,
    }

    def count_fixed(name: str, return_pct: float):
        def baseline(*args, **kwargs):
            calls[name] += 1
            return _baseline_metrics(return_pct)

        return baseline

    def randomized(*args, **kwargs):
        calls["randomized_entry_same_exposure"] += 1
        return _baseline_metrics(5.0), (0.1, 0.2)

    monkeypatch.setattr("trader.evaluation.runner.always_flat", count_fixed("always_flat", 0.0))
    monkeypatch.setattr("trader.evaluation.runner.buy_and_hold", count_fixed("buy_and_hold", 1.0))
    monkeypatch.setattr(
        "trader.evaluation.runner.regular_session_open_to_close_long",
        count_fixed("regular_session_open_to_close_long", 2.0),
    )
    monkeypatch.setattr(
        "trader.evaluation.runner.session_long_flat_at_close",
        count_fixed("session_long_flat_at_close", 3.0),
    )
    monkeypatch.setattr("trader.evaluation.runner.randomized_entry_same_exposure_with_samples", randomized)

    first = runner._evaluate_fold(
        StrategySpec(name="cache_one", signal=SignalSpec("ema_cross", {"fast_length": 2, "slow_length": 3})),
        fold,
        bars,
        data_snapshot_id="snapshot_a",
    )
    first.baseline_metrics["buy_and_hold"]["return_pct"] = 999.0
    same_spec_new_snapshot = runner._evaluate_fold(
        StrategySpec(name="cache_one", signal=SignalSpec("ema_cross", {"fast_length": 2, "slow_length": 3})),
        fold,
        bars,
        data_snapshot_id="snapshot_b",
    )
    second = runner._evaluate_fold(
        StrategySpec(name="cache_two", signal=SignalSpec("ema_cross", {"fast_length": 3, "slow_length": 4})),
        fold,
        bars,
        data_snapshot_id="snapshot_a",
    )
    third = runner._evaluate_fold(
        StrategySpec(name="cache_three", signal=SignalSpec("ema_cross", {"fast_length": 4, "slow_length": 5})),
        fold,
        bars,
        data_snapshot_id="snapshot_b",
    )
    fourth = runner._evaluate_fold(
        StrategySpec(
            name="cache_four",
            signal=SignalSpec("ema_cross", {"fast_length": 5, "slow_length": 6}),
            exec_config=ExecConfig(initial_cash=200_000.0),
        ),
        fold,
        bars,
        data_snapshot_id="snapshot_b",
    )

    assert first.baseline_metrics["buy_and_hold"]["return_pct"] == 999.0
    assert same_spec_new_snapshot.baseline_metrics["buy_and_hold"]["return_pct"] == 1.0
    assert second.baseline_metrics["buy_and_hold"]["return_pct"] == 1.0
    assert second.baseline_metrics["session_long_flat_at_close"]["return_pct"] == 3.0
    assert third.baseline_metrics["buy_and_hold"]["return_pct"] == 1.0
    assert fourth.baseline_metrics["session_long_flat_at_close"]["return_pct"] == 3.0
    assert calls == {
        "always_flat": 3,
        "buy_and_hold": 3,
        "regular_session_open_to_close_long": 3,
        "session_long_flat_at_close": 3,
        "randomized_entry_same_exposure": 5,
    }


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
