from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from statistics import mean
from zoneinfo import ZoneInfo

import pytest

from trader.data.models import MarketBar
from trader.cli.eval_cmd import _load_payload
from trader.evaluation.baselines import baseline_deltas
from trader.evaluation.metrics import MINUTE_BARS_PER_YEAR, aggregate_metric_dicts, annualized_sharpe, calculate_metrics
from trader.evaluation.splits import build_walk_forward_folds
from trader.execution.engine import BacktestResult
from trader.execution.fills import Trade
from trader.strategies.registry import REGISTRY
from trader.strategies.spec import ExecConfig, FilterSpec, SignalSpec, StrategySpec


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
    assert "annualized_sharpe" in metrics


def test_annualized_sharpe_uses_per_bar_returns_directly() -> None:
    starting_equity = 100_000.0
    returns = (0.001, -0.0005, 0.0015)
    equity = []
    current = starting_equity
    for period_return in returns:
        current *= 1.0 + period_return
        equity.append(current)

    avg_return = mean(returns)
    variance = mean([(value - avg_return) ** 2 for value in returns])
    expected = math.sqrt(MINUTE_BARS_PER_YEAR) * avg_return / math.sqrt(variance)

    assert annualized_sharpe(tuple(equity), starting_equity=starting_equity) == pytest.approx(expected)


def test_baseline_deltas_include_annualized_sharpe() -> None:
    deltas = baseline_deltas(
        {
            "return_pct": 3.0,
            "annualized_sharpe": 1.5,
            "sharpe_like": 0.6,
        },
        {
            "buy_and_hold": {
                "return_pct": 1.0,
                "annualized_sharpe": 0.25,
                "sharpe_like": 0.2,
            },
        },
    )

    assert deltas["delta_buy_and_hold_return_pct"] == 2.0
    assert deltas["delta_buy_and_hold_annualized_sharpe"] == 1.25
    assert deltas["delta_buy_and_hold_sharpe_like"] == pytest.approx(0.4)


def test_aggregate_metrics_weights_by_fold_size() -> None:
    metrics = aggregate_metric_dicts(
        (
            {"return_pct": 1.0, "sharpe_like": 0.1},
            {"return_pct": 5.0, "sharpe_like": 0.5},
        ),
        weights=(10, 30),
    )

    assert metrics["return_pct"] == 4.0
    assert metrics["sharpe_like"] == 0.4


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


def test_regular_session_filter_is_hash_equivalent_to_exec_config_default() -> None:
    with_filter = REGISTRY.validate_spec(
        StrategySpec(
            name="ema_with_filter",
            signal=SignalSpec("ema_cross", {"fast_length": 20, "slow_length": 80, "signal_buffer_bps": 0.0}),
            filters=(FilterSpec("session", {"session": "regular"}),),
        )
    )
    without_filter = REGISTRY.validate_spec(
        StrategySpec(
            name="ema_without_filter",
            signal=SignalSpec("ema_cross", {"fast_length": 20, "slow_length": 80, "signal_buffer_bps": 0.0}),
        )
    )
    default_filter = REGISTRY.validate_spec(
        StrategySpec(
            name="ema_default_filter",
            signal=SignalSpec("ema_cross", {"fast_length": 20, "slow_length": 80, "signal_buffer_bps": 0.0}),
            filters=(FilterSpec("session", {}),),
        )
    )

    assert with_filter.spec_hash() == without_filter.spec_hash()
    assert default_filter.spec_hash() == without_filter.spec_hash()
    assert with_filter.filters == tuple()
    assert with_filter.to_payload(include_name=False)["filters"] == []


def test_unsupported_session_filter_still_rejected() -> None:
    with pytest.raises(ValueError, match="Only regular-session filtering is supported"):
        REGISTRY.validate_spec(
            StrategySpec(
                name="ema_bad_session",
                signal=SignalSpec("ema_cross", {"fast_length": 20, "slow_length": 80, "signal_buffer_bps": 0.0}),
                filters=(FilterSpec("session", {"session": "extended"}),),
            )
        )


def test_strategy_validation_rejects_non_finite_numeric_values() -> None:
    with pytest.raises(ValueError, match="ema_cross.signal_buffer_bps must be finite"):
        REGISTRY.validate_spec(
            StrategySpec(
                name="bad_nan_param",
                signal=SignalSpec(
                    "ema_cross",
                    {"fast_length": 20, "slow_length": 80, "signal_buffer_bps": float("nan")},
                ),
            )
        )

    with pytest.raises(ValueError, match="exec_config.initial_cash must be finite"):
        REGISTRY.validate_spec(
            StrategySpec(
                name="bad_nan_cash",
                exec_config=ExecConfig(initial_cash=float("nan")),
            )
        )
    with pytest.raises(ValueError, match="exec_config.stop_loss_bps must be finite"):
        REGISTRY.validate_spec(
            StrategySpec(
                name="bad_nan_stop",
                exec_config=ExecConfig(stop_loss_bps=float("nan")),
            )
        )
    with pytest.raises(ValueError, match="exec_config.no_new_entry_minutes_before_close must be finite"):
        REGISTRY.validate_spec(
            StrategySpec(
                name="bad_nan_entry_cutoff",
                exec_config=ExecConfig(no_new_entry_minutes_before_close=float("nan")),  # type: ignore[arg-type]
            )
        )


@pytest.mark.parametrize("constant", ("NaN", "Infinity", "-Infinity"))
def test_strategy_json_input_rejects_non_standard_numbers(constant: str) -> None:
    with pytest.raises(SystemExit, match="non-standard JSON numeric value"):
        _load_payload(
            None,
            f'{{"name": "bad", "signal": {{"name": "ema_cross", "params": {{"signal_buffer_bps": {constant}}}}}}}',
        )


def test_invalid_breakout_spec_rejected() -> None:
    spec = StrategySpec(
        name="bad_breakout",
        signal=SignalSpec("breakout", {"entry_window": 10, "exit_window": 20, "buffer_bps": 0.0}),
    )
    with pytest.raises(ValueError):
        REGISTRY.validate_spec(spec)
