from __future__ import annotations

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from trader.data.models import MarketBar
from trader.data.view import DataSlice
from trader.evaluation.runner import EvaluationPreview, EvaluationRunner, FoldResult
from trader.execution.engine import BacktestResult
from trader.strategies.registry import REGISTRY
from trader.strategies.spec import SignalSpec, StrategySpec


NEW_YORK = ZoneInfo("America/New_York")


def _bars(count: int, *, start: datetime | None = None) -> tuple[MarketBar, ...]:
    current = start or datetime(2026, 1, 5, 9, 30, tzinfo=NEW_YORK)
    bars = []
    for index in range(count):
        timestamp = (current + timedelta(minutes=index)).astimezone(timezone.utc)
        price = 100.0 + (index * 0.1)
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


def _daily_bars(count: int) -> tuple[MarketBar, ...]:
    start = datetime(2026, 1, 1, 9, 30, tzinfo=NEW_YORK)
    return tuple(_bars(1, start=start + timedelta(days=index))[0] for index in range(count))


def _data_slice(bars: tuple[MarketBar, ...]) -> DataSlice:
    return DataSlice(
        bars=bars,
        snapshot_id=EvaluationRunner.__name__ + str(len(bars)),
        first_timestamp_utc=bars[0].timestamp_utc if bars else None,
        last_timestamp_utc=bars[-1].timestamp_utc if bars else None,
    )


def _preview() -> EvaluationPreview:
    train_bars = _bars(120)
    holdout_bars = _bars(20, start=datetime(2026, 1, 6, 9, 30, tzinfo=NEW_YORK))
    spec = REGISTRY.validate_spec(
        StrategySpec(
            name="ema_holdout",
            signal=SignalSpec("ema_cross", {"fast_length": 8, "slow_length": 20, "signal_buffer_bps": 0.0}),
        )
    )
    return EvaluationPreview(
        spec=spec,
        data_slice=_data_slice(train_bars),
        split_plan_id="split",
        folds=tuple(),
        required_history=20,
        cost_model_id=spec.exec_config.cost_model_id(),
        evaluation_key="key",
        holdout_bars=holdout_bars,
        holdout_snapshot_id="holdout",
    )


def _holdout_result(*, return_pct: float, p_value: float) -> FoldResult:
    bars = _bars(3)
    return FoldResult(
        fold_id="holdout",
        train_start_utc="train-start",
        train_end_utc="train-end",
        test_start_utc=bars[0].timestamp_utc,
        test_end_utc=bars[-1].timestamp_utc,
        metrics={
            "return_pct": return_pct,
            "p_value_vs_random_entry": p_value,
        },
        baseline_metrics={},
        baseline_deltas={},
        warnings=tuple(),
        backtest=BacktestResult(
            bars=bars,
            trades=tuple(),
            equity_curve=(100_000.0, 100_010.0, 100_020.0),
            initial_cash=100_000.0,
            final_cash=100_020.0,
        ),
    )


def test_split_research_and_holdout_reserves_recent_window() -> None:
    runner = EvaluationRunner.__new__(EvaluationRunner)
    data_slice = _data_slice(_daily_bars(130))

    research_slice, holdout_bars, holdout_snapshot_id = runner._split_research_and_holdout(
        data_slice,
        embargo_bars=1,
    )

    assert holdout_bars
    assert holdout_snapshot_id
    assert research_slice.bars[-1].timestamp_ms < holdout_bars[0].timestamp_ms
    assert holdout_bars[0].dt_local.date().isoformat() == "2026-02-10"
    assert research_slice.bars[-1].dt_local.date().isoformat() == "2026-02-08"


def test_locked_holdout_fails_closed_when_slice_is_too_short() -> None:
    runner = EvaluationRunner.__new__(EvaluationRunner)
    data_slice = _data_slice(_daily_bars(10))

    with pytest.raises(RuntimeError, match="Not enough pre-holdout bars"):
        runner._split_research_and_holdout(data_slice, embargo_bars=1)


def test_holdout_runs_only_for_promoted_results(monkeypatch) -> None:
    runner = EvaluationRunner.__new__(EvaluationRunner)
    runner.registry = REGISTRY
    preview = _preview()
    metrics = {
        "return_pct": 1.0,
        "annualized_sharpe": 1.0,
        "sharpe_like": 0.5,
        "trade_count": 10.0,
        "delta_buy_and_hold_return_pct": 1.0,
        "delta_exposure_adjusted_buy_and_hold_pct": 1.0,
    }

    monkeypatch.setattr(runner, "_evaluate_preview_folds", lambda spec, preview: (tuple(), metrics))
    monkeypatch.setattr("trader.evaluation.runner.promotion_stage", lambda aggregate, checks: "candidate")

    promoted = runner.evaluate_preview(preview, include_robustness=False)
    assert promoted.holdout_result is not None
    assert promoted.holdout_result.fold_id == "holdout"

    monkeypatch.setattr("trader.evaluation.runner.promotion_stage", lambda aggregate, checks: "exploratory")
    exploratory = runner.evaluate_preview(preview, include_robustness=False)
    assert exploratory.holdout_result is None


def test_candidate_promotion_fails_closed_without_holdout_bars(monkeypatch) -> None:
    runner = EvaluationRunner.__new__(EvaluationRunner)
    runner.registry = REGISTRY
    preview = _preview()
    preview = EvaluationPreview(
        spec=preview.spec,
        data_slice=preview.data_slice,
        split_plan_id=preview.split_plan_id,
        folds=preview.folds,
        required_history=preview.required_history,
        cost_model_id=preview.cost_model_id,
        evaluation_key=preview.evaluation_key,
        holdout_bars=tuple(),
        holdout_snapshot_id=None,
    )
    metrics = {
        "return_pct": 1.0,
        "annualized_sharpe": 1.0,
        "sharpe_like": 0.5,
        "trade_count": 10.0,
        "delta_buy_and_hold_return_pct": 1.0,
        "delta_exposure_adjusted_buy_and_hold_pct": 1.0,
    }

    monkeypatch.setattr(runner, "_evaluate_preview_folds", lambda spec, preview: (tuple(), metrics))
    monkeypatch.setattr(runner, "_aggregate_fold_results", lambda fold_results: metrics)
    monkeypatch.setattr("trader.evaluation.runner.promotion_stage", lambda aggregate, checks: "candidate")
    monkeypatch.setattr(runner, "_evaluate_holdout", lambda spec, preview: pytest.fail("holdout should not run"))

    result = runner.evaluate_preview(preview, include_robustness=False)

    assert result.holdout_result is None
    assert result.promotion_stage == "exploratory"
    assert result.robustness_checks["holdout_p_value_pass"] is False
    assert result.robustness_checks["holdout_directional_match"] is False


@pytest.mark.parametrize(
    ("holdout_return_pct", "holdout_p_value", "expected_stage", "expected_p_value_pass", "expected_directional_match"),
    (
        (0.5, 0.09, "candidate", True, True),
        (0.5, 0.10, "exploratory", False, True),
        (-0.1, 0.09, "exploratory", True, False),
    ),
)
def test_holdout_p_value_and_directional_match_gate_candidate_promotion(
    monkeypatch,
    holdout_return_pct: float,
    holdout_p_value: float,
    expected_stage: str,
    expected_p_value_pass: bool,
    expected_directional_match: bool,
) -> None:
    runner = EvaluationRunner.__new__(EvaluationRunner)
    runner.registry = REGISTRY
    preview = _preview()
    research_metrics = {
        "return_pct": 1.0,
        "annualized_sharpe": 1.0,
        "sharpe_like": 0.5,
        "trade_count": 10.0,
        "delta_buy_and_hold_return_pct": 1.0,
        "delta_exposure_adjusted_buy_and_hold_pct": 1.0,
    }

    monkeypatch.setattr(runner, "_evaluate_preview_folds", lambda spec, preview: (tuple(), research_metrics))
    monkeypatch.setattr(runner, "_aggregate_fold_results", lambda fold_results: research_metrics)
    monkeypatch.setattr("trader.evaluation.runner.promotion_stage", lambda aggregate, checks: "candidate")
    monkeypatch.setattr(
        runner,
        "_evaluate_holdout",
        lambda spec, preview: _holdout_result(return_pct=holdout_return_pct, p_value=holdout_p_value),
    )

    result = runner.evaluate_preview(preview, include_robustness=False)

    assert result.holdout_result is not None
    assert result.promotion_stage == expected_stage
    assert result.robustness_checks["holdout_p_value_pass"] is expected_p_value_pass
    assert result.robustness_checks["holdout_directional_match"] is expected_directional_match
