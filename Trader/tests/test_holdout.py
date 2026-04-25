from __future__ import annotations

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from trader.data.models import MarketBar
from trader.data.view import DataSlice
from trader.evaluation.runner import EvaluationPreview, EvaluationRunner
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
    }

    monkeypatch.setattr(runner, "_evaluate_preview_folds", lambda spec, preview: (tuple(), metrics))
    monkeypatch.setattr("trader.evaluation.runner.promotion_stage", lambda aggregate, checks: "candidate")

    promoted = runner.evaluate_preview(preview, include_robustness=False)
    assert promoted.holdout_result is not None
    assert promoted.holdout_result.fold_id == "holdout"

    monkeypatch.setattr("trader.evaluation.runner.promotion_stage", lambda aggregate, checks: "exploratory")
    exploratory = runner.evaluate_preview(preview, include_robustness=False)
    assert exploratory.holdout_result is None
