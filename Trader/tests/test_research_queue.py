from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from trader.data.models import AggregateBar
from trader.data.storage import SQLiteBarStore
from trader.data.view import DataView
from trader.evaluation.runner import EvaluationRunner
from trader.ledger.store import LedgerStore
from trader.research.candidates import DeterministicCandidateQueue
from trader.research.planner import PlannedSpec
from trader.strategies.registry import REGISTRY
from trader.strategies.spec import FilterSpec, SignalSpec, StrategySpec


NEW_YORK = ZoneInfo("America/New_York")


def _seed_db(path: Path, market_days: int = 5) -> None:
    store = SQLiteBarStore(path)
    bars = []
    start = datetime(2026, 1, 5, 9, 30, tzinfo=NEW_YORK)
    index = 0
    for day in range(market_days):
        session_start = start + timedelta(days=day)
        for minute in range(390):
            timestamp = (session_start + timedelta(minutes=minute)).astimezone(timezone.utc)
            close = 100.0 + (index * 0.1)
            bars.append(
                AggregateBar(
                    ticker="SPY",
                    multiplier=1,
                    timespan="minute",
                    timestamp_ms=int(timestamp.timestamp() * 1000),
                    open=close,
                    high=close + 0.25,
                    low=close - 0.25,
                    close=close,
                    volume=1_000.0,
                    vwap=close,
                    transactions=10,
                )
            )
            index += 1
    store.upsert_bars(bars)


def _spec(name: str, signal_name: str, params: dict[str, int | float]) -> StrategySpec:
    return StrategySpec(
        name=name,
        signal=SignalSpec(signal_name, params),
        filters=(FilterSpec("session", {"session": "regular"}),),
    )


def test_candidate_queue_skips_existing_evaluation_keys(tmp_path: Path) -> None:
    db_path = tmp_path / "market.db"
    _seed_db(db_path)
    runner = EvaluationRunner(DataView(db_path), REGISTRY)
    ledger = LedgerStore(tmp_path / "ledger.db")
    ledger.initialize()

    existing_preview = runner.preview_walk_forward(
        _spec("ema_existing", "ema_cross", {"fast_length": 20, "slow_length": 80, "signal_buffer_bps": 0.0}),
        num_folds=3,
        embargo_bars=1,
    )
    existing_result = runner.evaluate_preview(existing_preview)
    ledger.record_result(existing_result, artifact_paths={}, generator_kind="grid")

    queue = DeterministicCandidateQueue(
        history_entries=ledger.list_completed(limit=100),
        frontier_entries=ledger.top_experiments(limit=10),
    )
    queue_result = queue.build(
        planned_specs=(
            PlannedSpec(existing_preview.spec, generator_kind="grid"),
            PlannedSpec(
                _spec("ema_new", "ema_cross", {"fast_length": 12, "slow_length": 55, "signal_buffer_bps": 0.0}),
                generator_kind="grid",
            ),
        ),
        runner=runner,
        num_folds=3,
        embargo_bars=1,
    )

    assert queue_result.duplicate_count == 1
    assert len(queue_result.selected) == 1
    assert queue_result.selected[0].preview.spec.name == "ema_new"


def test_candidate_queue_prefers_underexplored_family_when_scores_are_close(tmp_path: Path) -> None:
    db_path = tmp_path / "market.db"
    _seed_db(db_path)
    runner = EvaluationRunner(DataView(db_path), REGISTRY)
    ledger = LedgerStore(tmp_path / "ledger.db")
    ledger.initialize()

    # Seed history with multiple EMA runs so breakout receives an exploration boost.
    for index in range(3):
        preview = runner.preview_walk_forward(
            _spec(
                f"ema_seed_{index}",
                "ema_cross",
                {"fast_length": 8 + (index * 2), "slow_length": 55, "signal_buffer_bps": 0.0},
            ),
            num_folds=3,
            embargo_bars=1,
        )
        ledger.record_result(runner.evaluate_preview(preview), artifact_paths={}, generator_kind="grid")

    queue = DeterministicCandidateQueue(
        history_entries=ledger.list_completed(limit=100),
        frontier_entries=ledger.top_experiments(limit=10),
    )
    queue_result = queue.build(
        planned_specs=(
            PlannedSpec(
                _spec("ema_candidate", "ema_cross", {"fast_length": 20, "slow_length": 80, "signal_buffer_bps": 0.0}),
                generator_kind="grid",
            ),
            PlannedSpec(
                _spec("breakout_candidate", "breakout", {"entry_window": 20, "exit_window": 10, "buffer_bps": 0.0}),
                generator_kind="grid",
            ),
        ),
        runner=runner,
        num_folds=3,
        embargo_bars=1,
    )

    assert len(queue_result.selected) == 2
    assert queue_result.selected[0].family == "breakout"
