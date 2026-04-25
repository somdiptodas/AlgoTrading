from __future__ import annotations

from dataclasses import replace

from pytest import approx

from trader.evaluation.runner import ExperimentResult
from trader.ledger.entry import LedgerEntry
from trader.ledger.store import LedgerStore
from trader.research.decay import (
    DECAY_MONITOR_GENERATOR_KIND,
    build_decay_report,
    reevaluate_promoted_specs,
)
from trader.strategies.spec import SignalSpec, StrategySpec


def _result(
    *,
    experiment_id: str,
    spec: StrategySpec,
    snapshot: str,
    stage: str,
    return_pct: float,
    sharpe: float,
    drawdown: float,
) -> ExperimentResult:
    return ExperimentResult(
        experiment_id=experiment_id,
        status="completed",
        spec=spec,
        spec_hash=spec.spec_hash(),
        data_snapshot_id=snapshot,
        split_plan_id=f"split_{snapshot}",
        cost_model_id=spec.exec_config.cost_model_id(),
        aggregate_metrics={
            "return_pct": return_pct,
            "annualized_sharpe": sharpe,
            "sharpe_like": sharpe,
            "max_drawdown_pct": drawdown,
            "trade_count": 25.0,
            "delta_buy_and_hold_return_pct": return_pct,
        },
        fold_results=tuple(),
        robustness_checks={"fold_consistency_pass": True, "neighborhood_pass": True},
        promotion_stage=stage,
    )


def _entry(
    *,
    experiment_id: str,
    spec: StrategySpec,
    snapshot: str,
    stage: str,
    return_pct: float,
    sharpe: float,
    drawdown: float,
    completed_at: str,
) -> LedgerEntry:
    result = _result(
        experiment_id=experiment_id,
        spec=spec,
        snapshot=snapshot,
        stage=stage,
        return_pct=return_pct,
        sharpe=sharpe,
        drawdown=drawdown,
    )
    return LedgerEntry.from_result(
        result,
        evaluation_key=f"key_{experiment_id}",
        artifact_paths={},
        generator_kind="grid",
        completed_at_utc=completed_at,
    )


class _FakeRunner:
    def __init__(self, result: ExperimentResult, *, existing_key: str | None = None) -> None:
        self.result = result
        self.existing_key = existing_key
        self.calls: list[StrategySpec] = []

    def evaluation_key_for_spec(self, spec: StrategySpec, **_: object) -> str:
        return self.existing_key or f"key_{spec.spec_hash()}_{self.result.data_snapshot_id}"

    def evaluate_walk_forward(self, spec: StrategySpec, **_: object) -> ExperimentResult:
        self.calls.append(spec)
        return self.result


def test_decay_report_links_snapshots_and_flags_demoted_strategy() -> None:
    spec = StrategySpec(
        name="ema_decay",
        signal=SignalSpec("ema_cross", {"fast_length": 20, "slow_length": 80, "signal_buffer_bps": 0.0}),
    )
    source = _entry(
        experiment_id="source",
        spec=spec,
        snapshot="snapshot_a",
        stage="candidate",
        return_pct=3.0,
        sharpe=1.2,
        drawdown=1.0,
        completed_at="2026-04-20T12:00:00+00:00",
    )
    latest = _entry(
        experiment_id="latest",
        spec=spec,
        snapshot="snapshot_b",
        stage="exploratory",
        return_pct=0.5,
        sharpe=0.4,
        drawdown=4.0,
        completed_at="2026-04-25T12:00:00+00:00",
    )

    report = build_decay_report((source, latest))

    assert len(report) == 1
    item = report[0]
    assert item.source_experiment_id == "source"
    assert item.latest_experiment_id == "latest"
    assert item.source_data_snapshot_id == "snapshot_a"
    assert item.latest_data_snapshot_id == "snapshot_b"
    assert item.evaluation_count == 2
    assert item.return_delta_pct == -2.5
    assert item.sharpe_delta == approx(-0.8)
    assert item.drawdown_delta_pct == 3.0
    assert item.rolling_return_delta_pct == -2.5
    assert item.rolling_sharpe_delta == approx(-0.8)
    assert item.rolling_drawdown_delta_pct == 3.0
    assert item.rolling_degradation_count == 1
    assert item.status == "demoted"


def test_decay_report_marks_missing_current_snapshot_as_pending_recheck() -> None:
    spec = StrategySpec(
        name="ema_pending",
        signal=SignalSpec("ema_cross", {"fast_length": 12, "slow_length": 55, "signal_buffer_bps": 0.0}),
    )
    source = _entry(
        experiment_id="source",
        spec=spec,
        snapshot="old_snapshot",
        stage="research_frontier",
        return_pct=2.0,
        sharpe=1.0,
        drawdown=1.0,
        completed_at="2026-04-20T12:00:00+00:00",
    )

    report = build_decay_report((source,), current_snapshot_id="new_snapshot")

    assert report[0].status == "pending_recheck"


def test_reevaluate_promoted_specs_records_new_snapshot_with_parent_link(tmp_path) -> None:
    spec = StrategySpec(
        name="ema_recheck",
        signal=SignalSpec("ema_cross", {"fast_length": 8, "slow_length": 34, "signal_buffer_bps": 0.0}),
    )
    old_result = _result(
        experiment_id="old_exp",
        spec=spec,
        snapshot="snapshot_a",
        stage="candidate",
        return_pct=3.0,
        sharpe=1.2,
        drawdown=1.0,
    )
    new_result = replace(
        _result(
            experiment_id="new_exp",
            spec=spec,
            snapshot="snapshot_b",
            stage="research_frontier",
            return_pct=2.0,
            sharpe=0.9,
            drawdown=1.5,
        ),
        spec_hash=old_result.spec_hash,
    )
    ledger = LedgerStore(tmp_path / "ledger.db")
    ledger.initialize()
    ledger.record_result(old_result, artifact_paths={}, generator_kind="grid")

    recorded = reevaluate_promoted_specs(ledger, _FakeRunner(new_result), limit=10)

    assert len(recorded) == 1
    assert recorded[0].experiment_id == "new_exp"
    assert recorded[0].generator_kind == DECAY_MONITOR_GENERATOR_KIND
    assert recorded[0].parent_experiment_ids == ("old_exp",)
    assert recorded[0].data_snapshot_id == "snapshot_b"


def test_reevaluate_promoted_specs_skips_existing_current_evaluation(tmp_path) -> None:
    spec = StrategySpec(
        name="ema_existing_current",
        signal=SignalSpec("ema_cross", {"fast_length": 8, "slow_length": 34, "signal_buffer_bps": 0.0}),
    )
    result = _result(
        experiment_id="old_exp",
        spec=spec,
        snapshot="snapshot_a",
        stage="candidate",
        return_pct=3.0,
        sharpe=1.2,
        drawdown=1.0,
    )
    ledger = LedgerStore(tmp_path / "ledger.db")
    ledger.initialize()
    existing = ledger.record_result(result, artifact_paths={}, generator_kind="grid")
    runner = _FakeRunner(result, existing_key=existing.evaluation_key)

    recorded = reevaluate_promoted_specs(ledger, runner, limit=10)

    assert recorded == tuple()
    assert runner.calls == []


def test_decay_report_tracks_rolling_degradation_sequence() -> None:
    spec = StrategySpec(
        name="ema_rolling",
        signal=SignalSpec("ema_cross", {"fast_length": 20, "slow_length": 80, "signal_buffer_bps": 0.0}),
    )
    source = _entry(
        experiment_id="source",
        spec=spec,
        snapshot="snapshot_a",
        stage="candidate",
        return_pct=4.0,
        sharpe=1.4,
        drawdown=1.0,
        completed_at="2026-04-20T12:00:00+00:00",
    )
    middle = _entry(
        experiment_id="middle",
        spec=spec,
        snapshot="snapshot_b",
        stage="candidate",
        return_pct=2.5,
        sharpe=1.0,
        drawdown=2.2,
        completed_at="2026-04-22T12:00:00+00:00",
    )
    latest = _entry(
        experiment_id="latest",
        spec=spec,
        snapshot="snapshot_c",
        stage="research_frontier",
        return_pct=1.0,
        sharpe=0.6,
        drawdown=3.3,
        completed_at="2026-04-25T12:00:00+00:00",
    )

    report = build_decay_report((source, middle, latest))

    assert report[0].latest_experiment_id == "latest"
    assert report[0].rolling_return_delta_pct == -1.5
    assert report[0].rolling_sharpe_delta == approx(-0.4)
    assert report[0].rolling_drawdown_delta_pct == approx(1.1)
    assert report[0].rolling_degradation_count == 2
    assert report[0].status == "degrading"
