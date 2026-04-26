from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from time import perf_counter
from typing import Callable, Sequence

from trader.artifacts.store import ArtifactStore
from trader.config import Settings, load_settings
from trader.data.view import DataView
from trader.evaluation.runner import DEFAULT_LOCKED_HOLDOUT_MONTHS, EvaluationPreview, EvaluationRunner, ExperimentResult
from trader.ledger.entry import json_dumps
from trader.ledger.entry import LedgerEntry
from trader.ledger.store import LedgerStore
from trader.reporting.report import render_experiment_report
from trader.reporting.run_dashboard import LoopRunOutputs, ReportPathConventions, write_loop_run_outputs
from trader.research.candidates import CandidateQueueResult, DeterministicCandidateQueue, ScoredCandidate
from trader.research.critic import HeuristicCritic
from trader.research.critic_memory import CriticRegionMemory
from trader.research.frontier import FrontierManager
from trader.research.generator import StrategyGenerator
from trader.research.planner import DeterministicPlanner, strategy_shape_key
from trader.research.suppressor import RegionSuppressor, SuppressedSpec, WithinRunStageAFailureMemory
from trader.strategies.registry import REGISTRY
from trader.strategies.spec import StrategySpec

_DEFAULT_SIGNAL_FAMILIES = ("multi_signal",)
_ALL_SIGNAL_FAMILIES = ("multi_signal", "ema_cross", "breakout", "rsi_reversion", "composite", "vwap_deviation")
DEFAULT_OVERPLAN_FACTOR = 4
DEFAULT_PREVIEW_FACTOR = 4
MIN_PLANNED_SPECS = 64
MAX_PLANNER_RESTARTS = 2
PLANNER_RESTART_DUPLICATE_RATE = 0.75
TIMING_PHASES = (
    "planning",
    "key_compute",
    "preview",
    "queue_scoring",
    "stage_a",
    "stage_b",
    "robustness_neighbors",
    "artifact_write",
    "ledger_write",
)
SEARCH_EXHAUSTION_SELECTED_FRACTION = 0.5
STAGE_A_PARALLEL_CHUNK_SIZE = 2


@dataclass(frozen=True)
class StageAPrescreenResult:
    stage_b_candidates: tuple[ScoredCandidate, ...]
    completed_results: tuple[tuple[ScoredCandidate, ExperimentResult], ...]
    suppression_records: tuple[SuppressedSpec, ...]
    timings: dict[str, float]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the deterministic autonomous research loop")
    parser.add_argument("--database")
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument(
        "--overplan-factor",
        type=int,
        default=DEFAULT_OVERPLAN_FACTOR,
        help=f"Plan this many specs per requested batch slot before preview selection (default: {DEFAULT_OVERPLAN_FACTOR})",
    )
    parser.add_argument(
        "--preview-factor",
        type=int,
        default=DEFAULT_PREVIEW_FACTOR,
        help=f"Preview at most this many specs per requested batch slot (default: {DEFAULT_PREVIEW_FACTOR})",
    )
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--embargo-bars", type=int, default=1)
    parser.add_argument("--holdout-months", type=int, default=DEFAULT_LOCKED_HOLDOUT_MONTHS)
    parser.add_argument("--frontier-limit", type=int, default=5)
    parser.add_argument(
        "--signal-family",
        action="append",
        choices=_ALL_SIGNAL_FAMILIES,
        help="Restrict the deterministic loop to one or more signal families",
    )
    parser.add_argument(
        "--suppressor-radius",
        type=float,
        default=0.15,
        help="Normalized parameter-space radius for region suppression (default: 0.15)",
    )
    parser.add_argument(
        "--suppressor-weight-cap",
        type=float,
        default=40.0,
        help="Max suppression penalty per candidate (default: 40.0)",
    )
    return parser


def _loop_run_id(args: argparse.Namespace) -> str:
    """Stable identifier for this loop invocation, used in the suppression audit log."""
    payload = "|".join([
        datetime.now(timezone.utc).isoformat(),
        str(args.batch_size),
        str(args.folds),
        str(sorted(args.signal_family or [])),
    ])
    return sha256(payload.encode()).hexdigest()[:16]


def _planned_spec_count(batch_size: int, overplan_factor: int) -> int:
    return max(batch_size * overplan_factor, MIN_PLANNED_SPECS)


def _max_preview_count(batch_size: int, preview_factor: int) -> int:
    return max(batch_size, batch_size * preview_factor)


def _should_restart_planner(queue_result: CandidateQueueResult, planned_count: int) -> bool:
    if planned_count <= 0:
        return False
    if queue_result.previewed_count == 0 and queue_result.duplicate_count > 0:
        return True
    return (queue_result.duplicate_count / planned_count) >= PLANNER_RESTART_DUPLICATE_RATE


def _suppression_audit_types(
    records: tuple[SuppressedSpec, ...],
    evaluated_spec_hashes: set[str],
    *,
    unevaluated_type: str = "preview_discarded",
) -> dict[str, str]:
    return {
        record.spec_hash: "evaluated" if record.spec_hash in evaluated_spec_hashes else unevaluated_type
        for record in records
    }


def _new_timings() -> dict[str, float]:
    return {phase: 0.0 for phase in TIMING_PHASES}


def _add_timing(timings: dict[str, float], phase: str, started: float) -> None:
    timings[phase] = timings.get(phase, 0.0) + (perf_counter() - started)


def _timing_payload(timings: dict[str, float]) -> dict[str, float]:
    return {phase: round(timings.get(phase, 0.0), 6) for phase in TIMING_PHASES}


def _load_or_seed_critic_memory(path: Path, history_entries: Sequence[LedgerEntry]) -> CriticRegionMemory:
    if path.exists():
        return CriticRegionMemory.load(path, registry=REGISTRY)
    memory = CriticRegionMemory.from_entries(history_entries, registry=REGISTRY)
    memory.write(path)
    return memory


def _active_data_snapshot_id(
    runner: EvaluationRunner,
    *,
    num_folds: int,
    embargo_bars: int,
    locked_holdout_months: int | None,
) -> str:
    return runner.preview_walk_forward(
        StrategySpec(name="snapshot_probe"),
        num_folds=num_folds,
        embargo_bars=embargo_bars,
        locked_holdout_months=locked_holdout_months,
    ).data_slice.snapshot_id


def _current_snapshot_entries(
    entries: Sequence[LedgerEntry],
    *,
    active_data_snapshot_id: str,
) -> tuple[LedgerEntry, ...]:
    return tuple(entry for entry in entries if entry.data_snapshot_id == active_data_snapshot_id)


def _stage_b_worker_count(candidate_count: int) -> int:
    if candidate_count <= 0:
        return 0
    return min(8, candidate_count)


def _evaluate_candidate_worker(payload: tuple[str, EvaluationPreview]) -> tuple[ExperimentResult, dict[str, float]]:
    database_path, preview = payload
    runner = EvaluationRunner(DataView(Path(database_path)), REGISTRY)
    result = runner.evaluate_preview(preview, include_robustness=True, run_stage_a=False)
    return result, dict(runner.phase_timings)


def _evaluate_stage_a_candidate_worker(payload: tuple[str, EvaluationPreview]) -> tuple[ExperimentResult | None, dict[str, float]]:
    database_path, preview = payload
    runner = EvaluationRunner(DataView(Path(database_path)), REGISTRY)
    started = perf_counter()
    result = runner.evaluate_stage_a_preview(preview)
    return result, {"stage_a": perf_counter() - started}


def _evaluate_selected_candidates(
    selected_candidates: Sequence[ScoredCandidate],
    database_path: str,
    *,
    executor_factory: Callable[..., object] = ProcessPoolExecutor,
) -> tuple[tuple[ExperimentResult, ...], dict[str, float]]:
    worker_count = _stage_b_worker_count(len(selected_candidates))
    if worker_count == 0:
        return tuple(), {}
    payloads = tuple((database_path, candidate.preview) for candidate in selected_candidates)
    if worker_count == 1:
        outputs = (_evaluate_candidate_worker(payloads[0]),)
    else:
        with executor_factory(max_workers=worker_count) as executor:
            outputs = tuple(executor.map(_evaluate_candidate_worker, payloads))
    timings: dict[str, float] = {}
    for _, worker_timings in outputs:
        for phase in ("stage_a", "stage_b", "robustness_neighbors"):
            timings[phase] = timings.get(phase, 0.0) + worker_timings.get(phase, 0.0)
    return tuple(result for result, _ in outputs), timings


def _mark_stage_a_passed(result: ExperimentResult) -> ExperimentResult:
    return replace(
        result,
        aggregate_metrics={
            **result.aggregate_metrics,
            "stage_a_pass": 1.0,
        },
    )


def _loop_experiment_summary(
    result: ExperimentResult,
    *,
    generator_kind: str,
    artifact_paths: dict[str, str],
) -> dict[str, object]:
    return {
        "experiment_id": result.experiment_id,
        "family": result.spec.signal.name,
        "promotion_stage": result.promotion_stage,
        "generator_kind": generator_kind,
        "shape_key": strategy_shape_key(result.spec),
        "aggregate_metrics": dict(result.aggregate_metrics),
        "artifact_paths": dict(artifact_paths),
    }


def _write_loop_outputs(loop_payload: dict[str, object], settings: Settings) -> LoopRunOutputs:
    paths = ReportPathConventions(reports_dir=settings.reports_dir, artifacts_dir=settings.artifacts_dir)
    return write_loop_run_outputs(loop_payload, paths)


def _prescreen_stage_a_candidates(
    selected_candidates: Sequence[ScoredCandidate],
    runner: EvaluationRunner,
    *,
    suppressor_radius: float,
    database_path: str | None = None,
    executor_factory: Callable[..., object] = ProcessPoolExecutor,
) -> StageAPrescreenResult:
    memory = WithinRunStageAFailureMemory(radius=suppressor_radius)
    stage_b_candidates: list[ScoredCandidate] = []
    completed_results: list[tuple[ScoredCandidate, ExperimentResult]] = []
    suppression_records: list[SuppressedSpec] = []
    timings = {"stage_a": 0.0}
    candidate_index = 0
    while candidate_index < len(selected_candidates):
        evaluation_batch: list[ScoredCandidate] = []
        while candidate_index < len(selected_candidates) and len(evaluation_batch) < STAGE_A_PARALLEL_CHUNK_SIZE:
            candidate = selected_candidates[candidate_index]
            candidate_index += 1
            suppression_record = memory.assess(candidate.preview.spec)
            if suppression_record is not None:
                suppression_records.append(suppression_record)
                continue
            evaluation_batch.append(candidate)
            if memory.would_reach_failure_threshold_if_failed(candidate.preview.spec):
                break
        if not evaluation_batch:
            continue
        outputs: tuple[tuple[ExperimentResult | None, dict[str, float]], ...]
        if database_path is not None and len(evaluation_batch) > 1:
            payloads = tuple((database_path, candidate.preview) for candidate in evaluation_batch)
            with executor_factory(max_workers=len(payloads)) as executor:
                outputs = tuple(executor.map(_evaluate_stage_a_candidate_worker, payloads))
        else:
            batch_outputs = []
            for candidate in evaluation_batch:
                started = perf_counter()
                stage_a_result = runner.evaluate_stage_a_preview(candidate.preview)
                batch_outputs.append((stage_a_result, {"stage_a": perf_counter() - started}))
            outputs = tuple(batch_outputs)
        for candidate, (stage_a_result, worker_timings) in zip(evaluation_batch, outputs):
            timings["stage_a"] += worker_timings.get("stage_a", 0.0)
            if stage_a_result is not None:
                completed_results.append((candidate, stage_a_result))
                memory.record(stage_a_result)
            else:
                stage_b_candidates.append(candidate)
    return StageAPrescreenResult(
        stage_b_candidates=tuple(stage_b_candidates),
        completed_results=tuple(completed_results),
        suppression_records=tuple(suppression_records),
        timings=timings,
    )


def _search_exhaustion_status(queue_result: CandidateQueueResult, *, batch_size: int) -> dict[str, object]:
    selected_count = len(queue_result.selected)
    if selected_count == 0 and queue_result.duplicate_count > 0:
        return {
            "exhausted": True,
            "reason": "no_unique_candidates",
        }
    if selected_count < batch_size and selected_count <= max(1, int(batch_size * SEARCH_EXHAUSTION_SELECTED_FRACTION)):
        return {
            "exhausted": True,
            "reason": "low_unique_candidate_yield",
        }
    return {
        "exhausted": False,
        "reason": "",
    }


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    timings = _new_timings()
    settings = load_settings(database_path=args.database)
    ledger = LedgerStore(settings.ledger_path)
    ledger.initialize()
    artifacts = ArtifactStore(settings.artifacts_dir, settings.reports_dir)
    runner = EvaluationRunner(DataView(settings.database_path), REGISTRY)
    planner = DeterministicPlanner(REGISTRY)
    generator = StrategyGenerator(REGISTRY)
    frontier_manager = FrontierManager(limit=args.frontier_limit)
    critic = HeuristicCritic()
    loop_run_id = _loop_run_id(args)

    all_history_entries = ledger.list_completed(limit=10_000)
    active_data_snapshot_id = _active_data_snapshot_id(
        runner,
        num_folds=args.folds,
        embargo_bars=args.embargo_bars,
        locked_holdout_months=args.holdout_months,
    )
    history_entries = _current_snapshot_entries(
        all_history_entries,
        active_data_snapshot_id=active_data_snapshot_id,
    )
    critic_memory_path = settings.research_dir / f"critic_memory_{active_data_snapshot_id[:12]}.json"
    critic_memory = _load_or_seed_critic_memory(critic_memory_path, history_entries)
    frontier_entries = ledger.query.top_experiments(history_entries, limit=args.frontier_limit)

    # --- Build suppressor from history entries that failed robustness gates ---
    suppressor = RegionSuppressor(
        history_entries,
        registry=REGISTRY,
        radius=args.suppressor_radius,
        weight_cap=args.suppressor_weight_cap,
    )

    signal_families = tuple(args.signal_family) if args.signal_family else _DEFAULT_SIGNAL_FAMILIES
    frontier_specs = tuple((entry.experiment_id, entry.spec) for entry in frontier_entries)
    candidate_queue = DeterministicCandidateQueue(
        history_entries=history_entries,
        frontier_entries=frontier_entries,
        critic_memory=critic_memory,
        suppressor=suppressor,
    )
    planned = tuple()
    generated = None
    queue_result = None
    planner_restarts = 0
    for restart_index in range(MAX_PLANNER_RESTARTS + 1):
        started = perf_counter()
        planned = planner.plan(
            batch_size=_planned_spec_count(args.batch_size, args.overplan_factor),
            frontier_specs=frontier_specs,
            allowed_signal_families=signal_families,
            history_entries=history_entries,
            optuna_dir=settings.research_dir / "optuna" / active_data_snapshot_id[:12],
            restart_seed=loop_run_id,
            restart_index=restart_index,
        )
        generated = generator.validate_and_filter(
            planned,
            seen_evaluation_key=lambda spec: False,
            evaluation_key_for_spec=lambda spec: spec.spec_hash(),
        )
        _add_timing(timings, "planning", started)
        queue_result = candidate_queue.build(
            planned_specs=generated.accepted,
            runner=runner,
            num_folds=args.folds,
            embargo_bars=args.embargo_bars,
            locked_holdout_months=args.holdout_months,
            max_preview_count=_max_preview_count(args.batch_size, args.preview_factor),
            timings=timings,
        )
        if (
            "multi_signal" not in signal_families
            or restart_index >= MAX_PLANNER_RESTARTS
            or not _should_restart_planner(queue_result, len(planned))
        ):
            break
        planner_restarts += 1

    if generated is None or queue_result is None:
        raise RuntimeError("planner did not produce a candidate queue")

    completed = []
    completed_experiments = []
    reused = queue_result.duplicate_count
    selected_candidates = queue_result.selected[: args.batch_size]
    search_exhaustion = _search_exhaustion_status(queue_result, batch_size=args.batch_size)
    prescreen = _prescreen_stage_a_candidates(
        selected_candidates,
        runner,
        suppressor_radius=args.suppressor_radius,
        database_path=str(settings.database_path),
    )
    timings["stage_a"] += prescreen.timings.get("stage_a", 0.0)
    results, worker_timings = _evaluate_selected_candidates(prescreen.stage_b_candidates, str(settings.database_path))
    for phase in ("stage_a", "stage_b", "robustness_neighbors"):
        timings[phase] += worker_timings.get(phase, 0.0)
    result_by_key = {
        candidate.preview.evaluation_key: result
        for candidate, result in prescreen.completed_results
    }
    result_by_key.update(
        {
            candidate.preview.evaluation_key: _mark_stage_a_passed(result)
            for candidate, result in zip(prescreen.stage_b_candidates, results)
        }
    )
    for candidate in selected_candidates:
        result = result_by_key.get(candidate.preview.evaluation_key)
        if result is None:
            continue
        critique = critic.critique(result)
        report_markdown = render_experiment_report(result, critique=critique.to_payload())
        started = perf_counter()
        artifact_paths = artifacts.write_experiment(
            result,
            report_markdown=report_markdown,
            critique=critique.to_payload(),
            generator_kind=candidate.planned.generator_kind,
        )
        _add_timing(timings, "artifact_write", started)
        started = perf_counter()
        ledger.record_result(
            result,
            artifact_paths=artifact_paths,
            generator_kind=candidate.planned.generator_kind,
            parent_experiment_ids=candidate.planned.parent_experiment_ids,
            critique=critique.to_payload(),
        )
        _add_timing(timings, "ledger_write", started)
        completed.append(result)
        completed_experiments.append(
            _loop_experiment_summary(
                result,
                generator_kind=candidate.planned.generator_kind,
                artifact_paths=artifact_paths,
            )
        )

    # --- Persist suppression audit records to the ledger ---
    evaluated_spec_hashes = {result.spec_hash for result in completed}
    suppression_records = queue_result.suppression_records + prescreen.suppression_records
    audit_type_by_spec_hash = {
        **_suppression_audit_types(queue_result.suppression_records, evaluated_spec_hashes),
        **_suppression_audit_types(
            prescreen.suppression_records,
            evaluated_spec_hashes,
            unevaluated_type="stage_a_suppressed",
        ),
    }
    started = perf_counter()
    suppression_logged = ledger.log_suppression_batch(
        loop_run_id,
        suppression_records,
        audit_type_by_spec_hash=audit_type_by_spec_hash,
    )
    _add_timing(timings, "ledger_write", started)
    suppression_summary = ledger.suppression_summary(loop_run_id) if suppression_logged > 0 else {}
    fresh_history_entries = _current_snapshot_entries(
        ledger.list_completed(limit=10_000),
        active_data_snapshot_id=active_data_snapshot_id,
    )
    CriticRegionMemory.from_entries(fresh_history_entries, registry=REGISTRY).write(critic_memory_path)

    frontier = frontier_manager.rank(
        [entry.to_result() for entry in ledger.query.top_experiments(fresh_history_entries, limit=args.frontier_limit)]
    )
    loop_payload = {
        "loop_run_id": loop_run_id,
        "active_data_snapshot_id": active_data_snapshot_id,
        "signal_families": list(signal_families),
        "history": {
            "total_completed": len(all_history_entries),
            "active_completed": len(history_entries),
            "ignored_stale_completed": len(all_history_entries) - len(history_entries),
        },
        "planned": len(planned),
        "accepted": len(queue_result.selected),
        "completed": len(completed),
        "reused": reused,
        "planner_restarts": planner_restarts,
        "search_exhaustion": search_exhaustion,
        "counts": {
            "planned": len(planned),
            "previewed": queue_result.previewed_count,
            "selected": len(queue_result.selected),
            "evaluated": len(completed),
            "duplicate": queue_result.duplicate_count,
            "suppressed": suppression_logged,
            "suppressed_evaluated": sum(1 for value in audit_type_by_spec_hash.values() if value == "evaluated"),
            "suppressed_preview_discarded": sum(
                1 for value in audit_type_by_spec_hash.values() if value == "preview_discarded"
            ),
            "suppressed_stage_a_suppressed": sum(
                1 for value in audit_type_by_spec_hash.values() if value == "stage_a_suppressed"
            ),
        },
        "experiments": completed_experiments,
        "rejected": list(generated.rejected),
        "timings_sec": _timing_payload(timings),
        "suppressor": {
            **suppressor.summary(),
            "candidates_suppressed": suppression_logged,
            "by_family": suppression_summary.get("by_family", []),
            "by_type": suppression_summary.get("by_type", []),
        },
        "frontier": [
            {
                "experiment_id": item.experiment_id,
                "family": item.family,
                "score_vector": item.score_vector,
                "promotion_stage": item.promotion_stage,
            }
            for item in frontier
        ],
    }
    outputs = _write_loop_outputs(loop_payload, settings)
    print(
        json_dumps(
            {
                **loop_payload,
                "reporting": {
                    "loop_json": str(outputs.loop_json_path.resolve()),
                    "run_report": str(outputs.run_report_path.resolve()),
                    "dashboard": str(outputs.dashboard_path.resolve()),
                    "trade_reports": [str(path.resolve()) for path in outputs.trade_reports],
                },
            },
            pretty=True,
        )
    )
