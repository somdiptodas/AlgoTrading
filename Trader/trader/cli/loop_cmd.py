from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from time import perf_counter
from typing import Callable, Sequence

from trader.artifacts.store import ArtifactStore
from trader.config import load_settings
from trader.data.view import DataView
from trader.evaluation.runner import DEFAULT_LOCKED_HOLDOUT_MONTHS, EvaluationPreview, EvaluationRunner, ExperimentResult
from trader.ledger.entry import json_dumps
from trader.ledger.entry import LedgerEntry
from trader.ledger.store import LedgerStore
from trader.reporting.report import render_experiment_report
from trader.research.candidates import DeterministicCandidateQueue, ScoredCandidate
from trader.research.critic import HeuristicCritic
from trader.research.critic_memory import CriticRegionMemory
from trader.research.frontier import FrontierManager
from trader.research.generator import StrategyGenerator
from trader.research.planner import DeterministicPlanner
from trader.research.suppressor import RegionSuppressor, SuppressedSpec
from trader.strategies.registry import REGISTRY

_ALL_SIGNAL_FAMILIES = ("ema_cross", "breakout", "rsi_reversion", "vwap_deviation", "composite")
DEFAULT_OVERPLAN_FACTOR = 4
DEFAULT_PREVIEW_FACTOR = 4
MIN_PLANNED_SPECS = 64
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


def _suppression_audit_types(
    records: tuple[SuppressedSpec, ...],
    evaluated_spec_hashes: set[str],
) -> dict[str, str]:
    return {
        record.spec_hash: "evaluated" if record.spec_hash in evaluated_spec_hashes else "preview_discarded"
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


def _stage_b_worker_count(candidate_count: int) -> int:
    if candidate_count <= 0:
        return 0
    return min(8, max(4, candidate_count))


def _evaluate_candidate_worker(payload: tuple[str, EvaluationPreview]) -> tuple[ExperimentResult, dict[str, float]]:
    database_path, preview = payload
    runner = EvaluationRunner(DataView(Path(database_path)), REGISTRY)
    result = runner.evaluate_preview(preview, include_robustness=True)
    return result, dict(runner.phase_timings)


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
    with executor_factory(max_workers=worker_count) as executor:
        outputs = tuple(executor.map(_evaluate_candidate_worker, payloads))
    timings: dict[str, float] = {}
    for _, worker_timings in outputs:
        for phase in ("stage_a", "stage_b", "robustness_neighbors"):
            timings[phase] = timings.get(phase, 0.0) + worker_timings.get(phase, 0.0)
    return tuple(result for result, _ in outputs), timings


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

    history_entries = ledger.list_completed(limit=10_000)
    critic_memory_path = settings.research_dir / "critic_memory.json"
    critic_memory = _load_or_seed_critic_memory(critic_memory_path, history_entries)
    frontier_entries = ledger.query.promoted_experiments(history_entries, limit=args.frontier_limit)

    # --- Build suppressor from history entries that failed robustness gates ---
    suppressor = RegionSuppressor(
        history_entries,
        registry=REGISTRY,
        radius=args.suppressor_radius,
        weight_cap=args.suppressor_weight_cap,
    )

    signal_families = tuple(args.signal_family) if args.signal_family else _ALL_SIGNAL_FAMILIES
    frontier_specs = tuple((entry.experiment_id, entry.spec) for entry in frontier_entries)
    started = perf_counter()
    planned = planner.plan(
        batch_size=_planned_spec_count(args.batch_size, args.overplan_factor),
        frontier_specs=frontier_specs,
        allowed_signal_families=signal_families,
        history_entries=history_entries,
        optuna_dir=settings.research_dir / "optuna",
    )
    generated = generator.validate_and_filter(
        planned,
        seen_evaluation_key=lambda spec: False,
        evaluation_key_for_spec=lambda spec: spec.spec_hash(),
    )
    _add_timing(timings, "planning", started)
    candidate_queue = DeterministicCandidateQueue(
        history_entries=history_entries,
        frontier_entries=frontier_entries,
        critic_memory=critic_memory,
        suppressor=suppressor,
    )
    queue_result = candidate_queue.build(
        planned_specs=generated.accepted,
        runner=runner,
        num_folds=args.folds,
        embargo_bars=args.embargo_bars,
        locked_holdout_months=args.holdout_months,
        max_preview_count=_max_preview_count(args.batch_size, args.preview_factor),
        timings=timings,
    )

    completed = []
    reused = queue_result.duplicate_count
    selected_candidates = queue_result.selected[: args.batch_size]
    results, worker_timings = _evaluate_selected_candidates(selected_candidates, str(settings.database_path))
    for phase in ("stage_a", "stage_b", "robustness_neighbors"):
        timings[phase] += worker_timings.get(phase, 0.0)
    for candidate, result in zip(selected_candidates, results):
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

    # --- Persist suppression audit records to the ledger ---
    evaluated_spec_hashes = {result.spec_hash for result in completed}
    audit_type_by_spec_hash = _suppression_audit_types(queue_result.suppression_records, evaluated_spec_hashes)
    started = perf_counter()
    suppression_logged = ledger.log_suppression_batch(
        loop_run_id,
        queue_result.suppression_records,
        audit_type_by_spec_hash=audit_type_by_spec_hash,
    )
    _add_timing(timings, "ledger_write", started)
    suppression_summary = ledger.suppression_summary(loop_run_id) if suppression_logged > 0 else {}
    CriticRegionMemory.from_entries(ledger.list_completed(limit=10_000), registry=REGISTRY).write(critic_memory_path)

    frontier = frontier_manager.rank([entry.to_result() for entry in ledger.top_experiments(limit=args.frontier_limit)])
    print(json_dumps(
        {
            "loop_run_id": loop_run_id,
            "signal_families": list(signal_families),
            "planned": len(planned),
            "accepted": len(queue_result.selected),
            "completed": len(completed),
            "reused": reused,
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
            },
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
        },
        pretty=True,
    ))
