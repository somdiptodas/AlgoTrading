from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256

from trader.artifacts.store import ArtifactStore
from trader.config import load_settings
from trader.data.view import DataView
from trader.evaluation.runner import DEFAULT_LOCKED_HOLDOUT_MONTHS, EvaluationRunner
from trader.ledger.entry import json_dumps
from trader.ledger.store import LedgerStore
from trader.reporting.report import render_experiment_report
from trader.research.candidates import DeterministicCandidateQueue
from trader.research.critic import HeuristicCritic
from trader.research.frontier import FrontierManager
from trader.research.generator import StrategyGenerator
from trader.research.planner import DeterministicPlanner
from trader.research.suppressor import RegionSuppressor, SuppressedSpec
from trader.strategies.registry import REGISTRY

_ALL_SIGNAL_FAMILIES = ("ema_cross", "breakout", "rsi_reversion")
DEFAULT_OVERPLAN_FACTOR = 12
DEFAULT_PREVIEW_FACTOR = 4
MIN_PLANNED_SPECS = 64


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


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
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
    planned = planner.plan(
        batch_size=_planned_spec_count(args.batch_size, args.overplan_factor),
        frontier_specs=frontier_specs,
        allowed_signal_families=signal_families,
    )
    generated = generator.validate_and_filter(
        planned,
        seen_evaluation_key=lambda spec: False,
        evaluation_key_for_spec=lambda spec: spec.spec_hash(),
    )
    candidate_queue = DeterministicCandidateQueue(
        history_entries=history_entries,
        frontier_entries=frontier_entries,
        suppressor=suppressor,
    )
    queue_result = candidate_queue.build(
        planned_specs=generated.accepted,
        runner=runner,
        num_folds=args.folds,
        embargo_bars=args.embargo_bars,
        locked_holdout_months=args.holdout_months,
        max_preview_count=_max_preview_count(args.batch_size, args.preview_factor),
    )

    completed = []
    reused = queue_result.duplicate_count
    selected_candidates = queue_result.selected[: args.batch_size]
    for candidate in selected_candidates:
        result = runner.evaluate_preview(candidate.preview, include_robustness=True)
        critique = critic.critique(result)
        report_markdown = render_experiment_report(result, critique=critique.to_payload())
        artifact_paths = artifacts.write_experiment(result, report_markdown=report_markdown, critique=critique.to_payload())
        ledger.record_result(
            result,
            artifact_paths=artifact_paths,
            generator_kind=candidate.planned.generator_kind,
            parent_experiment_ids=candidate.planned.parent_experiment_ids,
            critique=critique.to_payload(),
        )
        completed.append(result)

    # --- Persist suppression audit records to the ledger ---
    evaluated_spec_hashes = {result.spec_hash for result in completed}
    audit_type_by_spec_hash = _suppression_audit_types(queue_result.suppression_records, evaluated_spec_hashes)
    suppression_logged = ledger.log_suppression_batch(
        loop_run_id,
        queue_result.suppression_records,
        audit_type_by_spec_hash=audit_type_by_spec_hash,
    )
    suppression_summary = ledger.suppression_summary(loop_run_id) if suppression_logged > 0 else {}

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
