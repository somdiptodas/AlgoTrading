from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256

from trader.artifacts.store import ArtifactStore
from trader.config import load_settings
from trader.data.view import DataView
from trader.evaluation.runner import EvaluationRunner
from trader.ledger.entry import json_dumps
from trader.ledger.store import LedgerStore
from trader.reporting.report import render_experiment_report
from trader.research.candidates import DeterministicCandidateQueue
from trader.research.critic import HeuristicCritic
from trader.research.frontier import FrontierManager
from trader.research.generator import StrategyGenerator
from trader.research.planner import DeterministicPlanner
from trader.research.suppressor import RegionSuppressor
from trader.strategies.registry import REGISTRY

_ALL_SIGNAL_FAMILIES = ("ema_cross", "breakout", "rsi_reversion")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the deterministic autonomous research loop")
    parser.add_argument("--database")
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--embargo-bars", type=int, default=1)
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
        radius=args.suppressor_radius,
        weight_cap=args.suppressor_weight_cap,
    )

    signal_families = tuple(args.signal_family) if args.signal_family else _ALL_SIGNAL_FAMILIES
    frontier_specs = tuple((entry.experiment_id, entry.spec) for entry in frontier_entries)
    planned = planner.plan(
        batch_size=max(args.batch_size * 12, 64),
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
    )

    # --- Persist suppression audit records to the ledger ---
    suppression_logged = ledger.log_suppression_batch(loop_run_id, queue_result.suppression_records)
    suppression_summary = ledger.suppression_summary(loop_run_id) if suppression_logged > 0 else {}

    completed = []
    reused = queue_result.duplicate_count
    for candidate in queue_result.selected[: args.batch_size]:
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

    frontier = frontier_manager.rank([entry.to_result() for entry in ledger.top_experiments(limit=args.frontier_limit)])
    print(json_dumps(
        {
            "loop_run_id": loop_run_id,
            "signal_families": list(signal_families),
            "planned": len(planned),
            "accepted": len(queue_result.selected),
            "completed": len(completed),
            "reused": reused,
            "rejected": list(generated.rejected),
            "suppressor": {
                **suppressor.summary(),
                "candidates_suppressed": suppression_logged,
                "by_family": suppression_summary.get("by_family", []),
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
