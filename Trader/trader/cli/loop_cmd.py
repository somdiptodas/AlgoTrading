from __future__ import annotations

import argparse
import json

from trader.artifacts.store import ArtifactStore
from trader.config import load_settings
from trader.data.view import DataView
from trader.evaluation.runner import EvaluationRunner
from trader.ledger.store import LedgerStore
from trader.reporting.report import render_experiment_report
from trader.research.critic import HeuristicCritic
from trader.research.frontier import FrontierManager
from trader.research.generator import StrategyGenerator
from trader.research.planner import DeterministicPlanner
from trader.strategies.registry import REGISTRY


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the deterministic autonomous research loop")
    parser.add_argument("--database")
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--embargo-bars", type=int, default=1)
    parser.add_argument("--frontier-limit", type=int, default=5)
    return parser


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

    frontier_entries = ledger.top_experiments(limit=args.frontier_limit)
    frontier_specs = tuple((entry.experiment_id, entry.spec) for entry in frontier_entries)
    planned = planner.plan(batch_size=args.batch_size * 4, frontier_specs=frontier_specs)
    generated = generator.validate_and_filter(
        planned,
        seen_evaluation_key=lambda spec: False,
        evaluation_key_for_spec=lambda spec: spec.spec_hash(),
    )

    completed = []
    reused = 0
    for planned_spec in generated.accepted:
        preview = runner.preview_walk_forward(
            planned_spec.spec,
            num_folds=args.folds,
            embargo_bars=args.embargo_bars,
        )
        evaluation_key = ledger.evaluation_key_for_components(
            preview.spec.spec_hash(),
            preview.data_slice.snapshot_id,
            preview.split_plan_id,
            preview.spec.exec_config.cost_model_id(),
        )
        existing = ledger.get_by_evaluation_key(
            evaluation_key
        )
        if existing is not None:
            reused += 1
            continue
        result = runner.evaluate_walk_forward(
            preview.spec,
            num_folds=args.folds,
            embargo_bars=args.embargo_bars,
            include_robustness=True,
        )
        critique = critic.critique(result)
        report_markdown = render_experiment_report(result, critique=critique.to_payload())
        artifact_paths = artifacts.write_experiment(result, report_markdown=report_markdown, critique=critique.to_payload())
        ledger.record_result(
            result,
            artifact_paths=artifact_paths,
            generator_kind=planned_spec.generator_kind,
            parent_experiment_ids=planned_spec.parent_experiment_ids,
            critique=critique.to_payload(),
        )
        completed.append(result)
        if len(completed) >= args.batch_size:
            break

    frontier = frontier_manager.rank([entry.to_result() for entry in ledger.top_experiments(limit=args.frontier_limit)])
    print(json.dumps(
        {
            "planned": len(planned),
            "accepted": len(generated.accepted),
            "completed": len(completed),
            "reused": reused,
            "rejected": list(generated.rejected),
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
        indent=2,
        sort_keys=True,
    ))
