from __future__ import annotations

import argparse
import json
from pathlib import Path

from trader.config import load_settings
from trader.data.view import DataView
from trader.evaluation.runner import DEFAULT_LOCKED_HOLDOUT_MONTHS, EvaluationRunner
from trader.ledger.entry import json_dumps
from trader.ledger.store import LedgerStore
from trader.research.decay import build_decay_report, decay_report_to_payload, reevaluate_promoted_specs
from trader.strategies.registry import REGISTRY
from trader.strategies.spec import StrategySpec


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a StrategySpec through the fixed evaluator")
    parser.add_argument("--database")
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--embargo-bars", type=int, default=1)
    parser.add_argument("--holdout-months", type=int, default=DEFAULT_LOCKED_HOLDOUT_MONTHS)
    parser.add_argument("--spec-file")
    parser.add_argument("--spec-json")
    parser.add_argument("--ledger")
    parser.add_argument("--decay-promoted", action="store_true")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--no-robustness", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.decay_promoted:
        _run_decay_monitor(args)
        return
    if not args.spec_file and not args.spec_json:
        raise SystemExit("Provide --spec-file or --spec-json")
    payload = _load_payload(args.spec_file, args.spec_json)
    spec = REGISTRY.validate_spec(StrategySpec.from_payload(payload))
    settings = load_settings(database_path=args.database)
    runner = EvaluationRunner(DataView(settings.database_path), REGISTRY)
    result = runner.evaluate_walk_forward(
        spec,
        num_folds=args.folds,
        embargo_bars=args.embargo_bars,
        locked_holdout_months=args.holdout_months,
        include_robustness=not args.no_robustness,
    )
    print(f"experiment_id={result.experiment_id}")
    print(f"spec_hash={result.spec_hash}")
    print(json_dumps(result.aggregate_metrics, pretty=True))
    if result.robustness_checks:
        print(json_dumps(result.robustness_checks, pretty=True))


def _run_decay_monitor(args: argparse.Namespace) -> None:
    settings = load_settings(database_path=args.database)
    ledger = LedgerStore(args.ledger or settings.ledger_path)
    ledger.initialize()
    runner = EvaluationRunner(DataView(settings.database_path), REGISTRY)
    history = ledger.list_completed(limit=10_000)
    promoted = ledger.query.promoted_experiments(history, limit=args.limit)
    current_snapshot_id = None
    if promoted:
        preview = runner.preview_walk_forward(
            promoted[0].spec,
            num_folds=args.folds,
            embargo_bars=args.embargo_bars,
            locked_holdout_months=args.holdout_months,
        )
        current_snapshot_id = preview.data_slice.snapshot_id
    recorded = reevaluate_promoted_specs(
        ledger,
        runner,
        limit=args.limit,
        num_folds=args.folds,
        embargo_bars=args.embargo_bars,
        locked_holdout_months=args.holdout_months,
        include_robustness=not args.no_robustness,
    )
    history = ledger.list_completed(limit=10_000)
    report = build_decay_report(history, current_snapshot_id=current_snapshot_id, limit=args.limit)
    print(json_dumps(
        {
            "reevaluated": [_summary_item(entry) for entry in recorded],
            "decay_report": decay_report_to_payload(report),
        },
        pretty=True,
    ))


def _summary_item(entry: object) -> dict[str, object]:
    from trader.ledger.entry import LedgerEntry

    if not isinstance(entry, LedgerEntry):
        raise TypeError(f"Expected LedgerEntry, got {type(entry)!r}")
    return {
        "experiment_id": entry.experiment_id,
        "spec_hash": entry.spec_hash,
        "data_snapshot_id": entry.data_snapshot_id,
        "promotion_stage": entry.promotion_stage,
    }


def _load_payload(spec_file: str | None, spec_json: str | None) -> dict[str, object]:
    try:
        if spec_file:
            return _loads_strategy_json(Path(spec_file).read_text(encoding="utf-8"))
        if spec_json:
            return _loads_strategy_json(spec_json)
    except ValueError as exc:
        raise SystemExit(f"Invalid strategy JSON: {exc}") from exc
    raise ValueError("spec payload missing")


def _loads_strategy_json(payload: str) -> dict[str, object]:
    return json.loads(payload, parse_constant=_reject_non_standard_json_constant)


def _reject_non_standard_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON numeric value is not supported: {value}")
