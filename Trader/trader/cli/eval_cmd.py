from __future__ import annotations

import argparse
import json
from pathlib import Path

from trader.config import load_settings
from trader.data.view import DataView
from trader.evaluation.runner import EvaluationRunner
from trader.ledger.entry import json_dumps
from trader.strategies.registry import REGISTRY
from trader.strategies.spec import StrategySpec


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a StrategySpec through the fixed evaluator")
    parser.add_argument("--database")
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--embargo-bars", type=int, default=1)
    parser.add_argument("--spec-file")
    parser.add_argument("--spec-json")
    parser.add_argument("--no-robustness", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
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
        include_robustness=not args.no_robustness,
    )
    print(f"experiment_id={result.experiment_id}")
    print(f"spec_hash={result.spec_hash}")
    print(json_dumps(result.aggregate_metrics, pretty=True))
    if result.robustness_checks:
        print(json_dumps(result.robustness_checks, pretty=True))


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
