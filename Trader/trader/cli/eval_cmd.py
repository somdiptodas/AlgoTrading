from __future__ import annotations

import argparse
import json
from pathlib import Path

from trader.config import load_settings
from trader.data.view import DataView
from trader.evaluation.runner import EvaluationRunner
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
    print(json.dumps(result.aggregate_metrics, indent=2, sort_keys=True))
    if result.robustness_checks:
        print(json.dumps(result.robustness_checks, indent=2, sort_keys=True))


def _load_payload(spec_file: str | None, spec_json: str | None) -> dict[str, object]:
    if spec_file:
        return json.loads(Path(spec_file).read_text(encoding="utf-8"))
    if spec_json:
        return json.loads(spec_json)
    raise ValueError("spec payload missing")
