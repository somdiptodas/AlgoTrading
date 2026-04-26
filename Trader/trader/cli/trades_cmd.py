from __future__ import annotations

import argparse
import webbrowser
from dataclasses import dataclass
from pathlib import Path

from trader.config import load_settings
from trader.reporting.trade_visualization import write_trade_visualization


@dataclass(frozen=True)
class TradesRequest:
    experiment_id: str
    output_path: Path | None
    open_browser: bool


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render an experiment trade/equity HTML review")
    parser.add_argument("experiment_id", help="Experiment ID under data/research/artifacts")
    parser.add_argument("--output", help="Output HTML path")
    parser.add_argument("--no-open", action="store_true", help="Write the report without opening a browser")
    return parser


def parse_args(argv: list[str] | None = None) -> TradesRequest:
    args = build_parser().parse_args(argv)
    return TradesRequest(
        experiment_id=args.experiment_id,
        output_path=Path(args.output).expanduser() if args.output else None,
        open_browser=not args.no_open,
    )


def run_trades(request: TradesRequest) -> Path:
    settings = load_settings()
    experiment_dir = settings.artifacts_dir / request.experiment_id
    if not experiment_dir.exists():
        raise FileNotFoundError(f"No artifact directory found for experiment {request.experiment_id}: {experiment_dir}")
    output_path = request.output_path or settings.reports_dir / f"{request.experiment_id}_trades.html"
    written = write_trade_visualization(experiment_dir, output_path)
    print(f"Wrote trade review to {written}")
    if request.open_browser:
        webbrowser.open(written.resolve().as_uri())
    return written


def main(argv: list[str] | None = None) -> None:
    run_trades(parse_args(argv))
