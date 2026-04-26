from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from trader.config import load_settings
from trader.ledger.entry import json_dumps
from trader.reporting.run_dashboard import ReportPathConventions, RebuildReportsResult, rebuild_reports


@dataclass(frozen=True)
class ReportsRequest:
    reports_dir: Path | None = None
    artifacts_dir: Path | None = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rebuild generated research report HTML")
    parser.add_argument("command", nargs="?", default="rebuild", choices=("rebuild",))
    parser.add_argument("--reports-dir", help="Override the research reports directory")
    parser.add_argument("--artifacts-dir", help="Override the research artifacts directory")
    return parser


def parse_args(argv: list[str] | None = None) -> ReportsRequest:
    args = build_parser().parse_args(argv)
    return ReportsRequest(
        reports_dir=Path(args.reports_dir).expanduser() if args.reports_dir else None,
        artifacts_dir=Path(args.artifacts_dir).expanduser() if args.artifacts_dir else None,
    )


def run_reports(request: ReportsRequest) -> RebuildReportsResult:
    settings = load_settings()
    paths = ReportPathConventions(
        reports_dir=request.reports_dir or settings.reports_dir,
        artifacts_dir=request.artifacts_dir or settings.artifacts_dir,
    )
    result = rebuild_reports(paths)
    print(
        json_dumps(
            {
                "dashboard": str(result.dashboard_path.resolve()),
                "run_reports": [str(path.resolve()) for path in result.run_reports],
                "trade_reports": [str(path.resolve()) for path in result.trade_reports],
            },
            pretty=True,
        )
    )
    return result


def main(argv: list[str] | None = None) -> None:
    run_reports(parse_args(argv))
