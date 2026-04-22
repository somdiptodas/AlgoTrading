from __future__ import annotations

from trader.reporting.visualize import parse_args, run_visualize


def main(argv: list[str] | None = None) -> None:
    run_visualize(parse_args(argv))
