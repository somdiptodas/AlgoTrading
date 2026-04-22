from __future__ import annotations

from trader.data.ingest import parse_args, run_ingest


def main(argv: list[str] | None = None) -> None:
    settings, request = parse_args(argv)
    run_ingest(settings, request)
