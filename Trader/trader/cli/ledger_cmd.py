from __future__ import annotations

import argparse

from trader.config import load_settings
from trader.ledger.entry import json_dumps
from trader.ledger.store import LedgerStore
from trader.research.decay import build_decay_report, decay_report_to_payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect research ledger summaries")
    parser.add_argument("command", nargs="?", default="summary", choices=("summary", "decay"))
    parser.add_argument("--ledger")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--current-snapshot-id")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    settings = load_settings()
    ledger = LedgerStore(args.ledger or settings.ledger_path)
    ledger.initialize()

    if args.command == "summary":
        stats = ledger.stats()
        recent = ledger.list_completed(limit=args.limit)
        top = ledger.top_experiments(limit=args.limit)
        payload = {
            "ledger_path": str(ledger.database_path.resolve()),
            "total_entries": stats["total"],
            "by_status": stats["by_status"],
            "recent_completed": [_summary_item(entry) for entry in recent],
            "top_experiments": [_summary_item(entry) for entry in top],
        }
        print(json_dumps(payload, pretty=True))
        return
    if args.command == "decay":
        entries = ledger.list_completed(limit=10_000)
        report = build_decay_report(entries, current_snapshot_id=args.current_snapshot_id, limit=args.limit)
        payload = {
            "ledger_path": str(ledger.database_path.resolve()),
            "decay_report": decay_report_to_payload(report),
        }
        print(json_dumps(payload, pretty=True))
        return

    raise SystemExit(f"Unknown ledger command: {args.command}")


def _summary_item(entry: object) -> dict[str, object]:
    from trader.ledger.entry import LedgerEntry

    if not isinstance(entry, LedgerEntry):
        raise TypeError(f"Expected LedgerEntry, got {type(entry)!r}")
    return {
        "experiment_id": entry.experiment_id,
        "family": entry.spec.signal.name,
        "name": entry.spec.name,
        "promotion_stage": entry.promotion_stage,
        "return_pct": entry.metric("return_pct"),
        "sharpe_like": entry.metric("sharpe_like"),
        "max_drawdown_pct": entry.metric("max_drawdown_pct"),
        "trade_count": entry.metric("trade_count"),
        "completed_at_utc": entry.completed_at_utc,
    }
