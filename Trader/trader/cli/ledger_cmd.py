from __future__ import annotations

import argparse
import json

from trader.config import load_settings
from trader.ledger.store import LedgerStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect research ledger summaries")
    parser.add_argument("command", nargs="?", default="summary", choices=("summary",))
    parser.add_argument("--ledger")
    parser.add_argument("--limit", type=int, default=10)
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
        print(json.dumps(payload, indent=2, sort_keys=True))
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
