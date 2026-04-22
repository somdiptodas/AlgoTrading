from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date, datetime, timedelta
import time
from typing import Iterable

from Trader.config import Settings, load_settings
from Trader.massive_client import MassiveAggregatesClient
from Trader.storage import SQLiteBarStore


@dataclass(frozen=True)
class IngestRequest:
    ticker: str
    multiplier: int
    timespan: str
    start_date: date
    end_date: date
    chunk_days: int
    pause_seconds: int
    adjusted: bool
    limit: int


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Standalone Massive market data ingester")
    parser.add_argument("--ticker", default="SPY")
    parser.add_argument("--multiplier", type=int, default=1)
    parser.add_argument("--timespan", default="minute")
    parser.add_argument("--months-back", type=int, default=6)
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument("--chunk-days", type=int, default=30)
    parser.add_argument("--pause-seconds", type=int, default=15)
    parser.add_argument("--database")
    parser.add_argument("--limit", type=int, default=50_000)
    parser.add_argument("--unadjusted", action="store_true")
    return parser


def parse_args(argv: list[str] | None = None) -> tuple[Settings, IngestRequest]:
    parser = build_parser()
    args = parser.parse_args(argv)
    settings = load_settings(database_path=args.database)

    end_date = _parse_date(args.end_date) if args.end_date else date.today()
    start_date = _parse_date(args.start_date) if args.start_date else _subtract_months(end_date, args.months_back)

    if start_date > end_date:
        raise ValueError("start-date must be on or before end-date")

    request = IngestRequest(
        ticker=args.ticker.upper(),
        multiplier=args.multiplier,
        timespan=args.timespan,
        start_date=start_date,
        end_date=end_date,
        chunk_days=args.chunk_days,
        pause_seconds=args.pause_seconds,
        adjusted=not args.unadjusted,
        limit=args.limit,
    )
    return settings, request


def run_ingest(settings: Settings, request: IngestRequest) -> None:
    client = MassiveAggregatesClient(api_key=settings.api_key)
    store = SQLiteBarStore(settings.database_path)

    total_written = 0
    last_timestamp_ms: int | None = None

    print(
        f"Starting ingest for {request.ticker} {request.multiplier}-{request.timespan} "
        f"from {request.start_date.isoformat()} to {request.end_date.isoformat()}"
    )
    print(f"Writing to {settings.database_path}")

    windows = list(iter_date_windows(request.start_date, request.end_date, request.chunk_days))

    for index, (window_start, window_end) in enumerate(windows):
        print(f"Fetching {window_start.isoformat()} -> {window_end.isoformat()}")
        bars = client.list_aggregate_bars(
            ticker=request.ticker,
            multiplier=request.multiplier,
            timespan=request.timespan,
            start_date=window_start,
            end_date=window_end,
            adjusted=request.adjusted,
            limit=request.limit,
        )

        written = store.upsert_bars(bars)
        total_written += written

        if bars:
            last_timestamp_ms = max(bar.timestamp_ms for bar in bars)

        print(f"Fetched {len(bars)} bars, wrote {written}")

        if index < len(windows) - 1 and request.pause_seconds > 0:
            print(f"Sleeping {request.pause_seconds}s to avoid rate limiting")
            time.sleep(request.pause_seconds)

    if last_timestamp_ms is not None:
        store.update_checkpoint(
            ticker=request.ticker,
            multiplier=request.multiplier,
            timespan=request.timespan,
            last_timestamp_ms=last_timestamp_ms,
        )

    summary = store.fetch_summary(request.ticker, request.multiplier, request.timespan)
    if summary is None:
        print("No data written.")
        return

    print(
        "Completed ingest: "
        f"rows={summary['row_count']}, "
        f"first_bar_utc={summary['first_bar_utc']}, "
        f"last_bar_utc={summary['last_bar_utc']}, "
        f"writes_this_run={total_written}"
    )


def iter_date_windows(start_date: date, end_date: date, chunk_days: int) -> Iterable[tuple[date, date]]:
    if chunk_days < 1:
        raise ValueError("chunk-days must be >= 1")

    cursor = start_date
    while cursor <= end_date:
        window_end = min(cursor + timedelta(days=chunk_days - 1), end_date)
        yield cursor, window_end
        cursor = window_end + timedelta(days=1)


def _parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def _subtract_months(anchor: date, months: int) -> date:
    year = anchor.year
    month = anchor.month - months

    while month <= 0:
        month += 12
        year -= 1

    day = min(anchor.day, _days_in_month(year, month))
    return date(year, month, day)


def _days_in_month(year: int, month: int) -> int:
    if month == 12:
        next_month = date(year + 1, 1, 1)
    else:
        next_month = date(year, month + 1, 1)
    return (next_month - timedelta(days=1)).day
