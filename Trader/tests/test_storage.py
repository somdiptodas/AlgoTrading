from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from trader.data.models import AggregateBar
from trader.data.storage import SQLiteBarStore


NEW_YORK = ZoneInfo("America/New_York")


def _make_bar(index: int, close: float) -> AggregateBar:
    timestamp = datetime(2026, 1, 5, 9, 30, tzinfo=NEW_YORK) + timedelta(minutes=index)
    timestamp_utc = timestamp.astimezone(tz=None)
    return AggregateBar(
        ticker="SPY",
        multiplier=1,
        timespan="minute",
        timestamp_ms=int(timestamp_utc.timestamp() * 1000),
        open=close,
        high=close + 0.25,
        low=close - 0.25,
        close=close,
        volume=1000.0,
        vwap=close,
        transactions=10,
    )


def test_storage_upsert_and_summary(sample_store: SQLiteBarStore) -> None:
    bars = [_make_bar(index, 100.0 + index) for index in range(5)]
    assert sample_store.upsert_bars(bars) == 5
    assert sample_store.upsert_bars(bars) == 5
    summary = sample_store.fetch_summary("SPY", 1, "minute")
    assert summary is not None
    assert summary["row_count"] == 5
    rows = sample_store.fetch_bars("SPY", 1, "minute")
    assert len(rows) == 5


def test_bucketed_fetch_returns_ohlc_rows(sample_store: SQLiteBarStore) -> None:
    bars = [_make_bar(index, 100.0 + index) for index in range(6)]
    sample_store.upsert_bars(bars)
    rows = sample_store.fetch_bucketed_bars("SPY", 1, "minute", bucket_ms=120_000)
    assert len(rows) == 3
    first = rows[0]
    assert float(first["open"]) == 100.0
    assert float(first["close"]) == 101.0
