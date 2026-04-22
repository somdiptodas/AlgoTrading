from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from trader.data.models import AggregateBar
from trader.data.storage import SQLiteBarStore


NEW_YORK = ZoneInfo("America/New_York")


def _bar(timestamp: datetime, close: float) -> AggregateBar:
    timestamp_utc = timestamp.astimezone(timezone.utc)
    return AggregateBar(
        ticker="SPY",
        multiplier=1,
        timespan="minute",
        timestamp_ms=int(timestamp_utc.timestamp() * 1000),
        open=close,
        high=close + 0.25,
        low=close - 0.25,
        close=close,
        volume=1_000.0,
        vwap=close,
        transactions=10,
    )


@pytest.fixture
def sample_db_path(tmp_path: Path) -> Path:
    return tmp_path / "market_data.db"


@pytest.fixture
def sample_store(sample_db_path: Path) -> SQLiteBarStore:
    return SQLiteBarStore(sample_db_path)


@pytest.fixture
def seeded_store(sample_store: SQLiteBarStore) -> SQLiteBarStore:
    start = datetime(2026, 1, 5, 9, 30, tzinfo=NEW_YORK)
    bars = []
    for index in range(240):
        bars.append(_bar(start + timedelta(minutes=index), 100.0 + (index * 0.1)))
    sample_store.upsert_bars(bars)
    return sample_store
