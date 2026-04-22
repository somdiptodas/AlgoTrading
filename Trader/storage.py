from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable

from Trader.models import AggregateBar


class SQLiteBarStore:
    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS aggregate_bars (
                    ticker TEXT NOT NULL,
                    multiplier INTEGER NOT NULL,
                    timespan TEXT NOT NULL,
                    timestamp_ms INTEGER NOT NULL,
                    timestamp_utc TEXT NOT NULL,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume REAL,
                    vwap REAL,
                    transactions INTEGER,
                    PRIMARY KEY (ticker, multiplier, timespan, timestamp_ms)
                );

                CREATE TABLE IF NOT EXISTS ingest_checkpoints (
                    ticker TEXT NOT NULL,
                    multiplier INTEGER NOT NULL,
                    timespan TEXT NOT NULL,
                    last_timestamp_ms INTEGER NOT NULL,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (ticker, multiplier, timespan)
                );
                """
            )

    def upsert_bars(self, bars: Iterable[AggregateBar]) -> int:
        rows = [
            (
                bar.ticker,
                bar.multiplier,
                bar.timespan,
                bar.timestamp_ms,
                bar.timestamp_utc,
                bar.open,
                bar.high,
                bar.low,
                bar.close,
                bar.volume,
                bar.vwap,
                bar.transactions,
            )
            for bar in bars
        ]

        if not rows:
            return 0

        with self._connect() as connection:
            connection.executemany(
                """
                INSERT INTO aggregate_bars (
                    ticker,
                    multiplier,
                    timespan,
                    timestamp_ms,
                    timestamp_utc,
                    open,
                    high,
                    low,
                    close,
                    volume,
                    vwap,
                    transactions
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(ticker, multiplier, timespan, timestamp_ms) DO UPDATE SET
                    timestamp_utc = excluded.timestamp_utc,
                    open = excluded.open,
                    high = excluded.high,
                    low = excluded.low,
                    close = excluded.close,
                    volume = excluded.volume,
                    vwap = excluded.vwap,
                    transactions = excluded.transactions
                """,
                rows,
            )
        return len(rows)

    def update_checkpoint(self, ticker: str, multiplier: int, timespan: str, last_timestamp_ms: int) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO ingest_checkpoints (ticker, multiplier, timespan, last_timestamp_ms)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(ticker, multiplier, timespan) DO UPDATE SET
                    last_timestamp_ms = excluded.last_timestamp_ms,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (ticker, multiplier, timespan, last_timestamp_ms),
            )

    def fetch_summary(self, ticker: str, multiplier: int, timespan: str) -> sqlite3.Row | None:
        with self._connect() as connection:
            return connection.execute(
                """
                SELECT
                    COUNT(*) AS row_count,
                    MIN(timestamp_utc) AS first_bar_utc,
                    MAX(timestamp_utc) AS last_bar_utc
                FROM aggregate_bars
                WHERE ticker = ? AND multiplier = ? AND timespan = ?
                """,
                (ticker, multiplier, timespan),
            ).fetchone()
