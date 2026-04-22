from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Iterable, Sequence

from trader.data.models import MarketBar
from trader.data.storage import SQLiteBarStore


@dataclass(frozen=True)
class DataSlice:
    bars: tuple[MarketBar, ...]
    snapshot_id: str
    first_timestamp_utc: str | None
    last_timestamp_utc: str | None


class DataView:
    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        self.store = SQLiteBarStore(database_path)

    def bars(
        self,
        ticker: str,
        multiplier: int,
        timespan: str,
        start_ms: int | None = None,
        end_ms: int | None = None,
        *,
        regular_session_only: bool = False,
    ) -> tuple[MarketBar, ...]:
        rows = self.store.fetch_bars(
            ticker=ticker,
            multiplier=multiplier,
            timespan=timespan,
            start_timestamp_ms=start_ms,
            end_timestamp_ms=end_ms,
        )
        bars: list[MarketBar] = []
        for row in rows:
            if None in (row["open"], row["high"], row["low"], row["close"]):
                continue
            bar = MarketBar(
                timestamp_ms=int(row["timestamp_ms"]),
                timestamp_utc=str(row["timestamp_utc"]),
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row["volume"] or 0.0),
            )
            if regular_session_only and not bar.is_regular_session:
                continue
            bars.append(bar)
        return tuple(bars)

    def slice(
        self,
        ticker: str,
        multiplier: int,
        timespan: str,
        start_ms: int | None = None,
        end_ms: int | None = None,
        *,
        regular_session_only: bool = False,
    ) -> DataSlice:
        bars = self.bars(
            ticker=ticker,
            multiplier=multiplier,
            timespan=timespan,
            start_ms=start_ms,
            end_ms=end_ms,
            regular_session_only=regular_session_only,
        )
        return DataSlice(
            bars=bars,
            snapshot_id=self.snapshot_hash(bars),
            first_timestamp_utc=bars[0].timestamp_utc if bars else None,
            last_timestamp_utc=bars[-1].timestamp_utc if bars else None,
        )

    @staticmethod
    def snapshot_hash(bars: Sequence[MarketBar]) -> str:
        digest = sha256()
        for bar in bars:
            digest.update(
                f"{bar.timestamp_ms}|{bar.open:.6f}|{bar.high:.6f}|{bar.low:.6f}|{bar.close:.6f}|{bar.volume:.6f}".encode(
                    "utf-8"
                )
            )
        return digest.hexdigest()

    @staticmethod
    def bars_to_payload(bars: Iterable[MarketBar]) -> list[dict[str, float | int | str]]:
        return [
            {
                "timestamp_ms": bar.timestamp_ms,
                "timestamp_utc": bar.timestamp_utc,
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": bar.volume,
            }
            for bar in bars
        ]
