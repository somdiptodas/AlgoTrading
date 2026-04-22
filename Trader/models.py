from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


def _get_field(payload: Any, *names: str) -> Any:
    for name in names:
        if hasattr(payload, name):
            return getattr(payload, name)
        if isinstance(payload, dict) and name in payload:
            return payload[name]
    return None


@dataclass(frozen=True)
class AggregateBar:
    ticker: str
    multiplier: int
    timespan: str
    timestamp_ms: int
    open: float | None
    high: float | None
    low: float | None
    close: float | None
    volume: float | None
    vwap: float | None
    transactions: int | None

    @property
    def timestamp_utc(self) -> str:
        return datetime.fromtimestamp(self.timestamp_ms / 1000, tz=timezone.utc).isoformat()

    @classmethod
    def from_sdk(cls, ticker: str, multiplier: int, timespan: str, payload: Any) -> "AggregateBar":
        timestamp_ms = _get_field(payload, "timestamp", "t")
        if timestamp_ms is None:
            raise ValueError(f"Aggregate payload missing timestamp: {payload!r}")

        return cls(
            ticker=ticker,
            multiplier=multiplier,
            timespan=timespan,
            timestamp_ms=int(timestamp_ms),
            open=_coerce_float(_get_field(payload, "open", "o")),
            high=_coerce_float(_get_field(payload, "high", "h")),
            low=_coerce_float(_get_field(payload, "low", "l")),
            close=_coerce_float(_get_field(payload, "close", "c")),
            volume=_coerce_float(_get_field(payload, "volume", "v")),
            vwap=_coerce_float(_get_field(payload, "vwap", "vw")),
            transactions=_coerce_int(_get_field(payload, "transactions", "n")),
        )


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)
