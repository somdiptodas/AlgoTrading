from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Position:
    entry_timestamp_ms: int
    entry_timestamp_utc: str
    entry_price: float
    shares: int
    entry_commission: float
    bars_held: int = 0
