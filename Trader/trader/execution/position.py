from __future__ import annotations

from dataclasses import dataclass

from trader.strategies.decisions import RuleDecision


@dataclass
class Position:
    entry_timestamp_ms: int
    entry_timestamp_utc: str
    entry_reference_price: float
    entry_price: float
    shares: int
    entry_commission: float
    entry_cost_cash: float
    entry_reason: str = "signal_on"
    entry_rule: RuleDecision | None = None
    bars_held: int = 0
