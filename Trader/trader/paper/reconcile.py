from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from typing import Sequence

from trader.ledger.entry import json_dumps
from trader.paper.models import BrokerPosition, ExpectedPosition


@dataclass(frozen=True)
class PositionMismatch:
    symbol: str
    expected_quantity: int
    actual_quantity: int
    expected_average_price: float
    actual_average_price: float
    reason: str


@dataclass(frozen=True)
class ReconciliationReport:
    report_id: str
    generated_at_utc: str
    mismatches: tuple[PositionMismatch, ...]


def reconcile_positions(
    expected: Sequence[ExpectedPosition],
    actual: Sequence[BrokerPosition],
    *,
    average_price_tolerance: float = 0.01,
) -> tuple[PositionMismatch, ...]:
    expected_by_symbol = {position.symbol.upper(): position for position in expected}
    actual_by_symbol = {position.symbol.upper(): position for position in actual}
    mismatches: list[PositionMismatch] = []
    for symbol in sorted(set(expected_by_symbol) | set(actual_by_symbol)):
        expected_position = expected_by_symbol.get(symbol, ExpectedPosition(symbol, 0))
        actual_position = actual_by_symbol.get(symbol, BrokerPosition(symbol, 0))
        expected_quantity = expected_position.quantity
        actual_quantity = actual_position.quantity
        if expected_quantity != actual_quantity:
            mismatches.append(
                PositionMismatch(
                    symbol=symbol,
                    expected_quantity=expected_quantity,
                    actual_quantity=actual_quantity,
                    expected_average_price=expected_position.average_price,
                    actual_average_price=actual_position.average_price,
                    reason="quantity_mismatch",
                )
            )
        elif abs(expected_position.average_price - actual_position.average_price) > average_price_tolerance:
            mismatches.append(
                PositionMismatch(
                    symbol=symbol,
                    expected_quantity=expected_quantity,
                    actual_quantity=actual_quantity,
                    expected_average_price=expected_position.average_price,
                    actual_average_price=actual_position.average_price,
                    reason="average_price_mismatch",
                )
            )
    return tuple(mismatches)


def build_reconciliation_report(
    expected: Sequence[ExpectedPosition],
    actual: Sequence[BrokerPosition],
    *,
    generated_at_utc: str,
    average_price_tolerance: float = 0.01,
) -> ReconciliationReport:
    mismatches = reconcile_positions(
        expected,
        actual,
        average_price_tolerance=average_price_tolerance,
    )
    payload = json_dumps(
        {
            "generated_at_utc": generated_at_utc,
            "mismatches": [asdict(mismatch) for mismatch in mismatches],
        }
    )
    return ReconciliationReport(
        report_id=sha256(payload.encode("utf-8")).hexdigest()[:24],
        generated_at_utc=generated_at_utc,
        mismatches=mismatches,
    )
