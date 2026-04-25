from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from trader.ledger.entry import json_dumps, json_loads
from trader.paper.models import Fill, OrderAck, OrderRequest


AuditEventType = Literal["order_submitted", "order_ack", "fill", "reconciliation"]


@dataclass(frozen=True)
class AuditEvent:
    event_type: AuditEventType
    occurred_at_utc: str
    payload: dict[str, object]


class PaperTradeAuditLog:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def append(self, event: AuditEvent) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json_dumps(asdict(event)) + "\n")

    def read_events(self) -> tuple[AuditEvent, ...]:
        if not self.path.exists():
            return tuple()
        events: list[AuditEvent] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                payload = dict(json_loads(line, default={}))
                events.append(
                    AuditEvent(
                        event_type=payload["event_type"],
                        occurred_at_utc=str(payload["occurred_at_utc"]),
                        payload=dict(payload["payload"]),
                    )
                )
        return tuple(events)


def order_submitted_event(request: OrderRequest, *, occurred_at_utc: str) -> AuditEvent:
    return AuditEvent("order_submitted", occurred_at_utc, asdict(request))


def order_ack_event(ack: OrderAck) -> AuditEvent:
    return AuditEvent("order_ack", ack.submitted_at_utc, asdict(ack))


def fill_event(fill: Fill) -> AuditEvent:
    return AuditEvent("fill", fill.filled_at_utc, asdict(fill))
