from __future__ import annotations

from datetime import UTC, datetime

from trader.paper.audit import AuditEvent, order_submitted_event
from trader.paper.broker import PaperBroker, with_client_order_id
from trader.paper.models import ExpectedPosition, OrderAck, OrderRequest
from trader.paper.reconcile import ReconciliationReport, build_reconciliation_report
from trader.paper.store import PaperTradingStore


class PaperTradingGateway:
    def __init__(self, broker: PaperBroker, store: PaperTradingStore) -> None:
        self.broker = broker
        self.store = store

    def submit_order(self, request: OrderRequest, *, signal_id: str) -> OrderAck:
        prepared = with_client_order_id(request, signal_id=signal_id)
        existing = self.store.get_order_ack(prepared.client_order_id)
        if existing is not None:
            stored_request = self.store.get_order_request(prepared.client_order_id)
            if stored_request != prepared:
                raise ValueError("client_order_id already exists for a different order request")
            return existing
        ack = self.broker.submit_order(prepared)
        if ack.client_order_id != prepared.client_order_id:
            raise ValueError("broker ack client_order_id must match request client_order_id")
        self.store.record_order(prepared, ack)
        event = order_submitted_event(prepared, occurred_at_utc=ack.submitted_at_utc)
        self.store.append_audit_event(
            event_id=f"order_submitted:{prepared.client_order_id}",
            event=event,
            client_order_id=prepared.client_order_id,
            broker_order_id=ack.broker_order_id,
        )
        return ack

    def reconcile_positions(
        self,
        expected_positions: tuple[ExpectedPosition, ...],
        *,
        generated_at_utc: str | None = None,
    ) -> ReconciliationReport:
        generated_at = generated_at_utc or datetime.now(UTC).isoformat()
        report = build_reconciliation_report(
            expected_positions,
            tuple(self.broker.list_positions()),
            generated_at_utc=generated_at,
        )
        self.store.record_reconciliation_report(report)
        self.store.append_audit_event(
            event_id=f"reconciliation:{report.report_id}",
            event=AuditEvent(
                event_type="reconciliation",
                occurred_at_utc=report.generated_at_utc,
                payload={
                    "report_id": report.report_id,
                    "mismatch_count": len(report.mismatches),
                },
            ),
        )
        return report
