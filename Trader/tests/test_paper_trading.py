from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Sequence

from pytest import raises

from trader.paper.gateway import PaperTradingGateway
from trader.paper.models import BrokerPosition, ExpectedPosition, Fill, OrderAck, OrderRequest
from trader.paper.store import PaperTradingStore


class FakeBroker:
    def __init__(self, positions: Sequence[BrokerPosition] = (), *, ack_client_order_id: str | None = None) -> None:
        self.positions = tuple(positions)
        self.ack_client_order_id = ack_client_order_id
        self.submitted: list[OrderRequest] = []

    def submit_order(self, request: OrderRequest) -> OrderAck:
        self.submitted.append(request)
        return OrderAck(
            client_order_id=self.ack_client_order_id or request.client_order_id,
            broker_order_id=f"broker-{len(self.submitted)}",
            status="accepted",
            submitted_at_utc="2026-04-25T12:00:00+00:00",
        )

    def cancel_order(self, client_order_id: str) -> OrderAck:
        return OrderAck(
            client_order_id=client_order_id,
            broker_order_id="broker-cancel",
            status="canceled",
            submitted_at_utc="2026-04-25T12:01:00+00:00",
        )

    def list_open_orders(self) -> Sequence[OrderAck]:
        return tuple()

    def list_positions(self) -> Sequence[BrokerPosition]:
        return self.positions

    def list_fills(self, *, since_utc: str | None = None) -> Sequence[Fill]:
        return tuple()


def _store(tmp_path: Path) -> PaperTradingStore:
    store = PaperTradingStore(tmp_path / "paper.db")
    store.initialize()
    return store


def test_gateway_submit_order_is_persistently_idempotent(tmp_path: Path) -> None:
    store = _store(tmp_path)
    broker = FakeBroker()
    gateway = PaperTradingGateway(broker, store)
    request = OrderRequest(
        symbol="spy",
        side="buy",
        quantity=10,
        strategy_id="strategy_1",
        client_order_id="client-1",
    )

    first = gateway.submit_order(request, signal_id="signal-1")
    second = gateway.submit_order(request, signal_id="signal-1")

    assert first == second
    assert len(broker.submitted) == 1
    assert store.get_order_ack("client-1") == first


def test_gateway_rejects_reused_client_order_id_for_different_request(tmp_path: Path) -> None:
    store = _store(tmp_path)
    broker = FakeBroker()
    gateway = PaperTradingGateway(broker, store)
    gateway.submit_order(
        OrderRequest(symbol="SPY", side="buy", quantity=10, strategy_id="strategy_1", client_order_id="client-1"),
        signal_id="signal-1",
    )

    with raises(ValueError, match="different order request"):
        gateway.submit_order(
            OrderRequest(symbol="SPY", side="buy", quantity=11, strategy_id="strategy_1", client_order_id="client-1"),
            signal_id="signal-1",
        )
    assert len(broker.submitted) == 1


def test_gateway_rejects_mismatched_broker_ack_client_order_id(tmp_path: Path) -> None:
    store = _store(tmp_path)
    gateway = PaperTradingGateway(FakeBroker(ack_client_order_id="wrong-client-id"), store)

    with raises(ValueError, match="client_order_id must match"):
        gateway.submit_order(
            OrderRequest(symbol="SPY", side="buy", quantity=1, strategy_id="strategy_1", client_order_id="client-1"),
            signal_id="signal-1",
        )
    assert store.get_order_ack("client-1") is None


def test_gateway_generates_stable_client_order_ids(tmp_path: Path) -> None:
    store = _store(tmp_path)
    broker = FakeBroker()
    gateway = PaperTradingGateway(broker, store)
    request = OrderRequest(symbol="spy", side="buy", quantity=10, strategy_id="strategy_1")

    first = gateway.submit_order(request, signal_id="signal-1")
    second = gateway.submit_order(request, signal_id="signal-1")

    assert first.client_order_id
    assert first == second
    assert len(broker.submitted) == 1
    assert broker.submitted[0].symbol == "SPY"


def test_gateway_submit_order_writes_audit_event(tmp_path: Path) -> None:
    store = _store(tmp_path)
    gateway = PaperTradingGateway(FakeBroker(), store)
    request = OrderRequest(symbol="SPY", side="sell", quantity=3, strategy_id="strategy_1", client_order_id="client-2")

    ack = gateway.submit_order(request, signal_id="signal-1")
    events = store.audit_events()

    assert len(events) == 1
    assert events[0].event_type == "order_submitted"
    assert events[0].payload["client_order_id"] == ack.client_order_id
    assert events[0].payload["symbol"] == "SPY"


def test_gateway_reconcile_positions_persists_matching_report(tmp_path: Path) -> None:
    store = _store(tmp_path)
    gateway = PaperTradingGateway(
        FakeBroker((BrokerPosition("SPY", 10, average_price=100.0),)),
        store,
    )

    report = gateway.reconcile_positions(
        (ExpectedPosition("SPY", 10, average_price=100.0),),
        generated_at_utc="2026-04-25T12:00:00+00:00",
    )

    assert report.mismatches == tuple()
    assert store.reconciliation_reports() == (report,)
    assert store.audit_events()[0].event_type == "reconciliation"


def test_gateway_reconcile_positions_reports_mismatches(tmp_path: Path) -> None:
    store = _store(tmp_path)
    gateway = PaperTradingGateway(
        FakeBroker(
            (
                BrokerPosition("SPY", 8, average_price=100.0),
                BrokerPosition("QQQ", 1, average_price=400.0),
            )
        ),
        store,
    )

    report = gateway.reconcile_positions(
        (ExpectedPosition("SPY", 10, average_price=100.0),),
        generated_at_utc="2026-04-25T12:00:00+00:00",
    )

    assert {(item.symbol, item.expected_quantity, item.actual_quantity) for item in report.mismatches} == {
        ("QQQ", 0, 1),
        ("SPY", 10, 8),
    }
    assert len(store.reconciliation_reports()[0].mismatches) == 2


def test_store_enforces_unique_client_order_ids(tmp_path: Path) -> None:
    store = _store(tmp_path)
    request = OrderRequest(symbol="SPY", side="buy", quantity=1, strategy_id="strategy_1", client_order_id="client-1")
    ack = OrderAck(
        client_order_id="client-1",
        broker_order_id="broker-1",
        status="accepted",
        submitted_at_utc="2026-04-25T12:00:00+00:00",
    )

    store.record_order(request, ack)
    with raises(sqlite3.IntegrityError):
        store.record_order(request, ack)
