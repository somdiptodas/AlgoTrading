from __future__ import annotations

from pathlib import Path

from pytest import raises

from trader.paper.gateway import PaperTradingGateway
from trader.paper.models import BrokerPosition, Fill, OrderAck, OrderRequest
from trader.paper.risk import NoTradeWindow, RiskConfig, RiskManager
from trader.paper.store import PaperTradingStore


class FakeBroker:
    def __init__(self, positions: tuple[BrokerPosition, ...] = ()) -> None:
        self.positions = positions
        self.submitted: list[OrderRequest] = []

    def submit_order(self, request: OrderRequest) -> OrderAck:
        self.submitted.append(request)
        return OrderAck(
            client_order_id=request.client_order_id,
            broker_order_id=f"broker-{len(self.submitted)}",
            status="accepted",
            submitted_at_utc="2026-04-25T12:00:00+00:00",
        )

    def cancel_order(self, client_order_id: str) -> OrderAck:
        return OrderAck(client_order_id, "broker-cancel", "canceled", "2026-04-25T12:00:00+00:00")

    def list_open_orders(self) -> tuple[OrderAck, ...]:
        return tuple()

    def list_positions(self) -> tuple[BrokerPosition, ...]:
        return self.positions

    def list_fills(self, *, since_utc: str | None = None) -> tuple[Fill, ...]:
        return tuple()


def _gateway(
    tmp_path: Path,
    config: RiskConfig,
    *,
    positions: tuple[BrokerPosition, ...] = (),
) -> tuple[PaperTradingGateway, FakeBroker, PaperTradingStore]:
    store = PaperTradingStore(tmp_path / "paper.db")
    store.initialize()
    broker = FakeBroker(positions)
    return PaperTradingGateway(broker, store, RiskManager(config)), broker, store


def _request(client_order_id: str = "client-1", *, quantity: int = 10, reference_price: float = 100.0) -> OrderRequest:
    return OrderRequest(
        symbol="SPY",
        side="buy",
        quantity=quantity,
        strategy_id="strategy_1",
        client_order_id=client_order_id,
        reference_price=reference_price,
    )


def test_risk_manager_blocks_max_position_notional(tmp_path: Path) -> None:
    gateway, broker, _ = _gateway(tmp_path, RiskConfig(max_position_notional=500.0))

    with raises(ValueError, match="max_position_notional"):
        gateway.submit_order(_request(quantity=10, reference_price=100.0), signal_id="signal-1")

    assert broker.submitted == []


def test_risk_manager_allows_reducing_sell_when_current_notional_is_high(tmp_path: Path) -> None:
    gateway, broker, _ = _gateway(
        tmp_path,
        RiskConfig(max_position_notional=500.0),
        positions=(BrokerPosition("SPY", 10, average_price=100.0),),
    )

    ack = gateway.submit_order(
        OrderRequest(
            symbol="SPY",
            side="sell",
            quantity=5,
            strategy_id="strategy_1",
            client_order_id="client-1",
            reference_price=100.0,
        ),
        signal_id="signal-1",
    )

    assert ack.status == "accepted"
    assert len(broker.submitted) == 1


def test_risk_manager_rejects_missing_price_for_notional_check(tmp_path: Path) -> None:
    gateway, broker, _ = _gateway(tmp_path, RiskConfig(max_position_notional=500.0))

    with raises(ValueError, match="reference_price"):
        gateway.submit_order(
            OrderRequest(symbol="SPY", side="buy", quantity=1, strategy_id="strategy_1", client_order_id="client-1"),
            signal_id="signal-1",
        )

    assert broker.submitted == []


def test_risk_manager_blocks_max_daily_loss(tmp_path: Path) -> None:
    gateway, broker, store = _gateway(tmp_path, RiskConfig(max_daily_loss=1_000.0))
    store.set_daily_realized_pnl("2026-04-25", -1_000.0, updated_at_utc="2026-04-25T12:00:00+00:00")

    with raises(ValueError, match="max_daily_loss"):
        gateway.submit_order(
            _request(),
            signal_id="signal-1",
            now_utc="2026-04-25T20:00:00+00:00",
        )

    assert broker.submitted == []


def test_risk_manager_blocks_max_orders_per_day(tmp_path: Path) -> None:
    gateway, broker, _ = _gateway(tmp_path, RiskConfig(max_orders_per_day=1))

    gateway.submit_order(_request("client-1"), signal_id="signal-1", now_utc="2026-04-25T12:00:00+00:00")
    with raises(ValueError, match="max_orders_per_day"):
        gateway.submit_order(_request("client-2"), signal_id="signal-2", now_utc="2026-04-25T13:00:00+00:00")

    assert len(broker.submitted) == 1


def test_risk_manager_counts_orders_by_new_york_trading_day(tmp_path: Path) -> None:
    gateway, broker, _ = _gateway(tmp_path, RiskConfig(max_orders_per_day=1))

    gateway.submit_order(_request("client-1"), signal_id="signal-1", now_utc="2026-04-26T01:00:00+00:00")
    ack = gateway.submit_order(_request("client-2"), signal_id="signal-2", now_utc="2026-04-26T14:00:00+00:00")

    assert ack.status == "accepted"
    assert len(broker.submitted) == 2


def test_risk_manager_idempotent_replay_bypasses_new_risk_blocks(tmp_path: Path) -> None:
    gateway, broker, store = _gateway(tmp_path, RiskConfig(max_orders_per_day=1))

    first = gateway.submit_order(_request("client-1"), signal_id="signal-1", now_utc="2026-04-25T12:00:00+00:00")
    store.set_kill_switch(True, reason="manual", updated_at_utc="2026-04-25T12:01:00+00:00")
    second = gateway.submit_order(_request("client-1"), signal_id="signal-1", now_utc="2026-04-25T12:02:00+00:00")

    assert first == second
    assert len(broker.submitted) == 1


def test_risk_manager_blocks_kill_switch(tmp_path: Path) -> None:
    gateway, broker, store = _gateway(tmp_path, RiskConfig())
    store.set_kill_switch(True, reason="manual", updated_at_utc="2026-04-25T12:00:00+00:00")

    with raises(ValueError, match="kill_switch"):
        gateway.submit_order(_request(), signal_id="signal-1")

    assert broker.submitted == []
    assert store.audit_events()[0].event_type == "risk_blocked"
    assert store.audit_events()[0].payload["reason"] == "kill_switch:manual"


def test_risk_manager_blocks_no_trade_dates_and_windows(tmp_path: Path) -> None:
    gateway, broker, _ = _gateway(
        tmp_path,
        RiskConfig(
            no_trade_dates=("2026-04-25",),
            no_trade_windows=(NoTradeWindow("2026-04-26T14:00:00+00:00", "2026-04-26T15:00:00+00:00"),),
        ),
    )

    with raises(ValueError, match="no_trade_date"):
        gateway.submit_order(_request("client-1"), signal_id="signal-1", now_utc="2026-04-25T12:00:00+00:00")
    with raises(ValueError, match="no_trade_window"):
        gateway.submit_order(_request("client-2"), signal_id="signal-2", now_utc="2026-04-26T14:30:00+00:00")

    assert broker.submitted == []


def test_risk_manager_allows_order_when_controls_pass(tmp_path: Path) -> None:
    gateway, broker, store = _gateway(
        tmp_path,
        RiskConfig(max_position_notional=2_000.0, max_daily_loss=1_000.0, max_orders_per_day=2),
    )
    store.set_daily_realized_pnl("2026-04-25", -999.0, updated_at_utc="2026-04-25T12:00:00+00:00")

    ack = gateway.submit_order(
        _request(quantity=10, reference_price=100.0),
        signal_id="signal-1",
        now_utc="2026-04-25T12:00:00+00:00",
    )

    assert ack.status == "accepted"
    assert len(broker.submitted) == 1


def test_risk_config_rejects_invalid_values() -> None:
    with raises(ValueError, match="max_position_notional"):
        RiskConfig(max_position_notional=0.0)
    with raises(ValueError, match="max_position_notional"):
        RiskConfig(max_position_notional=float("nan"))
    with raises(ValueError, match="max_position_notional"):
        RiskConfig(max_position_notional=float("inf"))
    with raises(ValueError, match="max_daily_loss"):
        RiskConfig(max_daily_loss=0.0)
    with raises(ValueError, match="max_daily_loss"):
        RiskConfig(max_daily_loss=float("nan"))
    with raises(ValueError, match="max_daily_loss"):
        RiskConfig(max_daily_loss=float("inf"))
    with raises(ValueError, match="max_orders_per_day"):
        RiskConfig(max_orders_per_day=0)
    with raises(ValueError, match="end_utc"):
        NoTradeWindow("2026-04-25T13:00:00+00:00", "2026-04-25T12:00:00+00:00")
    with raises(ValueError, match="timezone-aware UTC"):
        NoTradeWindow("2026-04-25T12:00:00", "2026-04-25T13:00:00+00:00")
    with raises(ValueError, match="must be UTC"):
        NoTradeWindow("2026-04-25T12:00:00-05:00", "2026-04-25T13:00:00+00:00")
