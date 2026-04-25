from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from typing import Protocol, Sequence

from trader.paper.models import BrokerPosition, Fill, OrderAck, OrderRequest


class PaperBroker(Protocol):
    def submit_order(self, request: OrderRequest) -> OrderAck:
        """Submit an order using request.client_order_id as the idempotency key."""

    def cancel_order(self, client_order_id: str) -> OrderAck:
        """Cancel an open order by client id."""

    def list_open_orders(self) -> Sequence[OrderAck]:
        """Return currently open broker orders."""

    def list_positions(self) -> Sequence[BrokerPosition]:
        """Return current broker positions."""

    def list_fills(self, *, since_utc: str | None = None) -> Sequence[Fill]:
        """Return broker fills, optionally bounded by an ISO UTC timestamp."""


def client_order_id(
    *,
    strategy_id: str,
    signal_id: str,
    symbol: str,
    side: str,
) -> str:
    payload = "|".join([strategy_id, signal_id, symbol.upper(), side])
    return sha256(payload.encode("utf-8")).hexdigest()[:24]


def with_client_order_id(request: OrderRequest, *, signal_id: str) -> OrderRequest:
    validate_order_request(request)
    if request.client_order_id:
        return request
    if not request.strategy_id:
        raise ValueError("strategy_id is required when client_order_id is not provided")
    return replace(
        request,
        symbol=request.symbol.upper(),
        client_order_id=client_order_id(
            strategy_id=request.strategy_id,
            signal_id=signal_id,
            symbol=request.symbol,
            side=request.side,
        ),
    )


def validate_order_request(request: OrderRequest) -> None:
    if not request.symbol:
        raise ValueError("symbol is required")
    if request.quantity < 1:
        raise ValueError("quantity must be >= 1")
    if request.order_type == "limit" and request.limit_price is None:
        raise ValueError("limit orders require limit_price")
    if request.limit_price is not None and request.limit_price <= 0:
        raise ValueError("limit_price must be > 0 when set")
