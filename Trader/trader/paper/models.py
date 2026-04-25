from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


OrderSide = Literal["buy", "sell"]
OrderStatus = Literal["submitted", "accepted", "filled", "canceled", "rejected"]
OrderType = Literal["market", "limit"]
TimeInForce = Literal["day", "gtc"]


@dataclass(frozen=True)
class OrderRequest:
    symbol: str
    side: OrderSide
    quantity: int
    order_type: OrderType = "market"
    time_in_force: TimeInForce = "day"
    client_order_id: str = ""
    strategy_id: str = ""
    limit_price: float | None = None


@dataclass(frozen=True)
class OrderAck:
    client_order_id: str
    broker_order_id: str
    status: OrderStatus
    submitted_at_utc: str
    message: str = ""


@dataclass(frozen=True)
class Fill:
    client_order_id: str
    broker_order_id: str
    symbol: str
    side: OrderSide
    quantity: int
    fill_price: float
    filled_at_utc: str


@dataclass(frozen=True)
class BrokerPosition:
    symbol: str
    quantity: int
    average_price: float = 0.0


@dataclass(frozen=True)
class ExpectedPosition:
    symbol: str
    quantity: int
    average_price: float = 0.0
