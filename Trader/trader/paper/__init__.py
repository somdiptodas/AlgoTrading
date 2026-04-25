"""Paper-trading adapter boundaries."""

from trader.paper.broker import PaperBroker, client_order_id, with_client_order_id
from trader.paper.gateway import PaperTradingGateway
from trader.paper.models import BrokerPosition, ExpectedPosition, Fill, OrderAck, OrderRequest
from trader.paper.reconcile import PositionMismatch, ReconciliationReport, reconcile_positions
from trader.paper.store import PaperTradingStore

__all__ = [
    "BrokerPosition",
    "ExpectedPosition",
    "Fill",
    "OrderAck",
    "OrderRequest",
    "PaperBroker",
    "PaperTradingGateway",
    "PaperTradingStore",
    "PositionMismatch",
    "ReconciliationReport",
    "client_order_id",
    "reconcile_positions",
    "with_client_order_id",
]
