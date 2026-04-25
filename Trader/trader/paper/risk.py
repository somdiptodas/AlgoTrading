from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from math import isfinite
from zoneinfo import ZoneInfo

from trader.paper.models import BrokerPosition, OrderRequest
from trader.paper.store import PaperTradingStore


NEW_YORK = ZoneInfo("America/New_York")


class RiskControlViolation(ValueError):
    pass


@dataclass(frozen=True)
class NoTradeWindow:
    start_utc: str
    end_utc: str
    reason: str = "event"

    def __post_init__(self) -> None:
        start = _parse_aware_utc(self.start_utc, "start_utc")
        end = _parse_aware_utc(self.end_utc, "end_utc")
        if end <= start:
            raise ValueError("no-trade window end_utc must be after start_utc")


@dataclass(frozen=True)
class RiskConfig:
    max_position_notional: float | None = None
    max_daily_loss: float | None = None
    max_orders_per_day: int | None = None
    no_trade_dates: tuple[str, ...] = ()
    no_trade_windows: tuple[NoTradeWindow, ...] = ()

    def __post_init__(self) -> None:
        if self.max_position_notional is not None and not _is_positive_finite(self.max_position_notional):
            raise ValueError("max_position_notional must be > 0 when set")
        if self.max_daily_loss is not None and not _is_positive_finite(self.max_daily_loss):
            raise ValueError("max_daily_loss must be > 0 when set")
        if self.max_orders_per_day is not None and self.max_orders_per_day < 1:
            raise ValueError("max_orders_per_day must be >= 1 when set")


@dataclass(frozen=True)
class RiskDecision:
    allowed: bool
    reason: str = ""


class RiskManager:
    def __init__(self, config: RiskConfig) -> None:
        self.config = config

    def assess_order(
        self,
        request: OrderRequest,
        *,
        now_utc: str,
        store: PaperTradingStore,
        positions: tuple[BrokerPosition, ...] = (),
    ) -> RiskDecision:
        kill_enabled, kill_reason = store.get_kill_switch()
        if kill_enabled:
            return RiskDecision(False, f"kill_switch:{kill_reason}" if kill_reason else "kill_switch")
        trading_day = _trading_day(now_utc)
        if trading_day in set(self.config.no_trade_dates):
            return RiskDecision(False, "no_trade_date")
        if any(_contains(window, now_utc) for window in self.config.no_trade_windows):
            return RiskDecision(False, "no_trade_window")
        if self.config.max_daily_loss is not None:
            realized_pnl = store.get_daily_realized_pnl(trading_day)
            if realized_pnl <= -abs(self.config.max_daily_loss):
                return RiskDecision(False, "max_daily_loss")
        if self.config.max_orders_per_day is not None:
            order_count = store.accepted_order_count_for_trading_day(trading_day)
            if order_count >= self.config.max_orders_per_day:
                return RiskDecision(False, "max_orders_per_day")
        if self.config.max_position_notional is not None:
            try:
                price = _risk_price(request)
            except RiskControlViolation as exc:
                return RiskDecision(False, str(exc))
            if not _projected_notional_allowed(request, positions, price, self.config.max_position_notional):
                return RiskDecision(False, "max_position_notional")
        return RiskDecision(True)


def _projected_notional_allowed(
    request: OrderRequest,
    positions: tuple[BrokerPosition, ...],
    price: float,
    max_position_notional: float,
) -> bool:
    current_quantity = sum(position.quantity for position in positions if position.symbol.upper() == request.symbol.upper())
    signed_order_quantity = request.quantity if request.side == "buy" else -request.quantity
    projected_quantity = current_quantity + signed_order_quantity
    current_notional = abs(current_quantity) * price
    projected_notional = abs(projected_quantity) * price
    if projected_notional <= current_notional:
        return True
    return projected_notional <= max_position_notional


def _risk_price(request: OrderRequest) -> float:
    price = request.limit_price if request.limit_price is not None else request.reference_price
    if price is None:
        raise RiskControlViolation("reference_price is required for max_position_notional checks")
    if price <= 0:
        raise RiskControlViolation("risk price must be > 0")
    return price


def _contains(window: NoTradeWindow, timestamp_utc: str) -> bool:
    timestamp = _parse_aware_utc(timestamp_utc, "timestamp_utc")
    start = _parse_aware_utc(window.start_utc, "start_utc")
    end = _parse_aware_utc(window.end_utc, "end_utc")
    return start <= timestamp < end


def _trading_day(timestamp_utc: str) -> str:
    return _parse_aware_utc(timestamp_utc, "timestamp_utc").astimezone(NEW_YORK).date().isoformat()


def _is_positive_finite(value: float) -> bool:
    return isfinite(value) and value > 0


def _parse_aware_utc(value: str, field_name: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ValueError(f"{field_name} must be timezone-aware UTC")
    if parsed.utcoffset().total_seconds() != 0:
        raise ValueError(f"{field_name} must be UTC")
    return parsed
