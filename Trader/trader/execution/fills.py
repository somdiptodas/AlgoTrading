from __future__ import annotations

from dataclasses import dataclass

from trader.data.models import MarketBar
from trader.execution.position import Position
from trader.strategies.spec import ExecConfig


@dataclass(frozen=True)
class Trade:
    entry_timestamp_utc: str
    exit_timestamp_utc: str
    entry_price: float
    exit_price: float
    shares: int
    bars_held: int
    pnl_cash: float
    pnl_pct: float
    exit_reason: str


def enter_long(cash: float, bar: MarketBar, exec_config: ExecConfig) -> tuple[float, Position | None]:
    fill_price = bar.open * (1.0 + exec_config.slippage_bps / 10_000.0)
    max_shares = int((cash - exec_config.commission_per_order) // fill_price)
    if max_shares < 1:
        return cash, None
    cost = (max_shares * fill_price) + exec_config.commission_per_order
    new_cash = cash - cost
    position = Position(
        entry_timestamp_ms=bar.timestamp_ms,
        entry_timestamp_utc=bar.timestamp_utc,
        entry_price=fill_price,
        shares=max_shares,
        entry_commission=exec_config.commission_per_order,
    )
    return new_cash, position


def exit_long(
    cash: float,
    position: Position,
    bar: MarketBar,
    exec_config: ExecConfig,
    reason: str,
    *,
    fill_at_close: bool,
) -> tuple[float, Trade]:
    raw_price = bar.close if fill_at_close else bar.open
    fill_price = raw_price * (1.0 - exec_config.slippage_bps / 10_000.0)
    proceeds = (position.shares * fill_price) - exec_config.commission_per_order
    new_cash = cash + proceeds
    pnl_cash = (
        (fill_price - position.entry_price) * position.shares
        - position.entry_commission
        - exec_config.commission_per_order
    )
    invested = (position.entry_price * position.shares) + position.entry_commission
    trade = Trade(
        entry_timestamp_utc=position.entry_timestamp_utc,
        exit_timestamp_utc=bar.timestamp_utc,
        entry_price=position.entry_price,
        exit_price=fill_price,
        shares=position.shares,
        bars_held=position.bars_held,
        pnl_cash=pnl_cash,
        pnl_pct=(pnl_cash / invested) * 100 if invested else 0.0,
        exit_reason=reason,
    )
    return new_cash, trade
