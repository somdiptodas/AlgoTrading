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
    cost_cash: float = 0.0


def enter_long(
    cash: float,
    bar: MarketBar,
    exec_config: ExecConfig,
    sizing_fraction: float = 1.0,
) -> tuple[float, Position | None]:
    entry_bps = exec_config.slippage_bps + (exec_config.spread_bps / 2.0)
    fill_price = bar.open * (1.0 + entry_bps / 10_000.0)
    deployable = cash * sizing_fraction
    if exec_config.max_position_notional is not None:
        deployable = min(deployable, exec_config.max_position_notional)
    per_share_cost = fill_price + exec_config.commission_per_share
    max_shares = int((deployable - exec_config.commission_per_order) // per_share_cost)
    if max_shares < 1:
        return cash, None
    commission = exec_config.commission_per_order + (max_shares * exec_config.commission_per_share)
    cost = (max_shares * fill_price) + commission
    new_cash = cash - cost
    position = Position(
        entry_timestamp_ms=bar.timestamp_ms,
        entry_timestamp_utc=bar.timestamp_utc,
        entry_reference_price=bar.open,
        entry_price=fill_price,
        shares=max_shares,
        entry_commission=commission,
        entry_cost_cash=((fill_price - bar.open) * max_shares) + commission,
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
    exit_bps = exec_config.slippage_bps + (exec_config.spread_bps / 2.0)
    fill_price = raw_price * (1.0 - exit_bps / 10_000.0)
    exit_commission = exec_config.commission_per_order + (position.shares * exec_config.commission_per_share)
    proceeds = (position.shares * fill_price) - exit_commission
    new_cash = cash + proceeds
    exit_cost_cash = ((raw_price - fill_price) * position.shares) + exit_commission
    pnl_cash = (
        (fill_price - position.entry_price) * position.shares
        - position.entry_commission
        - exit_commission
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
        cost_cash=position.entry_cost_cash + exit_cost_cash,
    )
    return new_cash, trade
