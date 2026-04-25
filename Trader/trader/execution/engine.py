from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from trader.data.models import MarketBar
from trader.execution.fills import Trade, enter_long, exit_long, exit_long_at_price
from trader.execution.position import Position
from trader.strategies.spec import ExecConfig


@dataclass(frozen=True)
class BacktestResult:
    bars: tuple[MarketBar, ...]
    trades: tuple[Trade, ...]
    equity_curve: tuple[float, ...]
    initial_cash: float
    final_cash: float


def run_long_only_engine(
    bars: Sequence[MarketBar],
    regime_by_bar: Sequence[bool],
    exec_config: ExecConfig,
    sizing_fraction: float = 1.0,
) -> BacktestResult:
    if len(bars) != len(regime_by_bar):
        raise ValueError("bars and regime_by_bar must have identical lengths")
    if not bars:
        raise RuntimeError("No bars available for simulation")

    cash = exec_config.initial_cash
    equity_curve: list[float] = []
    trades: list[Trade] = []
    position: Position | None = None
    pending_action: tuple[str, str] | None = None

    for index, bar in enumerate(bars):
        if pending_action is not None:
            action, reason = pending_action
            if action == "enter":
                cash, position = enter_long(cash, bar, exec_config, sizing_fraction)
            elif action == "exit" and position is not None:
                stop_fill_reference = stop_loss_fill_reference(position, bar, exec_config)
                if stop_fill_reference is not None:
                    cash, trade = exit_long_at_price(cash, position, bar, exec_config, "stop_loss", stop_fill_reference)
                else:
                    cash, trade = exit_long(cash, position, bar, exec_config, reason, fill_at_close=False)
                trades.append(trade)
                position = None
            pending_action = None

        if position is not None:
            position.bars_held += 1

        regime_is_long = bool(regime_by_bar[index])
        last_bar = index == len(bars) - 1
        next_session_changes = (not last_bar) and bars[index + 1].session_date != bar.session_date

        if position is not None and exec_config.stop_loss_bps is not None:
            stop_fill_reference = stop_loss_fill_reference(position, bar, exec_config)
            if stop_fill_reference is not None:
                cash, trade = exit_long_at_price(cash, position, bar, exec_config, "stop_loss", stop_fill_reference)
                trades.append(trade)
                position = None
                pending_action = None

        if position is not None and exec_config.flat_at_close and (next_session_changes or last_bar):
            cash, trade = exit_long(cash, position, bar, exec_config, "session_close", fill_at_close=True)
            trades.append(trade)
            position = None
        elif position is not None and not regime_is_long and not last_bar:
            pending_action = ("exit", "signal_flip")
        elif position is None and regime_is_long and not last_bar:
            if exec_config.flat_at_close and next_session_changes:
                pass
            else:
                pending_action = ("enter", "signal_on")

        equity_curve.append(mark_to_market(cash, position, bar.close))

    if position is not None:
        cash, trade = exit_long(cash, position, bars[-1], exec_config, "final_bar", fill_at_close=True)
        trades.append(trade)
        position = None
        equity_curve[-1] = cash

    return BacktestResult(
        bars=tuple(bars),
        trades=tuple(trades),
        equity_curve=tuple(equity_curve),
        initial_cash=exec_config.initial_cash,
        final_cash=cash,
    )


def mark_to_market(cash: float, position: Position | None, close_price: float) -> float:
    if position is None:
        return cash
    return cash + (position.shares * close_price)


def stop_loss_fill_reference(position: Position, bar: MarketBar, exec_config: ExecConfig) -> float | None:
    if exec_config.stop_loss_bps is None:
        return None
    stop_price = position.entry_price * (1.0 - exec_config.stop_loss_bps / 10_000.0)
    if bar.low > stop_price:
        return None
    return bar.open if bar.open <= stop_price else stop_price
