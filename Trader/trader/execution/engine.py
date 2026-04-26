from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from trader.data.models import REGULAR_SESSION_END, REGULAR_SESSION_START, MarketBar
from trader.execution.fills import Trade, enter_long, exit_long, exit_long_at_price
from trader.execution.position import Position
from trader.strategies.decisions import RuleDecision, TradeDecision
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
                if entry_allowed(bar, exec_config):
                    cash, position = enter_long(cash, bar, exec_config, sizing_fraction, reason)
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
            elif not entry_allowed(bars[index + 1], exec_config):
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


def run_long_only_decision_engine(
    bars: Sequence[MarketBar],
    decisions_by_bar: Sequence[TradeDecision],
    exec_config: ExecConfig,
    sizing_fraction: float = 1.0,
) -> BacktestResult:
    if len(bars) != len(decisions_by_bar):
        raise ValueError("bars and decisions_by_bar must have identical lengths")
    if not bars:
        raise RuntimeError("No bars available for simulation")

    cash = exec_config.initial_cash
    equity_curve: list[float] = []
    trades: list[Trade] = []
    position: Position | None = None
    pending_action: tuple[str, RuleDecision] | None = None

    for index, bar in enumerate(bars):
        if pending_action is not None:
            action, decision = pending_action
            if action == "enter":
                if entry_allowed(bar, exec_config):
                    cash, position = enter_long(cash, bar, exec_config, sizing_fraction, decision.reason)
            elif action == "exit" and position is not None:
                stop_fill_reference = stop_loss_fill_reference(position, bar, exec_config)
                if stop_fill_reference is not None:
                    cash, trade = exit_long_at_price(cash, position, bar, exec_config, "stop_loss", stop_fill_reference)
                else:
                    cash, trade = exit_long(cash, position, bar, exec_config, decision.reason, fill_at_close=False)
                trades.append(trade)
                position = None
            pending_action = None

        if position is not None:
            position.bars_held += 1

        decision = decisions_by_bar[index]
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
        elif position is not None and decision.exit.passed and not last_bar:
            pending_action = ("exit", decision.exit)
        elif position is None and decision.entry.passed and not last_bar:
            if exec_config.flat_at_close and next_session_changes:
                pass
            elif not entry_allowed(bars[index + 1], exec_config):
                pass
            else:
                pending_action = ("enter", decision.entry)

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


def entry_allowed(bar: MarketBar, exec_config: ExecConfig) -> bool:
    minutes_since_open = _minutes_since_session_open(bar)
    minutes_until_close = _minutes_until_session_close(bar)
    if exec_config.no_new_entry_minutes_before_close is not None:
        if minutes_until_close <= exec_config.no_new_entry_minutes_before_close:
            return False
    if exec_config.entry_session_window == "first_30m":
        return 0 <= minutes_since_open < 30
    if exec_config.entry_session_window == "last_30m":
        return 0 < minutes_until_close <= 30
    if exec_config.entry_session_window == "avoid_midday":
        return not (120 <= minutes_since_open < 240)
    return True


def _minutes_since_session_open(bar: MarketBar) -> int:
    local = bar.dt_local
    return (local.hour * 60 + local.minute) - (REGULAR_SESSION_START.hour * 60 + REGULAR_SESSION_START.minute)


def _minutes_until_session_close(bar: MarketBar) -> int:
    local = bar.dt_local
    return (REGULAR_SESSION_END.hour * 60 + REGULAR_SESSION_END.minute) - (local.hour * 60 + local.minute)
