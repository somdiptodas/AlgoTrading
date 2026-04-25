from __future__ import annotations

import math
from statistics import mean
from typing import Sequence

from trader.execution.engine import BacktestResult
from trader.execution.fills import Trade

TRADING_DAYS_PER_YEAR = 252
REGULAR_SESSION_BARS_PER_DAY = 390
MINUTE_BARS_PER_YEAR = TRADING_DAYS_PER_YEAR * REGULAR_SESSION_BARS_PER_DAY


def calculate_metrics(result: BacktestResult) -> dict[str, float]:
    total_return_pct = ((result.final_cash / result.initial_cash) - 1.0) * 100.0
    winners = [trade for trade in result.trades if trade.pnl_cash > 0]
    losers = [trade for trade in result.trades if trade.pnl_cash <= 0]
    gross_profit = sum(trade.pnl_cash for trade in winners)
    gross_loss = abs(sum(trade.pnl_cash for trade in losers))
    cost_drag_cash = sum(trade.cost_cash for trade in result.trades)
    return {
        "return_pct": total_return_pct,
        "annualized_sharpe": annualized_sharpe(result.equity_curve, starting_equity=result.initial_cash),
        "sharpe_like": sharpe_like(result.equity_curve),
        "max_drawdown_pct": max_drawdown_pct(result.equity_curve),
        "exposure_pct": exposure_pct(result.trades, len(result.bars)),
        "trade_count": float(len(result.trades)),
        "win_rate_pct": safe_pct(len(winners), max(len(result.trades), 1)),
        "profit_factor": gross_profit / gross_loss if gross_loss else (float("inf") if gross_profit else 0.0),
        "avg_trade_pct": mean([trade.pnl_pct for trade in result.trades]) if result.trades else 0.0,
        "cost_drag_cash": cost_drag_cash,
        "cost_drag_pct": (cost_drag_cash / result.initial_cash) * 100.0,
    }


def aggregate_metric_dicts(
    metrics: Sequence[dict[str, float]],
    *,
    weights: Sequence[float] | None = None,
) -> dict[str, float]:
    if not metrics:
        return {}
    if weights is None:
        weights = tuple(1.0 for _ in metrics)
    if len(weights) != len(metrics):
        raise ValueError("weights must have the same length as metrics")
    if any(weight < 0 for weight in weights):
        raise ValueError("weights must be non-negative")

    total_weight = sum(weights)
    if math.isclose(total_weight, 0.0):
        raise ValueError("weights must include at least one positive value")
    keys = sorted({key for metric in metrics for key in metric})
    return {
        key: sum(metric[key] * weight for metric, weight in zip(metrics, weights)) / total_weight
        for key in keys
    }


def max_drawdown_pct(equity_curve: Sequence[float]) -> float:
    if not equity_curve:
        return 0.0
    peak = equity_curve[0]
    max_drawdown = 0.0
    for equity in equity_curve:
        peak = max(peak, equity)
        drawdown = 0.0 if peak == 0 else ((peak - equity) / peak) * 100.0
        max_drawdown = max(max_drawdown, drawdown)
    return max_drawdown


def exposure_pct(trades: Sequence[Trade], total_bars: int) -> float:
    if total_bars == 0:
        return 0.0
    return (sum(trade.bars_held for trade in trades) / total_bars) * 100.0


def safe_pct(numerator: int, denominator: int) -> float:
    return (numerator / denominator) * 100.0 if denominator else 0.0


def annualized_sharpe(
    equity_curve: Sequence[float],
    *,
    starting_equity: float | None = None,
    periods_per_year: int = MINUTE_BARS_PER_YEAR,
) -> float:
    returns = _period_returns(equity_curve, starting_equity=starting_equity)
    return annualized_sharpe_from_returns(returns, periods_per_year=periods_per_year)


def annualized_sharpe_for_backtests(
    results: Sequence[BacktestResult],
    *,
    periods_per_year: int = MINUTE_BARS_PER_YEAR,
) -> float:
    returns: list[float] = []
    for result in results:
        returns.extend(_period_returns(result.equity_curve, starting_equity=result.initial_cash))
    return annualized_sharpe_from_returns(returns, periods_per_year=periods_per_year)


def annualized_sharpe_from_returns(
    returns: Sequence[float],
    *,
    periods_per_year: int = MINUTE_BARS_PER_YEAR,
) -> float:
    if not returns:
        return 0.0
    avg_return = mean(returns)
    variance = mean([(value - avg_return) ** 2 for value in returns])
    if math.isclose(variance, 0.0):
        return 0.0
    return math.sqrt(periods_per_year) * avg_return / math.sqrt(variance)


def sharpe_like(equity_curve: Sequence[float]) -> float:
    returns = _period_returns(equity_curve)
    if not returns:
        return 0.0
    avg_return = mean(returns)
    variance = mean([(value - avg_return) ** 2 for value in returns])
    if math.isclose(variance, 0.0):
        return 0.0
    return math.sqrt(len(returns)) * avg_return / math.sqrt(variance)


def _period_returns(equity_curve: Sequence[float], *, starting_equity: float | None = None) -> list[float]:
    if len(equity_curve) < 2:
        if starting_equity is None or not equity_curve:
            return []
    values = list(equity_curve)
    if starting_equity is not None:
        values.insert(0, starting_equity)
    returns = []
    for previous, current in zip(values, values[1:]):
        if previous == 0:
            returns.append(0.0)
        else:
            returns.append((current / previous) - 1.0)
    return returns
