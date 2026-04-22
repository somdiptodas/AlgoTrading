from __future__ import annotations

import math
from statistics import mean
from typing import Iterable, Sequence

from trader.execution.engine import BacktestResult
from trader.execution.fills import Trade


def calculate_metrics(result: BacktestResult) -> dict[str, float]:
    total_return_pct = ((result.final_cash / result.initial_cash) - 1.0) * 100.0
    winners = [trade for trade in result.trades if trade.pnl_cash > 0]
    losers = [trade for trade in result.trades if trade.pnl_cash <= 0]
    gross_profit = sum(trade.pnl_cash for trade in winners)
    gross_loss = abs(sum(trade.pnl_cash for trade in losers))
    return {
        "return_pct": total_return_pct,
        "sharpe_like": sharpe_like(result.equity_curve),
        "max_drawdown_pct": max_drawdown_pct(result.equity_curve),
        "exposure_pct": exposure_pct(result.trades, len(result.bars)),
        "trade_count": float(len(result.trades)),
        "win_rate_pct": safe_pct(len(winners), max(len(result.trades), 1)),
        "profit_factor": gross_profit / gross_loss if gross_loss else (float("inf") if gross_profit else 0.0),
        "avg_trade_pct": mean([trade.pnl_pct for trade in result.trades]) if result.trades else 0.0,
    }


def aggregate_metric_dicts(metrics: Sequence[dict[str, float]]) -> dict[str, float]:
    if not metrics:
        return {}
    keys = sorted({key for metric in metrics for key in metric})
    return {key: mean([metric[key] for metric in metrics]) for key in keys}


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


def sharpe_like(equity_curve: Sequence[float]) -> float:
    if len(equity_curve) < 2:
        return 0.0
    returns = []
    for previous, current in zip(equity_curve, equity_curve[1:]):
        if previous == 0:
            returns.append(0.0)
        else:
            returns.append((current / previous) - 1.0)
    if not returns:
        return 0.0
    avg_return = mean(returns)
    variance = mean([(value - avg_return) ** 2 for value in returns])
    if math.isclose(variance, 0.0):
        return 0.0
    return math.sqrt(len(returns)) * avg_return / math.sqrt(variance)
