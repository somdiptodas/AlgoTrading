from __future__ import annotations

from statistics import mean

from trader.data.models import MarketBar
from trader.evaluation.metrics import max_drawdown_pct, sharpe_like


def evaluate_baselines(bars: tuple[MarketBar, ...], initial_cash: float) -> dict[str, dict[str, float]]:
    return {
        "always_flat": always_flat(initial_cash),
        "buy_and_hold": buy_and_hold(bars, initial_cash),
    }


def always_flat(initial_cash: float) -> dict[str, float]:
    return {
        "return_pct": 0.0,
        "sharpe_like": 0.0,
        "max_drawdown_pct": 0.0,
        "exposure_pct": 0.0,
        "trade_count": 0.0,
        "win_rate_pct": 0.0,
        "profit_factor": 0.0,
        "avg_trade_pct": 0.0,
        "final_cash": initial_cash,
    }


def buy_and_hold(bars: tuple[MarketBar, ...], initial_cash: float) -> dict[str, float]:
    if not bars:
        return always_flat(initial_cash)
    starting_close = bars[0].close
    equity_curve = tuple(initial_cash * (bar.close / starting_close) for bar in bars)
    total_return_pct = ((bars[-1].close / starting_close) - 1.0) * 100.0
    return {
        "return_pct": total_return_pct,
        "sharpe_like": sharpe_like(equity_curve),
        "max_drawdown_pct": max_drawdown_pct(equity_curve),
        "exposure_pct": 100.0,
        "trade_count": 1.0,
        "win_rate_pct": 100.0 if total_return_pct > 0 else 0.0,
        "profit_factor": float("inf") if total_return_pct > 0 else 0.0,
        "avg_trade_pct": total_return_pct,
        "final_cash": equity_curve[-1],
    }


def baseline_deltas(metrics: dict[str, float], baselines: dict[str, dict[str, float]]) -> dict[str, float]:
    deltas: dict[str, float] = {}
    for baseline_name, baseline_metrics in baselines.items():
        deltas[f"delta_{baseline_name}_return_pct"] = metrics["return_pct"] - baseline_metrics["return_pct"]
        deltas[f"delta_{baseline_name}_sharpe_like"] = metrics["sharpe_like"] - baseline_metrics["sharpe_like"]
    return deltas
