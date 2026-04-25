from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from statistics import mean, median
from typing import Callable, Iterable, Sequence

from trader.data.models import NEW_YORK
from trader.execution.engine import BacktestResult
from trader.strategies.registry import StrategyRegistry
from trader.strategies.spec import StrategySpec


MONTHLY_PNL_CONCENTRATION_LIMIT_PCT = 80.0


@dataclass(frozen=True)
class RobustnessResult:
    checks: dict[str, float | bool]
    passed: bool


def assess_robustness(
    *,
    spec: StrategySpec,
    aggregate_metrics: dict[str, float],
    fold_metrics: Sequence[dict[str, float]],
    fold_backtests: Sequence[BacktestResult],
    registry: StrategyRegistry,
    neighbor_metric_fn: Callable[[StrategySpec], dict[str, float]],
) -> RobustnessResult:
    fold_returns = [metrics["return_pct"] for metrics in fold_metrics]
    positive_fold_ratio = sum(1 for value in fold_returns if value > 0) / max(len(fold_returns), 1)
    monthly_pnl = _monthly_strategy_pnl_breakdown(fold_backtests)
    positive_monthly_concentration = _monthly_concentration_pct(
        (value for value in monthly_pnl.values() if value > 0),
        no_positive_value=100.0,
    )
    loss_monthly_concentration = _monthly_concentration_pct(
        -value for value in monthly_pnl.values() if value < 0
    )
    monthly_concentration = max(positive_monthly_concentration, loss_monthly_concentration)
    positive_pnl_present = any(value > 0 for value in monthly_pnl.values())
    regime_pass = positive_pnl_present and monthly_concentration <= MONTHLY_PNL_CONCENTRATION_LIMIT_PCT
    neighborhood_returns: list[float] = []
    neighborhood_sharpes: list[float] = []
    for neighbor_spec in registry.neighbors(spec)[:6]:
        try:
            neighbor_metrics = neighbor_metric_fn(neighbor_spec)
            neighborhood_returns.append(neighbor_metrics["return_pct"])
            neighborhood_sharpes.append(neighbor_metrics["sharpe_like"])
        except Exception:
            continue
    neighborhood_return_median = median(neighborhood_returns) if neighborhood_returns else aggregate_metrics["return_pct"]
    neighborhood_sharpe_median = (
        median(neighborhood_sharpes) if neighborhood_sharpes else aggregate_metrics.get("sharpe_like", 0.0)
    )
    neighborhood_return_gap = aggregate_metrics["return_pct"] - neighborhood_return_median
    neighborhood_sharpe_gap = aggregate_metrics.get("sharpe_like", 0.0) - neighborhood_sharpe_median
    # A strategy is neighborhood-stable only if it does not dramatically outperform its
    # parameter neighbors on BOTH return and risk-adjusted return. A spike in return
    # alone might be a lucky period; a spike in Sharpe too signals genuine overfitting.
    neighborhood_pass = neighborhood_return_gap <= 10.0 and neighborhood_sharpe_gap <= 0.5
    checks: dict[str, float | bool] = {
        "positive_fold_ratio": positive_fold_ratio,
        "fold_consistency_pass": positive_fold_ratio >= 0.5,
        "monthly_concentration_pct": monthly_concentration,
        "positive_monthly_pnl_concentration_pct": positive_monthly_concentration,
        "loss_monthly_pnl_concentration_pct": loss_monthly_concentration,
        "positive_monthly_pnl_present": positive_pnl_present,
        "monthly_pnl_month_count": float(len(monthly_pnl)),
        "regime_pass": regime_pass,
        "neighborhood_median_return_pct": neighborhood_return_median,
        "neighborhood_gap_pct": neighborhood_return_gap,  # kept for backward compat
        "neighborhood_return_gap_pct": neighborhood_return_gap,
        "neighborhood_median_sharpe_like": neighborhood_sharpe_median,
        "neighborhood_sharpe_gap": neighborhood_sharpe_gap,
        "neighborhood_pass": neighborhood_pass,
        "drawdown_pass": aggregate_metrics.get("max_drawdown_pct", 100.0) <= 20.0,
    }
    passed = bool(checks["fold_consistency_pass"] and checks["regime_pass"] and checks["neighborhood_pass"] and checks["drawdown_pass"])
    return RobustnessResult(checks=checks, passed=passed)


def _monthly_strategy_pnl_breakdown(backtests: Sequence[BacktestResult]) -> dict[str, float]:
    monthly_pnl: defaultdict[str, float] = defaultdict(float)
    for backtest in backtests:
        for trade in backtest.trades:
            monthly_pnl[_trade_exit_month(trade.exit_timestamp_utc)] += trade.pnl_cash
    return dict(sorted(monthly_pnl.items()))


def _trade_exit_month(timestamp_utc: str) -> str:
    timestamp = datetime.fromisoformat(timestamp_utc.replace("Z", "+00:00"))
    if timestamp.tzinfo is None:
        return timestamp.strftime("%Y-%m")
    return timestamp.astimezone(NEW_YORK).strftime("%Y-%m")


def _monthly_concentration_pct(monthly_values: Iterable[float], *, no_positive_value: float = 0.0) -> float:
    values = tuple(value for value in monthly_values if value > 0)
    total = sum(values)
    if total <= 0:
        return no_positive_value
    return (max(values) / total) * 100.0
