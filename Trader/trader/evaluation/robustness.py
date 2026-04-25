from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, median
from typing import Callable, Sequence

from trader.data.models import MarketBar
from trader.strategies.registry import StrategyRegistry
from trader.strategies.spec import StrategySpec


@dataclass(frozen=True)
class RobustnessResult:
    checks: dict[str, float | bool]
    passed: bool


def assess_robustness(
    *,
    spec: StrategySpec,
    aggregate_metrics: dict[str, float],
    fold_metrics: Sequence[dict[str, float]],
    full_test_bars: tuple[MarketBar, ...],
    registry: StrategyRegistry,
    neighbor_metric_fn: Callable[[StrategySpec], dict[str, float]],
) -> RobustnessResult:
    fold_returns = [metrics["return_pct"] for metrics in fold_metrics]
    positive_fold_ratio = sum(1 for value in fold_returns if value > 0) / max(len(fold_returns), 1)
    monthly_returns = _monthly_return_breakdown(full_test_bars)
    monthly_concentration = max((abs(value) for value in monthly_returns.values()), default=0.0)
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
        "regime_pass": monthly_concentration < 15.0,
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


def _monthly_return_breakdown(bars: Sequence[MarketBar]) -> dict[str, float]:
    if len(bars) < 2:
        return {}
    by_month: dict[str, list[MarketBar]] = {}
    for bar in bars:
        key = bar.dt_local.strftime("%Y-%m")
        by_month.setdefault(key, []).append(bar)
    monthly_returns: dict[str, float] = {}
    for key, month_bars in by_month.items():
        start_close = month_bars[0].close
        if start_close == 0:
            monthly_returns[key] = 0.0
        else:
            monthly_returns[key] = ((month_bars[-1].close / start_close) - 1.0) * 100.0
    return monthly_returns
