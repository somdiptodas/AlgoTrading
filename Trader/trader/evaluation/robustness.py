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
    neighborhood_returns = []
    for neighbor_spec in registry.neighbors(spec)[:6]:
        try:
            neighborhood_returns.append(neighbor_metric_fn(neighbor_spec)["return_pct"])
        except Exception:
            continue
    neighborhood_median = median(neighborhood_returns) if neighborhood_returns else aggregate_metrics["return_pct"]
    neighborhood_gap = aggregate_metrics["return_pct"] - neighborhood_median
    checks: dict[str, float | bool] = {
        "positive_fold_ratio": positive_fold_ratio,
        "fold_consistency_pass": positive_fold_ratio >= 0.5,
        "monthly_concentration_pct": monthly_concentration,
        "regime_pass": monthly_concentration < 15.0,
        "neighborhood_median_return_pct": neighborhood_median,
        "neighborhood_gap_pct": neighborhood_gap,
        "neighborhood_pass": neighborhood_gap <= 10.0,
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
