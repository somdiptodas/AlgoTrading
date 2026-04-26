from __future__ import annotations


MIN_PROMOTION_TRADE_COUNT = 10.0
MIN_INFORMATION_RATIO_VS_BUY_AND_HOLD = 0.5
_REQUIRED_ROBUSTNESS_CHECKS = (
    "fold_consistency_pass",
    "regime_pass",
    "neighborhood_pass",
    "drawdown_pass",
)


def promotion_stage(
    aggregate_metrics: dict[str, float],
    robustness_checks: dict[str, float | bool],
) -> str:
    if not robustness_checks:
        return "exploratory"

    robust = all(bool(robustness_checks.get(check_name)) for check_name in _REQUIRED_ROBUSTNESS_CHECKS)
    positive_return = aggregate_metrics.get("return_pct", 0.0) > 0.0
    risk_metric = aggregate_metrics.get("annualized_sharpe", aggregate_metrics.get("sharpe_like", 0.0))
    positive_risk_adjusted_return = risk_metric > 0.0
    enough_trades = aggregate_metrics.get("trade_count", 0.0) >= MIN_PROMOTION_TRADE_COUNT
    exposure_adjusted_edge_pct = aggregate_metrics.get("delta_exposure_adjusted_buy_and_hold_pct", -999.0)
    beats_relevant_benchmark = exposure_adjusted_edge_pct > 0.0
    information_ratio = aggregate_metrics.get("information_ratio_vs_buy_and_hold", -999.0)
    consistent_benchmark_edge = information_ratio > MIN_INFORMATION_RATIO_VS_BUY_AND_HOLD
    if not (
        robust
        and positive_return
        and positive_risk_adjusted_return
        and enough_trades
        and beats_relevant_benchmark
        and consistent_benchmark_edge
    ):
        return "exploratory"

    return "candidate"
