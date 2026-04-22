from __future__ import annotations


def promotion_stage(
    aggregate_metrics: dict[str, float],
    robustness_checks: dict[str, float | bool],
) -> str:
    if not robustness_checks:
        return "exploratory"
    beats_buy_hold = aggregate_metrics.get("delta_buy_and_hold_return_pct", -999.0) > 0.5
    positive_sharpe = aggregate_metrics.get("sharpe_like", 0.0) > 0
    if bool(robustness_checks.get("fold_consistency_pass")) and bool(robustness_checks.get("neighborhood_pass")):
        if beats_buy_hold and positive_sharpe:
            return "candidate"
        return "frontier"
    return "exploratory"
