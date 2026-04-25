from __future__ import annotations

from types import SimpleNamespace

import pytest

from trader.evaluation.promotion import MIN_PROMOTION_TRADE_COUNT, promotion_stage
from trader.research.critic import HeuristicCritic, planning_penalty_from_critique


def _metrics(**overrides: float) -> dict[str, float]:
    metrics = {
        "return_pct": 1.0,
        "annualized_sharpe": 1.0,
        "sharpe_like": 0.2,
        "trade_count": MIN_PROMOTION_TRADE_COUNT,
        "delta_buy_and_hold_return_pct": 0.25,
    }
    metrics.update(overrides)
    return metrics


def _robustness(**overrides: bool) -> dict[str, bool]:
    checks = {
        "fold_consistency_pass": True,
        "regime_pass": True,
        "neighborhood_pass": True,
        "drawdown_pass": True,
    }
    checks.update(overrides)
    return checks


def test_missing_robustness_stays_exploratory() -> None:
    assert promotion_stage(_metrics(), {}) == "exploratory"


@pytest.mark.parametrize(
    "metric_overrides",
    (
        {"return_pct": 0.0},
        {"annualized_sharpe": 0.0},
        {"trade_count": MIN_PROMOTION_TRADE_COUNT - 1.0},
        {"delta_buy_and_hold_return_pct": 0.0},
        {"delta_buy_and_hold_return_pct": -0.1},
    ),
)
def test_edge_failures_stay_exploratory(metric_overrides: dict[str, float]) -> None:
    assert promotion_stage(_metrics(**metric_overrides), _robustness()) == "exploratory"


@pytest.mark.parametrize(
    "robustness_overrides",
    (
        {"fold_consistency_pass": False},
        {"regime_pass": False},
        {"neighborhood_pass": False},
        {"drawdown_pass": False},
    ),
)
def test_robustness_failures_stay_exploratory(robustness_overrides: dict[str, bool]) -> None:
    assert promotion_stage(_metrics(), _robustness(**robustness_overrides)) == "exploratory"


def test_passing_edge_and_robustness_is_research_frontier() -> None:
    assert promotion_stage(_metrics(delta_buy_and_hold_return_pct=0.25), _robustness()) == "research_frontier"


def test_buy_hold_margin_promotes_candidate() -> None:
    assert promotion_stage(_metrics(delta_buy_and_hold_return_pct=0.6), _robustness()) == "candidate"


def test_legacy_sharpe_like_is_used_when_annualized_sharpe_is_absent() -> None:
    metrics = _metrics()
    del metrics["annualized_sharpe"]

    assert promotion_stage(metrics, _robustness()) == "research_frontier"


def test_critic_does_not_treat_legacy_frontier_as_promising() -> None:
    result = SimpleNamespace(
        aggregate_metrics={"return_pct": 1.0, "delta_buy_and_hold_return_pct": 1.0, "trade_count": 10.0},
        robustness_checks={"fold_consistency_pass": True, "neighborhood_pass": True},
        promotion_stage="frontier",
    )

    assert HeuristicCritic().critique(result).verdict == "fragile"


def test_critic_emits_planning_penalties_for_actionable_notes() -> None:
    result = SimpleNamespace(
        aggregate_metrics={"return_pct": -1.0, "delta_buy_and_hold_return_pct": -2.0, "trade_count": 750.0},
        robustness_checks={"fold_consistency_pass": False, "neighborhood_pass": True},
        promotion_stage="exploratory",
    )

    critique = HeuristicCritic().critique(result)

    assert critique.planning_penalties == {
        "non_positive_return": 12.0,
        "benchmark_failure": 10.0,
        "fold_inconsistency": 8.0,
        "excessive_trading": 6.0,
    }
    assert critique.to_payload()["planning_penalties"] == critique.planning_penalties


def test_planning_penalty_from_critique_supports_legacy_notes() -> None:
    payload = {
        "notes": [
            "Strategy does not beat buy-and-hold on average OOS return.",
            "Returns are too concentrated across folds.",
        ]
    }

    assert planning_penalty_from_critique(payload) == 18.0
