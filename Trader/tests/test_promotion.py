from __future__ import annotations

from types import SimpleNamespace

import pytest

from trader.evaluation.promotion import (
    MIN_INFORMATION_RATIO_VS_BUY_AND_HOLD,
    MIN_PROMOTION_TRADE_COUNT,
    promotion_stage,
)
from trader.research.critic import HeuristicCritic, planning_penalty_from_critique


def _metrics(**overrides: float) -> dict[str, float]:
    metrics = {
        "return_pct": 1.0,
        "annualized_sharpe": 1.0,
        "sharpe_like": 0.2,
        "trade_count": MIN_PROMOTION_TRADE_COUNT,
        "delta_buy_and_hold_return_pct": -10.0,
        "delta_exposure_adjusted_buy_and_hold_pct": 0.25,
        "information_ratio_vs_buy_and_hold": MIN_INFORMATION_RATIO_VS_BUY_AND_HOLD + 0.1,
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
        {"delta_exposure_adjusted_buy_and_hold_pct": 0.0},
        {"delta_exposure_adjusted_buy_and_hold_pct": -0.1},
        {"information_ratio_vs_buy_and_hold": MIN_INFORMATION_RATIO_VS_BUY_AND_HOLD},
        {"information_ratio_vs_buy_and_hold": MIN_INFORMATION_RATIO_VS_BUY_AND_HOLD - 0.1},
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


def test_exposure_adjusted_edge_promotes_candidate_even_when_raw_buy_hold_delta_lags() -> None:
    assert promotion_stage(
        _metrics(delta_buy_and_hold_return_pct=-10.0, delta_exposure_adjusted_buy_and_hold_pct=0.25),
        _robustness(),
    ) == "candidate"


def test_raw_buy_hold_margin_no_longer_promotes_without_exposure_adjusted_edge() -> None:
    assert promotion_stage(
        _metrics(delta_buy_and_hold_return_pct=0.6, delta_exposure_adjusted_buy_and_hold_pct=0.0),
        _robustness(),
    ) == "exploratory"


def test_legacy_sharpe_like_is_used_when_annualized_sharpe_is_absent() -> None:
    metrics = _metrics()
    del metrics["annualized_sharpe"]

    assert promotion_stage(metrics, _robustness()) == "candidate"


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


def test_critic_uses_exposure_adjusted_benchmark_when_present() -> None:
    result = SimpleNamespace(
        aggregate_metrics={
            "return_pct": 1.0,
            "delta_buy_and_hold_return_pct": -10.0,
            "delta_exposure_adjusted_buy_and_hold_pct": 0.5,
            "trade_count": 10.0,
        },
        robustness_checks={"fold_consistency_pass": True, "neighborhood_pass": True},
        promotion_stage="candidate",
    )

    critique = HeuristicCritic().critique(result)

    assert "benchmark_failure" not in critique.planning_penalties


def test_planning_penalty_from_critique_supports_legacy_notes() -> None:
    payload = {
        "notes": [
            "Strategy does not beat buy-and-hold on average OOS return.",
            "Returns are too concentrated across folds.",
        ]
    }

    assert planning_penalty_from_critique(payload) == 18.0
