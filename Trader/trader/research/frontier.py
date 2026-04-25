from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from trader.evaluation.runner import ExperimentResult
from trader.strategies.spec import StrategySpec

_STAGE_RANK = {"candidate": 3, "research_frontier": 2, "frontier": 1, "exploratory": 1}


@dataclass(frozen=True)
class FrontierRecord:
    experiment_id: str
    spec: StrategySpec
    family: str
    score_vector: dict[str, float]
    dominates_baseline: bool
    stability_pass: bool
    promotion_stage: str


class FrontierManager:
    def __init__(self, limit: int = 10) -> None:
        self.limit = limit

    def rank(self, results: Iterable[ExperimentResult]) -> tuple[FrontierRecord, ...]:
        frontier = [
            FrontierRecord(
                experiment_id=result.experiment_id,
                spec=result.spec,
                family=result.spec.signal.name,
                score_vector={
                    "return_pct": result.aggregate_metrics.get("return_pct", 0.0),
                    "sharpe_like": result.aggregate_metrics.get("sharpe_like", 0.0),
                    "max_drawdown_pct": result.aggregate_metrics.get("max_drawdown_pct", 0.0),
                },
                dominates_baseline=result.aggregate_metrics.get("delta_buy_and_hold_return_pct", -999.0) > 0.0,
                stability_pass=bool(result.robustness_checks.get("fold_consistency_pass")),
                promotion_stage=result.promotion_stage,
            )
            for result in results
        ]
        ranked = sorted(
            frontier,
            key=lambda item: (
                _STAGE_RANK.get(item.promotion_stage, 0),
                item.stability_pass,
                item.dominates_baseline,
                item.score_vector["sharpe_like"],
                item.score_vector["return_pct"],
                -item.score_vector["max_drawdown_pct"],
                item.experiment_id,
            ),
            reverse=True,
        )
        return tuple(ranked[: self.limit])
