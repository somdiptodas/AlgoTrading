from __future__ import annotations

from dataclasses import dataclass

from trader.evaluation.runner import ExperimentResult


@dataclass(frozen=True)
class Critique:
    verdict: str
    notes: tuple[str, ...]
    next_focus: tuple[str, ...]

    def to_payload(self) -> dict[str, object]:
        return {
            "verdict": self.verdict,
            "notes": list(self.notes),
            "next_focus": list(self.next_focus),
        }


class HeuristicCritic:
    def critique(self, result: ExperimentResult) -> Critique:
        notes: list[str] = []
        next_focus: list[str] = []
        if result.aggregate_metrics.get("return_pct", 0.0) <= 0:
            notes.append("Aggregate OOS return is non-positive.")
            next_focus.append("Shift to alternate family or shorter lookback neighborhood.")
        if result.aggregate_metrics.get("delta_buy_and_hold_return_pct", -999.0) <= 0:
            notes.append("Strategy does not beat buy-and-hold on average OOS return.")
            next_focus.append("Search for lower-drawdown variants before promoting.")
        if not result.robustness_checks.get("neighborhood_pass", False):
            notes.append("Parameter neighborhood appears peaky.")
            next_focus.append("Broaden the search around smoother parameter regions.")
        if not result.robustness_checks.get("fold_consistency_pass", False):
            notes.append("Returns are too concentrated across folds.")
            next_focus.append("Prefer more stable variants over higher headline return.")
        verdict = "promising" if result.promotion_stage in {"frontier", "candidate"} else "fragile"
        if not notes:
            notes.append("Metrics and robustness checks are acceptable for continued search.")
            next_focus.append("Expand the local neighborhood around this spec.")
        return Critique(verdict=verdict, notes=tuple(notes), next_focus=tuple(next_focus))
