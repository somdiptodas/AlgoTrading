from __future__ import annotations

from dataclasses import dataclass

from trader.evaluation.runner import ExperimentResult

_PLANNING_NOTE_PENALTIES = {
    "Aggregate OOS return is non-positive.": 12.0,
    "Strategy does not beat buy-and-hold on average OOS return.": 10.0,
    "Parameter neighborhood appears peaky.": 8.0,
    "Returns are too concentrated across folds.": 8.0,
    "Trade count is high enough to increase cost sensitivity.": 6.0,
}
_MAX_PLANNING_PENALTY = 25.0


@dataclass(frozen=True)
class Critique:
    verdict: str
    notes: tuple[str, ...]
    next_focus: tuple[str, ...]
    planning_penalties: dict[str, float]

    def to_payload(self) -> dict[str, object]:
        return {
            "verdict": self.verdict,
            "notes": list(self.notes),
            "next_focus": list(self.next_focus),
            "planning_penalties": dict(self.planning_penalties),
        }


def planning_penalty_from_critique(payload: dict[str, object] | None) -> float:
    if not payload:
        return 0.0
    penalties = payload.get("planning_penalties")
    if isinstance(penalties, dict):
        total = sum(float(value) for value in penalties.values() if isinstance(value, (int, float)))
        return min(total, _MAX_PLANNING_PENALTY)
    notes = payload.get("notes")
    if not isinstance(notes, list):
        return 0.0
    total = sum(_PLANNING_NOTE_PENALTIES.get(str(note), 0.0) for note in notes)
    return min(total, _MAX_PLANNING_PENALTY)


class HeuristicCritic:
    def critique(self, result: ExperimentResult) -> Critique:
        notes: list[str] = []
        next_focus: list[str] = []
        planning_penalties: dict[str, float] = {}
        if result.aggregate_metrics.get("return_pct", 0.0) <= 0:
            notes.append("Aggregate OOS return is non-positive.")
            next_focus.append("Shift to alternate family or shorter lookback neighborhood.")
            planning_penalties["non_positive_return"] = 12.0
        if result.aggregate_metrics.get("delta_buy_and_hold_return_pct", -999.0) <= 0:
            notes.append("Strategy does not beat buy-and-hold on average OOS return.")
            next_focus.append("Search for lower-drawdown variants before promoting.")
            planning_penalties["benchmark_failure"] = 10.0
        if not result.robustness_checks.get("neighborhood_pass", False):
            notes.append("Parameter neighborhood appears peaky.")
            next_focus.append("Broaden the search around smoother parameter regions.")
            planning_penalties["peaky_neighborhood"] = 8.0
        if not result.robustness_checks.get("fold_consistency_pass", False):
            notes.append("Returns are too concentrated across folds.")
            next_focus.append("Prefer more stable variants over higher headline return.")
            planning_penalties["fold_inconsistency"] = 8.0
        if result.aggregate_metrics.get("trade_count", 0.0) > 500:
            notes.append("Trade count is high enough to increase cost sensitivity.")
            next_focus.append("Prefer lower-turnover variants before adding more trading rules.")
            planning_penalties["excessive_trading"] = 6.0
        verdict = "promising" if result.promotion_stage in {"research_frontier", "candidate"} else "fragile"
        if not notes:
            notes.append("Metrics and robustness checks are acceptable for continued search.")
            next_focus.append("Expand the local neighborhood around this spec.")
        return Critique(
            verdict=verdict,
            notes=tuple(notes),
            next_focus=tuple(next_focus),
            planning_penalties=planning_penalties,
        )
