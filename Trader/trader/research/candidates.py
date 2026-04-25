from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

from trader.evaluation.runner import EvaluationPreview, EvaluationRunner
from trader.ledger.entry import LedgerEntry
from trader.research.critic import planning_penalty_from_critique
from trader.research.planner import PlannedSpec
from trader.research.suppressor import RegionSuppressor, SuppressedSpec, spec_distance
from trader.strategies.spec import StrategySpec


@dataclass(frozen=True)
class ScoredCandidate:
    planned: PlannedSpec
    preview: EvaluationPreview
    family: str
    static_score: float
    novelty_to_history: float
    parent_score: float
    suppression_record: SuppressedSpec | None = None


@dataclass(frozen=True)
class CandidateQueueResult:
    selected: tuple[ScoredCandidate, ...]
    previewed_count: int
    duplicate_count: int
    suppression_records: tuple[SuppressedSpec, ...] = field(default_factory=tuple)


class DeterministicCandidateQueue:
    def __init__(
        self,
        *,
        history_entries: Sequence[LedgerEntry],
        frontier_entries: Sequence[LedgerEntry],
        suppressor: RegionSuppressor | None = None,
    ) -> None:
        self.history_entries = tuple(entry for entry in history_entries if entry.status == "completed")
        self.frontier_entries = tuple(frontier_entries)
        self.suppressor = suppressor
        self._history_by_family = self._group_by_family(self.history_entries)
        self._frontier_by_id = {entry.experiment_id: entry for entry in self.frontier_entries}
        self._family_counts = {family: len(entries) for family, entries in self._history_by_family.items()}
        self._max_family_count = max(self._family_counts.values(), default=0)
        self._critic_penalty_by_family = {
            family: self._critic_penalty(entries) for family, entries in self._history_by_family.items()
        }
        self._historical_eval_keys = {entry.evaluation_key for entry in self.history_entries}

    def build(
        self,
        *,
        planned_specs: Sequence[PlannedSpec],
        runner: EvaluationRunner,
        num_folds: int,
        embargo_bars: int,
        locked_holdout_months: int | None = None,
        max_preview_count: int | None = None,
    ) -> CandidateQueueResult:
        candidates: list[ScoredCandidate] = []
        suppression_records: list[SuppressedSpec] = []
        previewed_count = 0
        duplicate_count = 0
        queued_evaluation_keys: set[str] = set()
        for planned in self._cheap_rank_plans(planned_specs, runner):
            try:
                evaluation_key = runner.evaluation_key_for_spec(
                    planned.spec,
                    num_folds=num_folds,
                    embargo_bars=embargo_bars,
                    locked_holdout_months=locked_holdout_months,
                )
            except Exception:
                continue
            if evaluation_key in self._historical_eval_keys:
                duplicate_count += 1
                continue
            if evaluation_key in queued_evaluation_keys:
                duplicate_count += 1
                continue
            queued_evaluation_keys.add(evaluation_key)
            if max_preview_count is not None and previewed_count >= max_preview_count:
                continue
            try:
                preview = runner.preview_walk_forward(
                    planned.spec,
                    num_folds=num_folds,
                    embargo_bars=embargo_bars,
                    locked_holdout_months=locked_holdout_months,
                )
            except Exception:
                continue
            previewed_count += 1
            family = preview.spec.signal.name
            novelty = self._novelty_to_history(preview)
            parent_score = self._parent_score(planned)
            # Suppressor: compute penalty and collect audit record before scoring.
            suppression_record: SuppressedSpec | None = None
            if self.suppressor is not None:
                suppression_record = self.suppressor.assess(preview.spec)
                if suppression_record is not None:
                    suppression_records.append(suppression_record)
            static_score = self._static_score(
                planned, preview, novelty, parent_score, suppression_record
            )
            candidates.append(
                ScoredCandidate(
                    planned=planned,
                    preview=preview,
                    family=family,
                    static_score=static_score,
                    novelty_to_history=novelty,
                    parent_score=parent_score,
                    suppression_record=suppression_record,
                )
            )
        return CandidateQueueResult(
            selected=self._select_candidates(candidates),
            previewed_count=previewed_count,
            duplicate_count=duplicate_count,
            suppression_records=tuple(suppression_records),
        )

    def _cheap_rank_plans(
        self,
        planned_specs: Sequence[PlannedSpec],
        runner: EvaluationRunner,
    ) -> tuple[PlannedSpec, ...]:
        return tuple(
            sorted(
                planned_specs,
                key=lambda planned: (
                    self._cheap_score(planned, runner),
                    planned.spec.spec_hash(),
                ),
                reverse=True,
            )
        )

    def _cheap_score(self, planned: PlannedSpec, runner: EvaluationRunner) -> float:
        try:
            spec = runner.registry.validate_spec(planned.spec)
            required_history = runner.registry.required_history(spec)
        except ValueError:
            return float("-inf")
        family_count = self._family_counts.get(spec.signal.name, 0)
        family_quota_boost = float(self._max_family_count - family_count)
        generator_boost = 25.0 if planned.generator_kind == "frontier_neighborhood" else 0.0
        simplicity_boost = max(0.0, 20.0 - (required_history / 10.0))
        return (
            generator_boost
            + (family_quota_boost * 3.0)
            + self._family_quality(spec.signal.name)
            + (self._parent_score(planned) * 0.25)
            + (self._novelty_to_history_spec(spec) * 10.0)
            + simplicity_boost
            - self._critic_penalty_by_family.get(spec.signal.name, 0.0)
        )

    def _select_candidates(self, candidates: Sequence[ScoredCandidate]) -> tuple[ScoredCandidate, ...]:
        grouped: dict[str, list[ScoredCandidate]] = {}
        for candidate in candidates:
            grouped.setdefault(candidate.family, []).append(candidate)
        for family in grouped:
            grouped[family].sort(
                key=lambda candidate: (
                    candidate.static_score,
                    candidate.novelty_to_history,
                    candidate.preview.spec.spec_hash(),
                ),
                reverse=True,
            )

        selected: list[ScoredCandidate] = []
        selected_specs_by_family: dict[str, list[ScoredCandidate]] = {}
        families = sorted(grouped)

        # Deterministic round-robin keeps batches mixed across families when multiple are available.
        while any(grouped.get(family) for family in families):
            made_progress = False
            for family in families:
                family_candidates = grouped.get(family, [])
                if not family_candidates:
                    continue
                best_index = 0
                best_score: float | None = None
                for index, candidate in enumerate(family_candidates):
                    diversity_bonus = self._diversity_bonus(candidate, selected_specs_by_family.get(family, ()))
                    total_score = candidate.static_score + diversity_bonus
                    if best_score is None or total_score > best_score:
                        best_score = total_score
                        best_index = index
                chosen = family_candidates.pop(best_index)
                selected.append(chosen)
                selected_specs_by_family.setdefault(family, []).append(chosen)
                made_progress = True
            if not made_progress:
                break
        return tuple(selected)

    def _static_score(
        self,
        planned: PlannedSpec,
        preview: EvaluationPreview,
        novelty: float,
        parent_score: float,
        suppression_record: SuppressedSpec | None = None,
    ) -> float:
        family_count = self._family_counts.get(preview.spec.signal.name, 0)
        family_quota_boost = float(self._max_family_count - family_count)
        family_quality = self._family_quality(preview.spec.signal.name)
        generator_boost = 25.0 if planned.generator_kind == "frontier_neighborhood" else 0.0
        simplicity_boost = max(0.0, 20.0 - (preview.required_history / 10.0))
        suppression_penalty = suppression_record.suppression_weight if suppression_record is not None else 0.0
        return (
            generator_boost
            + (family_quota_boost * 3.0)
            + family_quality
            + (parent_score * 0.25)
            + (novelty * 10.0)
            + simplicity_boost
            - suppression_penalty
            - self._critic_penalty_by_family.get(preview.spec.signal.name, 0.0)
        )

    def _family_quality(self, family: str) -> float:
        entries = self._history_by_family.get(family, ())
        if not entries:
            return 0.0
        best = max(
            entries,
            key=lambda entry: (
                entry.metric("sharpe_like"),
                entry.metric("return_pct"),
                -entry.metric("max_drawdown_pct"),
            ),
        )
        return (
            max(best.metric("sharpe_like"), 0.0) * 30.0
            + best.metric("return_pct") * 4.0
            - best.metric("max_drawdown_pct")
        )

    def _parent_score(self, planned: PlannedSpec) -> float:
        if not planned.parent_experiment_ids:
            return 0.0
        scores = []
        for parent_id in planned.parent_experiment_ids:
            entry = self._frontier_by_id.get(parent_id)
            if entry is None:
                continue
            scores.append(
                max(entry.metric("sharpe_like"), 0.0) * 40.0
                + entry.metric("return_pct") * 5.0
                - entry.metric("max_drawdown_pct")
            )
        return max(scores, default=0.0)

    @staticmethod
    def _critic_penalty(entries: Sequence[LedgerEntry]) -> float:
        penalties = []
        for entry in entries:
            penalties.append(planning_penalty_from_critique(entry.critique))
        if not penalties:
            return 0.0
        return min(sum(penalties) / len(penalties), 25.0)

    def _novelty_to_history(self, preview: EvaluationPreview) -> float:
        return self._novelty_to_history_spec(preview.spec)

    def _novelty_to_history_spec(self, spec: StrategySpec) -> float:
        family_entries = self._history_by_family.get(spec.signal.name, ())
        if not family_entries:
            return 1.0
        distances = [
            self._spec_distance(spec.to_payload(include_name=False), entry.spec.to_payload(include_name=False))
            for entry in family_entries
        ]
        return min(distances) if distances else 1.0

    def _diversity_bonus(
        self,
        candidate: ScoredCandidate,
        selected_family_candidates: Sequence[ScoredCandidate],
    ) -> float:
        if not selected_family_candidates:
            return 0.0
        distances = [
            self._spec_distance(
                candidate.preview.spec.to_payload(include_name=False),
                selected.preview.spec.to_payload(include_name=False),
            )
            for selected in selected_family_candidates
        ]
        return min(distances, default=0.0) * 20.0

    @staticmethod
    def _group_by_family(entries: Iterable[LedgerEntry]) -> dict[str, tuple[LedgerEntry, ...]]:
        grouped: dict[str, list[LedgerEntry]] = {}
        for entry in entries:
            grouped.setdefault(entry.spec.signal.name, []).append(entry)
        return {family: tuple(items) for family, items in grouped.items()}

    @staticmethod
    def _spec_distance(left: dict[str, object], right: dict[str, object]) -> float:
        """Thin wrapper kept for internal call-sites; delegates to shared function."""
        return spec_distance(left, right)
