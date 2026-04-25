from __future__ import annotations

from dataclasses import dataclass, field
from math import log, sqrt
from time import perf_counter
from typing import Iterable, Sequence

from trader.evaluation.runner import EvaluationPreview, EvaluationRunner
from trader.ledger.entry import LedgerEntry
from trader.research.critic import planning_penalty_from_critique
from trader.research.planner import PlannedSpec
from trader.research.suppressor import RegionSuppressor, SuppressedSpec, spec_distance
from trader.strategies.spec import StrategySpec


_UCB_RETURN_WEIGHT = 4.0
_UCB_EXPLORATION_WEIGHT = 12.0
_UNEXPLORED_FAMILY_UCB_SCORE = 30.0


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
        timings: dict[str, float] | None = None,
    ) -> CandidateQueueResult:
        candidates: list[ScoredCandidate] = []
        suppression_records: list[SuppressedSpec] = []
        previewed_count = 0
        duplicate_count = 0
        queued_evaluation_keys: set[str] = set()
        started = perf_counter()
        ranked_plans = self._cheap_rank_plans(planned_specs, runner)
        _add_timing(timings, "queue_scoring", started)
        for planned in ranked_plans:
            try:
                started = perf_counter()
                evaluation_key = runner.evaluation_key_for_spec(
                    planned.spec,
                    num_folds=num_folds,
                    embargo_bars=embargo_bars,
                    locked_holdout_months=locked_holdout_months,
                )
                _add_timing(timings, "key_compute", started)
            except Exception:
                _add_timing(timings, "key_compute", started)
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
                started = perf_counter()
                preview = runner.preview_walk_forward(
                    planned.spec,
                    num_folds=num_folds,
                    embargo_bars=embargo_bars,
                    locked_holdout_months=locked_holdout_months,
                )
                _add_timing(timings, "preview", started)
            except Exception:
                _add_timing(timings, "preview", started)
                continue
            previewed_count += 1
            started = perf_counter()
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
            _add_timing(timings, "queue_scoring", started)
        started = perf_counter()
        selected = self._select_candidates(candidates)
        _add_timing(timings, "queue_scoring", started)
        return CandidateQueueResult(
            selected=selected,
            previewed_count=previewed_count,
            duplicate_count=duplicate_count,
            suppression_records=tuple(suppression_records),
        )

    def _cheap_rank_plans(
        self,
        planned_specs: Sequence[PlannedSpec],
        runner: EvaluationRunner,
    ) -> tuple[PlannedSpec, ...]:
        grouped: dict[str, list[tuple[PlannedSpec, float]]] = {}
        for planned in planned_specs:
            family = self._planned_family(planned, runner)
            grouped.setdefault(family, []).append((planned, self._cheap_score(planned, runner)))
        family_order = {family: index for index, family in enumerate(sorted(grouped))}
        selected_counts_by_family: dict[str, int] = {}
        ranked: list[PlannedSpec] = []
        while any(grouped.values()):
            best_family: str | None = None
            best_index = 0
            best_score: float | None = None
            for family, family_plans in grouped.items():
                if not family_plans:
                    continue
                base_family_score = self._family_ucb_score(family)
                virtual_family_score = self._family_ucb_score(
                    family,
                    selected_count=selected_counts_by_family.get(family, 0),
                    total_selected_count=len(ranked),
                )
                for index, (planned, static_score) in enumerate(family_plans):
                    total_score = static_score - base_family_score + virtual_family_score
                    if (
                        best_score is None
                        or total_score > best_score
                        or (
                            total_score == best_score
                            and (
                                family_order[family],
                                planned.spec.spec_hash(),
                            )
                            < (
                                family_order[best_family or family],
                                grouped[best_family or family][best_index][0].spec.spec_hash(),
                            )
                        )
                    ):
                        best_score = total_score
                        best_family = family
                        best_index = index
            if best_family is None:
                break
            planned, _ = grouped[best_family].pop(best_index)
            ranked.append(planned)
            selected_counts_by_family[best_family] = selected_counts_by_family.get(best_family, 0) + 1
        return tuple(ranked)

    def _planned_family(self, planned: PlannedSpec, runner: EvaluationRunner) -> str:
        try:
            return runner.registry.validate_spec(planned.spec).signal.name
        except ValueError:
            return planned.spec.signal.name

    def _cheap_score(self, planned: PlannedSpec, runner: EvaluationRunner) -> float:
        try:
            spec = runner.registry.validate_spec(planned.spec)
            required_history = runner.registry.required_history(spec)
        except ValueError:
            return float("-inf")
        generator_boost = 25.0 if planned.generator_kind == "frontier_neighborhood" else 0.0
        simplicity_boost = max(0.0, 20.0 - (required_history / 10.0))
        return (
            generator_boost
            + self._family_ucb_score(spec.signal.name)
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
        selected_counts_by_family: dict[str, int] = {}
        family_order = {family: index for index, family in enumerate(sorted(grouped))}

        while any(grouped.values()):
            best_family: str | None = None
            best_index = 0
            best_score: float | None = None
            for family, family_candidates in grouped.items():
                if not family_candidates:
                    continue
                base_family_score = self._family_ucb_score(family)
                virtual_family_score = self._family_ucb_score(
                    family,
                    selected_count=selected_counts_by_family.get(family, 0),
                    total_selected_count=len(selected),
                )
                for index, candidate in enumerate(family_candidates):
                    diversity_bonus = self._diversity_bonus(candidate, selected_specs_by_family.get(family, ()))
                    total_score = candidate.static_score - base_family_score + virtual_family_score + diversity_bonus
                    if (
                        best_score is None
                        or total_score > best_score
                        or (total_score == best_score and family_order[family] < family_order[best_family or family])
                    ):
                        best_score = total_score
                        best_family = family
                        best_index = index
            if best_family is None:
                break
            chosen = grouped[best_family].pop(best_index)
            selected.append(chosen)
            selected_specs_by_family.setdefault(best_family, []).append(chosen)
            selected_counts_by_family[best_family] = selected_counts_by_family.get(best_family, 0) + 1
        return tuple(selected)

    def _static_score(
        self,
        planned: PlannedSpec,
        preview: EvaluationPreview,
        novelty: float,
        parent_score: float,
        suppression_record: SuppressedSpec | None = None,
    ) -> float:
        generator_boost = 25.0 if planned.generator_kind == "frontier_neighborhood" else 0.0
        simplicity_boost = max(0.0, 20.0 - (preview.required_history / 10.0))
        suppression_penalty = suppression_record.suppression_weight if suppression_record is not None else 0.0
        return (
            generator_boost
            + self._family_ucb_score(preview.spec.signal.name)
            + (parent_score * 0.25)
            + (novelty * 10.0)
            + simplicity_boost
            - suppression_penalty
            - self._critic_penalty_by_family.get(preview.spec.signal.name, 0.0)
        )

    def _family_ucb_score(self, family: str, *, selected_count: int = 0, total_selected_count: int = 0) -> float:
        entries = self._history_by_family.get(family, ())
        history_count = self._family_counts.get(family, 0)
        effective_count = history_count + selected_count
        if effective_count == 0:
            return _UNEXPLORED_FAMILY_UCB_SCORE
        total_count = max(len(self.history_entries) + total_selected_count, 2)
        best_return = max((entry.metric("return_pct") for entry in entries), default=0.0)
        exploration = sqrt(log(total_count) / effective_count)
        return (best_return * _UCB_RETURN_WEIGHT) + (exploration * _UCB_EXPLORATION_WEIGHT)

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


def _add_timing(timings: dict[str, float] | None, phase: str, started: float) -> None:
    if timings is not None:
        timings[phase] = timings.get(phase, 0.0) + (perf_counter() - started)
