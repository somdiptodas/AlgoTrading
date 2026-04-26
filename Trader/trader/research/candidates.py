from __future__ import annotations

from dataclasses import dataclass, field
from math import log, sqrt
from time import perf_counter
from typing import Iterable, Sequence

from trader.evaluation.runner import EvaluationPreview, EvaluationRunner
from trader.ledger.entry import LedgerEntry
from trader.research.critic import planning_penalty_from_critique
from trader.research.critic_memory import CriticRegionMemory
from trader.research.planner import PlannedSpec, strategy_shape_key
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
    shape_key: str
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
        critic_memory: CriticRegionMemory | None = None,
        suppressor: RegionSuppressor | None = None,
    ) -> None:
        self.history_entries = tuple(entry for entry in history_entries if entry.status == "completed")
        self.frontier_entries = tuple(frontier_entries)
        self.critic_memory = critic_memory
        self.suppressor = suppressor
        self._history_by_shape = self._group_by_shape(self.history_entries)
        self._frontier_by_id = {entry.experiment_id: entry for entry in self.frontier_entries}
        self._shape_counts = {shape_key: len(entries) for shape_key, entries in self._history_by_shape.items()}
        self._critic_penalty_by_shape = {
            shape_key: self._shape_critic_penalty(entries) for shape_key, entries in self._history_by_shape.items()
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
            shape_key = strategy_shape_key(preview.spec)
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
                    shape_key=shape_key,
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
            shape_key = self._planned_shape_key(planned, runner)
            grouped.setdefault(shape_key, []).append((planned, self._cheap_score(planned, runner)))
        shape_order = {shape_key: index for index, shape_key in enumerate(sorted(grouped))}
        selected_counts_by_shape: dict[str, int] = {}
        ranked: list[PlannedSpec] = []
        while any(grouped.values()):
            best_shape: str | None = None
            best_index = 0
            best_score: float | None = None
            for shape_key, shape_plans in grouped.items():
                if not shape_plans:
                    continue
                base_shape_score = self._shape_ucb_score(shape_key)
                virtual_shape_score = self._shape_ucb_score(
                    shape_key,
                    selected_count=selected_counts_by_shape.get(shape_key, 0),
                    total_selected_count=len(ranked),
                )
                for index, (planned, static_score) in enumerate(shape_plans):
                    total_score = static_score - base_shape_score + virtual_shape_score
                    if (
                        best_score is None
                        or total_score > best_score
                        or (
                            total_score == best_score
                            and (
                                shape_order[shape_key],
                                planned.spec.spec_hash(),
                            )
                            < (
                                shape_order[best_shape or shape_key],
                                grouped[best_shape or shape_key][best_index][0].spec.spec_hash(),
                            )
                        )
                    ):
                        best_score = total_score
                        best_shape = shape_key
                        best_index = index
            if best_shape is None:
                break
            planned, _ = grouped[best_shape].pop(best_index)
            ranked.append(planned)
            selected_counts_by_shape[best_shape] = selected_counts_by_shape.get(best_shape, 0) + 1
        return tuple(ranked)

    def _planned_shape_key(self, planned: PlannedSpec, runner: EvaluationRunner) -> str:
        try:
            return strategy_shape_key(runner.registry.validate_spec(planned.spec))
        except ValueError:
            return strategy_shape_key(planned.spec)

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
            + self._shape_ucb_score(strategy_shape_key(spec))
            + (self._parent_score(planned) * 0.25)
            + (self._novelty_to_history_spec(spec) * 10.0)
            + simplicity_boost
            - self._critic_penalty(spec)
        )

    def _select_candidates(self, candidates: Sequence[ScoredCandidate]) -> tuple[ScoredCandidate, ...]:
        grouped: dict[str, list[ScoredCandidate]] = {}
        for candidate in candidates:
            grouped.setdefault(candidate.shape_key, []).append(candidate)
        for shape_key in grouped:
            grouped[shape_key].sort(
                key=lambda candidate: (
                    candidate.static_score,
                    candidate.novelty_to_history,
                    candidate.preview.spec.spec_hash(),
                ),
                reverse=True,
            )

        selected: list[ScoredCandidate] = []
        selected_specs_by_shape: dict[str, list[ScoredCandidate]] = {}
        selected_counts_by_shape: dict[str, int] = {}
        shape_order = {shape_key: index for index, shape_key in enumerate(sorted(grouped))}

        while any(grouped.values()):
            best_shape: str | None = None
            best_index = 0
            best_score: float | None = None
            for shape_key, shape_candidates in grouped.items():
                if not shape_candidates:
                    continue
                base_shape_score = self._shape_ucb_score(shape_key)
                virtual_shape_score = self._shape_ucb_score(
                    shape_key,
                    selected_count=selected_counts_by_shape.get(shape_key, 0),
                    total_selected_count=len(selected),
                )
                for index, candidate in enumerate(shape_candidates):
                    diversity_bonus = self._diversity_bonus(candidate, selected_specs_by_shape.get(shape_key, ()))
                    total_score = candidate.static_score - base_shape_score + virtual_shape_score + diversity_bonus
                    if (
                        best_score is None
                        or total_score > best_score
                        or (total_score == best_score and shape_order[shape_key] < shape_order[best_shape or shape_key])
                    ):
                        best_score = total_score
                        best_shape = shape_key
                        best_index = index
            if best_shape is None:
                break
            chosen = grouped[best_shape].pop(best_index)
            selected.append(chosen)
            selected_specs_by_shape.setdefault(best_shape, []).append(chosen)
            selected_counts_by_shape[best_shape] = selected_counts_by_shape.get(best_shape, 0) + 1
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
            + self._shape_ucb_score(strategy_shape_key(preview.spec))
            + (parent_score * 0.25)
            + (novelty * 10.0)
            + simplicity_boost
            - suppression_penalty
            - self._critic_penalty(preview.spec)
        )

    def _shape_ucb_score(self, shape_key: str, *, selected_count: int = 0, total_selected_count: int = 0) -> float:
        entries = self._history_by_shape.get(shape_key, ())
        history_count = self._shape_counts.get(shape_key, 0)
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
    def _shape_critic_penalty(entries: Sequence[LedgerEntry]) -> float:
        penalties = []
        for entry in entries:
            penalties.append(planning_penalty_from_critique(entry.critique))
        if not penalties:
            return 0.0
        return min(sum(penalties) / len(penalties), 25.0)

    def _critic_penalty(self, spec: StrategySpec) -> float:
        if self.critic_memory is not None and self.critic_memory.record_count > 0:
            return self.critic_memory.penalty(spec)
        return self._critic_penalty_by_shape.get(strategy_shape_key(spec), 0.0)

    def _novelty_to_history(self, preview: EvaluationPreview) -> float:
        return self._novelty_to_history_spec(preview.spec)

    def _novelty_to_history_spec(self, spec: StrategySpec) -> float:
        shape_entries = self._history_by_shape.get(strategy_shape_key(spec), ())
        if not shape_entries:
            return 1.0
        distances = [
            self._spec_distance(spec.to_payload(include_name=False), entry.spec.to_payload(include_name=False))
            for entry in shape_entries
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
    def _group_by_shape(entries: Iterable[LedgerEntry]) -> dict[str, tuple[LedgerEntry, ...]]:
        grouped: dict[str, list[LedgerEntry]] = {}
        for entry in entries:
            grouped.setdefault(strategy_shape_key(entry.spec), []).append(entry)
        return {shape_key: tuple(items) for shape_key, items in grouped.items()}

    @staticmethod
    def _spec_distance(left: dict[str, object], right: dict[str, object]) -> float:
        """Thin wrapper kept for internal call-sites; delegates to shared function."""
        return spec_distance(left, right)


def _add_timing(timings: dict[str, float] | None, phase: str, started: float) -> None:
    if timings is not None:
        timings[phase] = timings.get(phase, 0.0) + (perf_counter() - started)
