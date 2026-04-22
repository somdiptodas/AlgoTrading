from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from trader.research.planner import PlannedSpec
from trader.strategies.registry import StrategyRegistry
from trader.strategies.spec import StrategySpec


@dataclass(frozen=True)
class GeneratedBatch:
    accepted: tuple[PlannedSpec, ...]
    rejected: tuple[str, ...]


class StrategyGenerator:
    def __init__(self, registry: StrategyRegistry) -> None:
        self.registry = registry

    def validate_and_filter(
        self,
        planned_specs: Sequence[PlannedSpec],
        *,
        seen_evaluation_key: Callable[[StrategySpec], bool],
        evaluation_key_for_spec: Callable[[StrategySpec], str],
    ) -> GeneratedBatch:
        accepted: list[PlannedSpec] = []
        rejected: list[str] = []
        for planned in planned_specs:
            try:
                validated = self.registry.validate_spec(planned.spec)
            except ValueError as exc:
                rejected.append(f"{planned.spec.name}: {exc}")
                continue
            if seen_evaluation_key(validated):
                rejected.append(f"{validated.name}: duplicate {evaluation_key_for_spec(validated)}")
                continue
            accepted.append(
                PlannedSpec(
                    spec=validated,
                    generator_kind=planned.generator_kind,
                    parent_experiment_ids=planned.parent_experiment_ids,
                )
            )
        return GeneratedBatch(accepted=tuple(accepted), rejected=tuple(rejected))
