from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from trader.strategies.registry import StrategyRegistry
from trader.strategies.spec import ExecConfig, SignalSpec, SizingSpec, StrategySpec


_SIZING_GRID = (
    SizingSpec("fixed_fraction", {"fraction": 0.25}),
    SizingSpec("fixed_fraction", {"fraction": 0.50}),
    SizingSpec("fixed_fraction", {"fraction": 1.00}),
)
_EXECUTION_GRID = (
    ExecConfig(),
    ExecConfig(entry_session_window="first_30m"),
    ExecConfig(entry_session_window="last_30m"),
    ExecConfig(entry_session_window="avoid_midday"),
    ExecConfig(no_new_entry_minutes_before_close=30),
)


@dataclass(frozen=True)
class PlannedSpec:
    spec: StrategySpec
    generator_kind: str
    parent_experiment_ids: tuple[str, ...] = ()


class DeterministicPlanner:
    def __init__(self, registry: StrategyRegistry) -> None:
        self.registry = registry

    def plan(
        self,
        *,
        batch_size: int,
        frontier_specs: Sequence[tuple[str, StrategySpec]] = (),
        allowed_signal_families: Sequence[str] | None = None,
    ) -> tuple[PlannedSpec, ...]:
        allowed = tuple(allowed_signal_families or ("ema_cross", "breakout"))
        grid_buckets: list[list[PlannedSpec]] = []
        for signal_name in allowed:
            family_candidates: list[PlannedSpec] = []
            for params in self.registry.parameter_grid(signal_name):
                for sizing in _SIZING_GRID:
                    for exec_config in _EXECUTION_GRID:
                        family_candidates.append(
                            PlannedSpec(
                                spec=self._rename(
                                    StrategySpec(
                                        name=f"{signal_name}_grid",
                                        signal=SignalSpec(signal_name, params),
                                        sizing=sizing,
                                        exec_config=exec_config,
                                    )
                                ),
                                generator_kind="grid",
                            )
                        )
            grid_buckets.append(family_candidates)
        frontier_buckets = {signal_name: [] for signal_name in allowed}
        for parent_experiment_id, parent_spec in sorted(frontier_specs, key=lambda item: item[0]):
            if parent_spec.signal.name not in allowed:
                continue
            for neighbor_spec in self.registry.neighbors(parent_spec):
                frontier_buckets[parent_spec.signal.name].append(
                    PlannedSpec(
                        spec=self._rename(neighbor_spec),
                        generator_kind="frontier_neighborhood",
                        parent_experiment_ids=(parent_experiment_id,),
                    )
                )
        buckets = [bucket for signal_name in allowed if (bucket := frontier_buckets[signal_name])] + grid_buckets
        indexes = [0 for _ in buckets]
        deduped: list[PlannedSpec] = []
        seen_hashes: set[str] = set()
        while len(deduped) < batch_size and any(index < len(bucket) for index, bucket in zip(indexes, buckets)):
            for bucket_index, bucket in enumerate(buckets):
                while indexes[bucket_index] < len(bucket):
                    candidate = bucket[indexes[bucket_index]]
                    indexes[bucket_index] += 1
                    spec_hash = candidate.spec.spec_hash()
                    if spec_hash in seen_hashes:
                        continue
                    seen_hashes.add(spec_hash)
                    deduped.append(candidate)
                    break
                if len(deduped) >= batch_size:
                    break
        return tuple(deduped)

    def _rename(self, spec: StrategySpec) -> StrategySpec:
        return StrategySpec.from_payload(
            {
                **spec.to_payload(),
                "name": f"{spec.signal.name}_{spec.spec_hash()[:8]}",
            }
        )
