from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from trader.strategies.registry import StrategyRegistry
from trader.strategies.spec import FilterSpec, SignalSpec, StrategySpec


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
    ) -> tuple[PlannedSpec, ...]:
        candidates: list[PlannedSpec] = []
        for signal_name in ("ema_cross", "breakout"):
            for params in self.registry.parameter_grid(signal_name):
                candidates.append(
                    PlannedSpec(
                        spec=self._rename(
                            StrategySpec(
                                name=f"{signal_name}_grid",
                                signal=SignalSpec(signal_name, params),
                                filters=(FilterSpec("session", {"session": "regular"}),),
                            )
                        ),
                        generator_kind="grid",
                    )
                )
        for parent_experiment_id, parent_spec in sorted(frontier_specs, key=lambda item: item[0]):
            for neighbor_spec in self.registry.neighbors(parent_spec):
                candidates.append(
                    PlannedSpec(
                        spec=self._rename(neighbor_spec),
                        generator_kind="frontier_neighborhood",
                        parent_experiment_ids=(parent_experiment_id,),
                    )
                )
        deduped: list[PlannedSpec] = []
        seen_hashes: set[str] = set()
        for candidate in candidates:
            spec_hash = candidate.spec.spec_hash()
            if spec_hash in seen_hashes:
                continue
            seen_hashes.add(spec_hash)
            deduped.append(candidate)
            if len(deduped) >= batch_size:
                break
        return tuple(deduped)

    def _rename(self, spec: StrategySpec) -> StrategySpec:
        return StrategySpec.from_payload(
            {
                **spec.to_payload(),
                "name": f"{spec.signal.name}_{spec.spec_hash()[:8]}",
                "filters": [{"name": "session", "params": {"session": "regular"}}],
            }
        )
