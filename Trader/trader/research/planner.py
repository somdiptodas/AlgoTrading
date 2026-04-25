from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from trader.strategies.registry import StrategyRegistry
from trader.strategies.spec import ExecConfig, FilterSpec, SignalSpec, SizingSpec, StrategySpec


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
_FILTER_GRID = (
    (),
    (FilterSpec("intraday_volatility", {"lookback_bars": 10, "percentile_window": 60, "min_percentile": 70.0}),),
    (
        FilterSpec(
            "intraday_volatility",
            {"lookback_bars": 10, "percentile_window": 60, "min_percentile": 0.0, "max_percentile": 30.0},
        ),
    ),
    (FilterSpec("prior_day_range", {"min_range_bps": 100.0}),),
    (FilterSpec("relative_volume", {"lookback_bars": 20, "min_ratio": 1.25}),),
    (FilterSpec("day_type", {"mode": "trend", "min_bars": 30, "trend_bps": 50.0, "min_efficiency": 0.60}),),
    (
        FilterSpec(
            "day_type",
            {"mode": "mean_reversion", "min_bars": 30, "trend_bps": 50.0, "max_efficiency": 0.35},
        ),
    ),
)
_CONFIRMATION_FILTER_GRID = {
    "rsi_reversion": (
        (FilterSpec("intraday_volatility", {"lookback_bars": 10, "percentile_window": 60, "min_percentile": 70.0}),),
        (
            FilterSpec(
                "intraday_volatility",
                {"lookback_bars": 10, "percentile_window": 60, "min_percentile": 0.0, "max_percentile": 30.0},
            ),
        ),
        (FilterSpec("vwap_distance", {"side": "below", "min_deviation_bps": 25.0}),),
    ),
    "vwap_deviation": (
        (FilterSpec("relative_volume", {"lookback_bars": 20, "min_ratio": 1.25}),),
    ),
}
_COMPOSITE_VARIANTS_PER_RECIPE = 3


@dataclass(frozen=True)
class _CompositeRecipe:
    name: str
    combiner: str
    children: tuple[str, ...] = ()
    signal_name: str | None = None
    filters: tuple[FilterSpec, ...] = ()
    exec_config: ExecConfig = ExecConfig()
    min_agreeing: int | None = None
    primary_index: int | None = None


_COMPOSITE_RECIPES = (
    _CompositeRecipe(
        name="ema_rsi_all",
        combiner="all",
        children=("ema_cross", "rsi_reversion"),
    ),
    _CompositeRecipe(
        name="breakout_trend_volume",
        combiner="",
        signal_name="breakout",
        filters=(
            FilterSpec("relative_volume", {"lookback_bars": 20, "min_ratio": 1.5}),
            FilterSpec("day_type", {"mode": "trend", "min_bars": 30, "trend_bps": 50.0, "min_efficiency": 0.60}),
        ),
    ),
    _CompositeRecipe(
        name="vwap_rsi_all",
        combiner="all",
        children=("vwap_deviation", "rsi_reversion"),
    ),
    _CompositeRecipe(
        name="ema_breakout_rsi_vote",
        combiner="vote_k_of_n",
        children=("ema_cross", "breakout", "rsi_reversion"),
        min_agreeing=2,
    ),
    _CompositeRecipe(
        name="ema_chop_rsi_all",
        combiner="",
        signal_name="ema_cross",
        filters=(
            FilterSpec(
                "day_type",
                {"mode": "mean_reversion", "min_bars": 30, "trend_bps": 50.0, "max_efficiency": 0.35},
            ),
        ),
    ),
    _CompositeRecipe(
        name="breakout_open_volume",
        combiner="",
        signal_name="breakout",
        filters=(FilterSpec("relative_volume", {"lookback_bars": 20, "min_ratio": 1.5}),),
        exec_config=ExecConfig(entry_session_window="first_30m"),
    ),
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
        composite_buckets: list[list[PlannedSpec]] = []
        if "composite" in allowed:
            composite_buckets = self._composite_candidate_buckets()
        confirmation_buckets: list[list[PlannedSpec]] = []
        for signal_name in allowed:
            filters_for_signal = _CONFIRMATION_FILTER_GRID.get(signal_name, ())
            if not filters_for_signal:
                continue
            confirmation_candidates: list[PlannedSpec] = []
            for params in self.registry.parameter_grid(signal_name):
                for filters in filters_for_signal:
                    confirmation_candidates.append(
                        PlannedSpec(
                            spec=self._rename(
                                StrategySpec(
                                    name=f"{signal_name}_confirmation",
                                    signal=SignalSpec(signal_name, params),
                                    sizing=_SIZING_GRID[1],
                                    filters=filters,
                                    exec_config=ExecConfig(),
                                    tags=("confirmation",),
                                )
                            ),
                            generator_kind="confirmation_grid",
                        )
                    )
            confirmation_buckets.append(confirmation_candidates)
        grid_buckets: list[list[PlannedSpec]] = []
        for signal_name in allowed:
            family_candidates: list[PlannedSpec] = []
            for params in self.registry.parameter_grid(signal_name):
                for sizing, exec_config, filters in _parameter_combinations():
                    family_candidates.append(
                        PlannedSpec(
                            spec=self._rename(
                                StrategySpec(
                                    name=f"{signal_name}_grid",
                                    signal=SignalSpec(signal_name, params),
                                    sizing=sizing,
                                    filters=filters,
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
        buckets = (
            [bucket for signal_name in allowed if (bucket := frontier_buckets[signal_name])]
            + composite_buckets
            + grid_buckets
            + confirmation_buckets
        )
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

    def _composite_candidate_buckets(self) -> list[list[PlannedSpec]]:
        buckets: list[list[PlannedSpec]] = []
        for recipe in _COMPOSITE_RECIPES:
            candidates: list[PlannedSpec] = []
            for variant_index in range(_COMPOSITE_VARIANTS_PER_RECIPE):
                if recipe.signal_name is not None:
                    signal_grid = self.registry.parameter_grid(recipe.signal_name)
                    signal = SignalSpec(recipe.signal_name, signal_grid[variant_index % len(signal_grid)])
                else:
                    signal = SignalSpec("composite", self._composite_signal_params(recipe, variant_index))
                candidates.append(
                    PlannedSpec(
                        spec=self._rename(
                            StrategySpec(
                                name=f"composite_{recipe.name}",
                                signal=signal,
                                sizing=_SIZING_GRID[1],
                                filters=recipe.filters,
                                exec_config=recipe.exec_config,
                                tags=("composite", recipe.name),
                            )
                        ),
                        generator_kind="composite_grid",
                    )
                )
            buckets.append(candidates)
        return buckets

    def _composite_signal_params(self, recipe: _CompositeRecipe, variant_index: int) -> dict[str, object]:
        children = []
        for child_offset, child_name in enumerate(recipe.children):
            child_grid = self.registry.parameter_grid(child_name)
            params = child_grid[(variant_index + child_offset) % len(child_grid)]
            children.append({"name": child_name, "params": params})
        signal_params: dict[str, object] = {
            "combiner": recipe.combiner,
            "children": children,
        }
        if recipe.min_agreeing is not None:
            signal_params["min_agreeing"] = recipe.min_agreeing
        if recipe.primary_index is not None:
            signal_params["primary_index"] = recipe.primary_index
        return signal_params


def _parameter_combinations() -> tuple[tuple[SizingSpec, ExecConfig, tuple[FilterSpec, ...]], ...]:
    combination_count = len(_SIZING_GRID) * len(_EXECUTION_GRID) * len(_FILTER_GRID)
    return tuple(
        (
            _SIZING_GRID[index % len(_SIZING_GRID)],
            _EXECUTION_GRID[index % len(_EXECUTION_GRID)],
            _FILTER_GRID[index % len(_FILTER_GRID)],
        )
        for index in range(combination_count)
    )
