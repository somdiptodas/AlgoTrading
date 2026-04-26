from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from math import exp
from pathlib import Path
from typing import Iterable, Sequence

from trader.ledger.entry import LedgerEntry
from trader.strategies.registry import StrategyRegistry
from trader.strategies.spec import ExecConfig, FilterSpec, SignalSpec, SizingSpec, StrategySpec


_OPTUNA_SEED_EVALUATION_THRESHOLD = 10
_OPTUNA_SUGGESTIONS_PER_FAMILY = 8
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
MULTI_SIGNAL_SEARCH_SPACE_VERSION = "multi_signal_v1"
_MULTI_SIGNAL_VARIANTS_PER_SHAPE = 12


@dataclass(frozen=True)
class MultiSignalRuleShape:
    combiner: str
    predicates: tuple[str, ...]
    k: int | None = None


@dataclass(frozen=True)
class MultiSignalSearchGrammar:
    version: str
    entry_shapes: tuple[MultiSignalRuleShape, ...]
    exit_shapes: tuple[MultiSignalRuleShape, ...]
    predicate_param_grids: dict[str, tuple[dict[str, object], ...]]


_MULTI_SIGNAL_PREDICATE_PARAM_GRIDS: dict[str, tuple[dict[str, object], ...]] = {
    "rsi_below": (
        {"length": 7, "threshold": 25.0},
        {"length": 7, "threshold": 30.0},
        {"length": 14, "threshold": 30.0},
        {"length": 21, "threshold": 35.0},
    ),
    "rsi_above": (
        {"length": 7, "threshold": 70.0},
        {"length": 14, "threshold": 70.0},
        {"length": 14, "threshold": 75.0},
        {"length": 21, "threshold": 65.0},
    ),
    "ema_trend_up": (
        {"fast": 8, "slow": 34, "buffer_bps": 0.0},
        {"fast": 12, "slow": 55, "buffer_bps": 0.0},
        {"fast": 20, "slow": 80, "buffer_bps": 5.0},
    ),
    "ema_trend_down": (
        {"fast": 8, "slow": 34, "buffer_bps": 0.0},
        {"fast": 12, "slow": 55, "buffer_bps": 0.0},
        {"fast": 20, "slow": 80, "buffer_bps": 5.0},
    ),
    "breakout_up": (
        {"window": 20, "buffer_bps": 0.0},
        {"window": 40, "buffer_bps": 5.0},
        {"window": 60, "buffer_bps": 10.0},
    ),
    "breakout_failed": (
        {"window": 20, "buffer_bps": 0.0},
        {"window": 40, "buffer_bps": 5.0},
        {"window": 60, "buffer_bps": 10.0},
    ),
    "vwap_distance": (
        {"side": "below", "min_bps": 10.0},
        {"side": "below", "min_bps": 25.0},
        {"side": "below", "min_bps": 50.0},
    ),
    "vwap_reclaimed": (
        {"min_bps": 0.0},
        {"min_bps": 5.0},
        {"min_bps": 10.0},
    ),
    "relative_volume": (
        {"lookback": 20, "min_ratio": 1.25},
        {"lookback": 20, "min_ratio": 1.50},
        {"lookback": 40, "min_ratio": 1.25},
    ),
    "intraday_volatility": (
        {"lookback": 10, "percentile_window": 60, "min_percentile": 50.0},
        {"lookback": 20, "percentile_window": 120, "min_percentile": 70.0},
    ),
    "day_type": (
        {"mode": "trend", "min_bars": 30, "trend_bps": 50.0, "min_efficiency": 0.60},
        {"mode": "mean_reversion", "min_bars": 30, "trend_bps": 50.0, "max_efficiency": 0.35},
    ),
}
_MULTI_SIGNAL_GRAMMAR = MultiSignalSearchGrammar(
    version=MULTI_SIGNAL_SEARCH_SPACE_VERSION,
    entry_shapes=(
        MultiSignalRuleShape("k_of_n", ("rsi_below", "vwap_distance", "relative_volume", "ema_trend_up"), k=3),
        MultiSignalRuleShape("all", ("rsi_below", "vwap_distance", "relative_volume")),
        MultiSignalRuleShape("k_of_n", ("breakout_up", "relative_volume", "ema_trend_up"), k=2),
        MultiSignalRuleShape("all", ("breakout_up", "day_type", "relative_volume")),
        MultiSignalRuleShape("k_of_n", ("ema_trend_up", "intraday_volatility", "day_type", "breakout_up"), k=3),
        MultiSignalRuleShape("k_of_n", ("rsi_below", "intraday_volatility", "vwap_distance"), k=2),
    ),
    exit_shapes=(
        MultiSignalRuleShape("any", ("rsi_above", "ema_trend_down", "vwap_reclaimed")),
        MultiSignalRuleShape("k_of_n", ("rsi_above", "ema_trend_down", "breakout_failed"), k=2),
        MultiSignalRuleShape("any", ("vwap_reclaimed", "intraday_volatility", "day_type")),
        MultiSignalRuleShape("k_of_n", ("rsi_above", "vwap_reclaimed", "breakout_failed"), k=2),
        MultiSignalRuleShape("any", ("ema_trend_down", "day_type", "relative_volume")),
    ),
    predicate_param_grids=_MULTI_SIGNAL_PREDICATE_PARAM_GRIDS,
)


def multi_signal_search_grammar() -> MultiSignalSearchGrammar:
    return _MULTI_SIGNAL_GRAMMAR


def strategy_shape_key(spec: StrategySpec) -> str:
    if spec.signal.name == "multi_signal":
        params = spec.signal.params
        entry_key = _multi_signal_rule_shape_key("entry", params.get("entry_rule"))
        exit_key = _multi_signal_rule_shape_key("exit", params.get("exit_rule"))
        if entry_key is not None and exit_key is not None:
            return f"{entry_key}|{exit_key}"
    if spec.signal.name == "composite":
        children = spec.signal.params.get("children")
        if isinstance(children, list):
            child_names = sorted(
                str(child.get("name", ""))
                for child in children
                if isinstance(child, dict) and child.get("name")
            )
            if child_names:
                return f"composite:{spec.signal.params.get('combiner', 'all')}:{'+'.join(child_names)}"
    return spec.signal.name


def _multi_signal_rule_shape_key(scope: str, rule: object) -> str | None:
    if not isinstance(rule, dict):
        return None
    signals = rule.get("signals")
    if not isinstance(signals, list):
        return None
    names = sorted(
        str(signal.get("name", ""))
        for signal in signals
        if isinstance(signal, dict) and signal.get("name")
    )
    if len(names) < 3:
        return None
    combiner = str(rule.get("combiner", "all"))
    k_part = f":k={int(rule['k'])}" if combiner == "k_of_n" and "k" in rule else ""
    return f"{scope}:{combiner}{k_part}:{'+'.join(names)}"


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
        history_entries: Sequence[LedgerEntry] = (),
        optuna_dir: Path | None = None,
        restart_seed: str | None = None,
        restart_index: int = 0,
    ) -> tuple[PlannedSpec, ...]:
        allowed = tuple(allowed_signal_families or ("ema_cross", "breakout"))
        optuna_buckets = self._optuna_candidate_buckets(
            allowed_signal_families=allowed,
            history_entries=history_entries,
            optuna_dir=optuna_dir,
        )
        composite_buckets: list[Iterable[PlannedSpec]] = []
        if "composite" in allowed:
            composite_buckets = self._composite_candidate_buckets()
        confirmation_buckets: list[Iterable[PlannedSpec]] = []
        for signal_name in allowed:
            filters_for_signal = _CONFIRMATION_FILTER_GRID.get(signal_name, ())
            if not filters_for_signal:
                continue
            confirmation_buckets.append(self._iter_confirmation_candidates(signal_name, filters_for_signal))
        grid_buckets: list[Iterable[PlannedSpec]] = []
        for signal_name in allowed:
            if signal_name == "multi_signal":
                grid_buckets.append(
                    self._iter_multi_signal_candidates(
                        restart_offset=_restart_offset(
                            restart_seed,
                            restart_index,
                            "multi_signal_grid",
                            _multi_signal_candidate_count(),
                        )
                    )
                )
            else:
                grid_buckets.append(self._iter_grid_candidates(signal_name))
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
            + optuna_buckets
            + composite_buckets
            + grid_buckets
            + confirmation_buckets
        )
        return _round_robin_deduped(buckets, batch_size)

    def _iter_confirmation_candidates(
        self,
        signal_name: str,
        filters_for_signal: Sequence[tuple[FilterSpec, ...]],
    ) -> Iterable[PlannedSpec]:
        for params in self.registry.parameter_grid(signal_name):
            for filters in filters_for_signal:
                yield PlannedSpec(
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

    def _iter_grid_candidates(self, signal_name: str) -> Iterable[PlannedSpec]:
        for params in self.registry.parameter_grid(signal_name):
            for sizing, exec_config, filters in _parameter_combinations():
                yield PlannedSpec(
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

    def _iter_multi_signal_candidates(self, *, restart_offset: int = 0) -> Iterable[PlannedSpec]:
        grammar = multi_signal_search_grammar()
        exit_shape_count = len(grammar.exit_shapes)
        for offset in range(_multi_signal_candidate_count()):
            candidate_index = (restart_offset + offset) % _multi_signal_candidate_count()
            variant_index = candidate_index % _MULTI_SIGNAL_VARIANTS_PER_SHAPE
            shape_index = candidate_index // _MULTI_SIGNAL_VARIANTS_PER_SHAPE
            entry_shape = grammar.entry_shapes[shape_index // exit_shape_count]
            exit_shape = grammar.exit_shapes[shape_index % exit_shape_count]
            sizing, exec_config, filters = _parameter_combination_at(variant_index)
            spec = self.registry.validate_spec(
                StrategySpec(
                    name="multi_signal_grid",
                    signal=SignalSpec(
                        "multi_signal",
                        {
                            "entry_rule": _multi_signal_rule_payload(entry_shape, variant_index),
                            "exit_rule": _multi_signal_rule_payload(exit_shape, variant_index),
                        },
                    ),
                    sizing=sizing,
                    filters=filters,
                    exec_config=exec_config,
                    tags=(grammar.version,),
                )
            )
            yield PlannedSpec(
                spec=self._rename(spec),
                generator_kind="multi_signal_grid",
            )

    def _rename(self, spec: StrategySpec) -> StrategySpec:
        return StrategySpec.from_payload(
            {
                **spec.to_payload(),
                "name": f"{spec.signal.name}_{spec.spec_hash()[:8]}",
            }
        )

    def _optuna_candidate_buckets(
        self,
        *,
        allowed_signal_families: Sequence[str],
        history_entries: Sequence[LedgerEntry],
        optuna_dir: Path | None,
    ) -> list[Iterable[PlannedSpec]]:
        if optuna_dir is None:
            return []
        completed_by_family = _completed_entries_by_family(history_entries)
        buckets: list[Iterable[PlannedSpec]] = []
        for signal_name in allowed_signal_families:
            if signal_name == "multi_signal":
                buckets.extend(
                    self._multi_signal_optuna_candidate_buckets(
                        history_entries=history_entries,
                        optuna_dir=optuna_dir,
                    )
                )
                continue
            if signal_name == "composite":
                continue
            signal_grid = self.registry.parameter_grid(signal_name)
            if not signal_grid:
                continue
            family_entries = completed_by_family.get(signal_name, ())
            if len(family_entries) < _OPTUNA_SEED_EVALUATION_THRESHOLD:
                continue
            best_entry = max(family_entries, key=lambda entry: entry.metric("return_pct"))
            suggested_params = _suggest_tpe_params(
                signal_name=signal_name,
                signal_grid=signal_grid,
                history_entries=family_entries,
                limit=_OPTUNA_SUGGESTIONS_PER_FAMILY,
                optuna_dir=optuna_dir,
            )
            candidates: list[PlannedSpec] = []
            for params in suggested_params:
                candidates.append(
                    PlannedSpec(
                        spec=self._rename(
                            StrategySpec(
                                name=f"{signal_name}_optuna_tpe",
                                signal=SignalSpec(signal_name, params),
                                sizing=best_entry.spec.sizing,
                                filters=best_entry.spec.filters,
                                exec_config=best_entry.spec.exec_config,
                                tags=("optuna_tpe",),
                            )
                        ),
                        generator_kind="optuna_tpe",
                        parent_experiment_ids=(best_entry.experiment_id,),
                    )
                )
            if candidates:
                buckets.append(candidates)
        return buckets

    def _multi_signal_optuna_candidate_buckets(
        self,
        *,
        history_entries: Sequence[LedgerEntry],
        optuna_dir: Path,
    ) -> list[Iterable[PlannedSpec]]:
        buckets: list[Iterable[PlannedSpec]] = []
        completed_by_shape = _completed_multi_signal_entries_by_shape(history_entries)
        for shape_key, shape_entries in completed_by_shape.items():
            if len(shape_entries) < _OPTUNA_SEED_EVALUATION_THRESHOLD:
                continue
            best_entry = max(shape_entries, key=lambda entry: entry.metric("return_pct"))
            shapes = _multi_signal_shapes_from_spec(best_entry.spec)
            if shapes is None:
                continue
            entry_shape, exit_shape = shapes
            choices_by_key = _multi_signal_choices_by_key(entry_shape, exit_shape)
            if not choices_by_key:
                continue
            suggested_params = _suggest_tpe_flat_params(
                study_label=f"multi_signal_{_stable_seed(shape_key):08x}",
                choices_by_key=choices_by_key,
                history_entries=shape_entries,
                flatten_entry_params=lambda entry: _flatten_multi_signal_params(entry.spec.signal.params, choices_by_key),
                limit=_OPTUNA_SUGGESTIONS_PER_FAMILY,
                optuna_dir=optuna_dir,
            )
            candidates: list[PlannedSpec] = []
            for flat_params in suggested_params:
                spec = self.registry.validate_spec(
                    StrategySpec(
                        name="multi_signal_optuna_tpe",
                        signal=SignalSpec(
                            "multi_signal",
                            _multi_signal_params_from_flat(entry_shape, exit_shape, flat_params),
                        ),
                        sizing=best_entry.spec.sizing,
                        filters=best_entry.spec.filters,
                        exec_config=best_entry.spec.exec_config,
                        tags=(MULTI_SIGNAL_SEARCH_SPACE_VERSION, "optuna_tpe"),
                    )
                )
                candidates.append(
                    PlannedSpec(
                        spec=self._rename(spec),
                        generator_kind="optuna_tpe",
                        parent_experiment_ids=(best_entry.experiment_id,),
                    )
                )
            if candidates:
                _write_multi_signal_study_snapshot(
                    shape_key,
                    optuna_dir,
                    shape_entries,
                    choices_by_key,
                    suggested_params,
                )
                buckets.append(candidates)
        return buckets

    def _composite_candidate_buckets(self) -> list[Iterable[PlannedSpec]]:
        return [self._iter_composite_recipe_candidates(recipe) for recipe in _COMPOSITE_RECIPES]

    def _iter_composite_recipe_candidates(self, recipe: _CompositeRecipe) -> Iterable[PlannedSpec]:
        for variant_index in range(_COMPOSITE_VARIANTS_PER_RECIPE):
            if recipe.signal_name is not None:
                signal_grid = self.registry.parameter_grid(recipe.signal_name)
                signal = SignalSpec(recipe.signal_name, signal_grid[variant_index % len(signal_grid)])
            else:
                signal = SignalSpec("composite", self._composite_signal_params(recipe, variant_index))
            yield PlannedSpec(
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


def _round_robin_deduped(buckets: Sequence[Iterable[PlannedSpec]], batch_size: int) -> tuple[PlannedSpec, ...]:
    iterators = [iter(bucket) for bucket in buckets]
    deduped: list[PlannedSpec] = []
    seen_hashes: set[str] = set()
    while len(deduped) < batch_size and iterators:
        active_iterators = []
        for iterator in iterators:
            for candidate in iterator:
                spec_hash = candidate.spec.spec_hash()
                if spec_hash in seen_hashes:
                    continue
                seen_hashes.add(spec_hash)
                deduped.append(candidate)
                active_iterators.append(iterator)
                break
            if len(deduped) >= batch_size:
                break
        iterators = active_iterators
    return tuple(deduped)


def _parameter_combinations() -> Iterable[tuple[SizingSpec, ExecConfig, tuple[FilterSpec, ...]]]:
    combination_count = len(_SIZING_GRID) * len(_EXECUTION_GRID) * len(_FILTER_GRID)
    for index in range(combination_count):
        yield _parameter_combination_at(index)


def _parameter_combination_at(index: int) -> tuple[SizingSpec, ExecConfig, tuple[FilterSpec, ...]]:
    return (
        _SIZING_GRID[index % len(_SIZING_GRID)],
        _EXECUTION_GRID[index % len(_EXECUTION_GRID)],
        _FILTER_GRID[index % len(_FILTER_GRID)],
    )


def _multi_signal_candidate_count() -> int:
    grammar = multi_signal_search_grammar()
    return len(grammar.entry_shapes) * len(grammar.exit_shapes) * _MULTI_SIGNAL_VARIANTS_PER_SHAPE


def _restart_offset(seed: str | None, restart_index: int, label: str, modulo: int) -> int:
    if seed is None or restart_index <= 0 or modulo <= 1:
        return 0
    return 1 + (_stable_seed(f"{seed}:{restart_index}:{label}") % (modulo - 1))


def _multi_signal_rule_payload(shape: MultiSignalRuleShape, variant_index: int) -> dict[str, object]:
    grammar = multi_signal_search_grammar()
    rule: dict[str, object] = {
        "combiner": shape.combiner,
        "signals": [
            {
                "name": predicate,
                "params": grammar.predicate_param_grids[predicate][
                    (variant_index + offset) % len(grammar.predicate_param_grids[predicate])
                ],
            }
            for offset, predicate in enumerate(shape.predicates)
        ],
    }
    if shape.k is not None:
        rule["k"] = shape.k
    return rule


def _multi_signal_shapes_from_spec(spec: StrategySpec) -> tuple[MultiSignalRuleShape, MultiSignalRuleShape] | None:
    if spec.signal.name != "multi_signal":
        return None
    entry_shape = _multi_signal_shape_from_rule(spec.signal.params.get("entry_rule"))
    exit_shape = _multi_signal_shape_from_rule(spec.signal.params.get("exit_rule"))
    if entry_shape is None or exit_shape is None:
        return None
    return entry_shape, exit_shape


def _multi_signal_shape_from_rule(rule: object) -> MultiSignalRuleShape | None:
    if not isinstance(rule, dict):
        return None
    signals = rule.get("signals")
    if not isinstance(signals, list):
        return None
    predicates = tuple(
        sorted(
            str(signal.get("name", ""))
            for signal in signals
            if isinstance(signal, dict) and signal.get("name") in _MULTI_SIGNAL_PREDICATE_PARAM_GRIDS
        )
    )
    if len(predicates) < 3:
        return None
    combiner = str(rule.get("combiner", "all"))
    k = int(rule["k"]) if combiner == "k_of_n" and "k" in rule else None
    return MultiSignalRuleShape(combiner, predicates, k=k)


def _multi_signal_choices_by_key(
    entry_shape: MultiSignalRuleShape,
    exit_shape: MultiSignalRuleShape,
) -> dict[str, tuple[object, ...]]:
    choices: dict[str, tuple[object, ...]] = {}
    for scope, shape in (("entry", entry_shape), ("exit", exit_shape)):
        for predicate in shape.predicates:
            for param_name in sorted({key for params in _MULTI_SIGNAL_PREDICATE_PARAM_GRIDS[predicate] for key in params}):
                values = tuple(
                    sorted(
                        {params[param_name] for params in _MULTI_SIGNAL_PREDICATE_PARAM_GRIDS[predicate] if param_name in params},
                        key=lambda value: str(value),
                    )
                )
                if values:
                    choices[f"{scope}.{predicate}.{param_name}"] = values
    return choices


def _flatten_multi_signal_params(
    params: dict[str, object],
    choices_by_key: dict[str, tuple[object, ...]],
) -> dict[str, object]:
    flattened: dict[str, object] = {}
    for scope, rule_name in (("entry", "entry_rule"), ("exit", "exit_rule")):
        rule = params.get(rule_name)
        if not isinstance(rule, dict):
            continue
        signals = rule.get("signals")
        if not isinstance(signals, list):
            continue
        by_name = {
            str(signal.get("name")): signal.get("params", {})
            for signal in signals
            if isinstance(signal, dict) and isinstance(signal.get("params"), dict)
        }
        for flat_key in choices_by_key:
            key_scope, predicate, param_name = flat_key.split(".", 2)
            if key_scope != scope:
                continue
            signal_params = by_name.get(predicate, {})
            if param_name in signal_params:
                flattened[flat_key] = signal_params[param_name]
    return flattened


def _multi_signal_params_from_flat(
    entry_shape: MultiSignalRuleShape,
    exit_shape: MultiSignalRuleShape,
    flat_params: dict[str, object],
) -> dict[str, object]:
    return {
        "entry_rule": _multi_signal_rule_from_flat("entry", entry_shape, flat_params),
        "exit_rule": _multi_signal_rule_from_flat("exit", exit_shape, flat_params),
    }


def _multi_signal_rule_from_flat(
    scope: str,
    shape: MultiSignalRuleShape,
    flat_params: dict[str, object],
) -> dict[str, object]:
    signals = []
    for predicate in shape.predicates:
        param_names = sorted({key for params in _MULTI_SIGNAL_PREDICATE_PARAM_GRIDS[predicate] for key in params})
        signals.append(
            {
                "name": predicate,
                "params": {
                    param_name: flat_params[f"{scope}.{predicate}.{param_name}"]
                    for param_name in param_names
                    if f"{scope}.{predicate}.{param_name}" in flat_params
                },
            }
        )
    rule: dict[str, object] = {"combiner": shape.combiner, "signals": signals}
    if shape.k is not None:
        rule["k"] = shape.k
    return rule


def _completed_entries_by_family(entries: Sequence[LedgerEntry]) -> dict[str, tuple[LedgerEntry, ...]]:
    grouped: dict[str, list[LedgerEntry]] = {}
    for entry in entries:
        if entry.status != "completed":
            continue
        grouped.setdefault(entry.spec.signal.name, []).append(entry)
    return {family: tuple(items) for family, items in grouped.items()}


def _completed_multi_signal_entries_by_shape(entries: Sequence[LedgerEntry]) -> dict[str, tuple[LedgerEntry, ...]]:
    grouped: dict[str, list[LedgerEntry]] = {}
    for entry in entries:
        if entry.status != "completed" or entry.spec.signal.name != "multi_signal":
            continue
        grouped.setdefault(strategy_shape_key(entry.spec), []).append(entry)
    return {shape_key: tuple(items) for shape_key, items in grouped.items()}


def _suggest_tpe_params(
    *,
    signal_name: str,
    signal_grid: Sequence[dict[str, object]],
    history_entries: Sequence[LedgerEntry],
    limit: int,
    optuna_dir: Path,
) -> tuple[dict[str, object], ...]:
    optuna_dir.mkdir(parents=True, exist_ok=True)
    suggestions = _optuna_library_suggestions(signal_name, signal_grid, history_entries, limit, optuna_dir)
    if len(suggestions) < limit:
        existing = {_params_key(params) for params in suggestions}
        fallback = tuple(
            params
            for params in _history_density_suggestions(signal_grid, history_entries, limit)
            if _params_key(params) not in existing
        )
        suggestions = (*suggestions, *fallback)[:limit]
    _write_study_snapshot(signal_name, optuna_dir, history_entries, suggestions)
    return suggestions


def _suggest_tpe_flat_params(
    *,
    study_label: str,
    choices_by_key: dict[str, tuple[object, ...]],
    history_entries: Sequence[LedgerEntry],
    flatten_entry_params,
    limit: int,
    optuna_dir: Path,
) -> tuple[dict[str, object], ...]:
    optuna_dir.mkdir(parents=True, exist_ok=True)
    suggestions = _optuna_library_flat_suggestions(
        study_label=study_label,
        choices_by_key=choices_by_key,
        history_entries=history_entries,
        flatten_entry_params=flatten_entry_params,
        limit=limit,
        optuna_dir=optuna_dir,
    )
    if len(suggestions) < limit:
        existing = {_params_key(params) for params in suggestions}
        fallback = tuple(
            params
            for params in _flat_neighbor_suggestions(choices_by_key, history_entries, flatten_entry_params, limit)
            if _params_key(params) not in existing
        )
        suggestions = (*suggestions, *fallback)[:limit]
    return suggestions


def _optuna_library_flat_suggestions(
    *,
    study_label: str,
    choices_by_key: dict[str, tuple[object, ...]],
    history_entries: Sequence[LedgerEntry],
    flatten_entry_params,
    limit: int,
    optuna_dir: Path,
) -> tuple[dict[str, object], ...]:
    try:
        import optuna  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        return ()
    distributions = {
        key: optuna.distributions.CategoricalDistribution(choices)
        for key, choices in choices_by_key.items()
    }
    sampler = optuna.samplers.TPESampler(seed=_stable_seed(study_label), warn_independent_sampling=False)
    study = optuna.create_study(
        direction="maximize",
        load_if_exists=True,
        sampler=sampler,
        storage=f"sqlite:///{optuna_dir / f'{study_label}.db'}",
        study_name=f"{study_label}_return_pct",
    )
    history_params = {
        entry.experiment_id: flatten_entry_params(entry)
        for entry in history_entries
    }
    seen_history = {_params_key(params) for params in history_params.values()}
    synced_experiment_ids = {
        trial.user_attrs.get("ledger_experiment_id")
        for trial in study.get_trials(deepcopy=False)
        if trial.user_attrs.get("ledger_experiment_id") is not None
    }
    for entry in history_entries:
        if entry.experiment_id in synced_experiment_ids:
            continue
        params = history_params[entry.experiment_id]
        if not _params_match_distributions(params, distributions):
            continue
        study.add_trial(
            optuna.trial.create_trial(
                params=params,
                distributions=distributions,
                value=entry.metric("return_pct"),
                user_attrs={"ledger_experiment_id": entry.experiment_id},
            )
        )
    suggestions: list[dict[str, object]] = []
    seen_suggestions = {
        _params_key(trial.params)
        for trial in study.get_trials(deepcopy=False)
        if trial.params
    }
    attempts = max(limit * 8, 32)
    for _ in range(attempts):
        trial = study.ask(fixed_distributions=distributions)
        params = dict(trial.params)
        params_key = _params_key(params)
        if params_key in seen_history or params_key in seen_suggestions:
            study.tell(trial, state=optuna.trial.TrialState.FAIL)
            continue
        suggestions.append(params)
        seen_suggestions.add(params_key)
        if len(suggestions) >= limit:
            break
    return tuple(suggestions)


def _flat_neighbor_suggestions(
    choices_by_key: dict[str, tuple[object, ...]],
    history_entries: Sequence[LedgerEntry],
    flatten_entry_params,
    limit: int,
) -> tuple[dict[str, object], ...]:
    if not history_entries:
        return ()
    best_entry = max(history_entries, key=lambda entry: entry.metric("return_pct"))
    base_params = flatten_entry_params(best_entry)
    seen_history = {_params_key(flatten_entry_params(entry)) for entry in history_entries}
    suggestions: list[dict[str, object]] = []
    for key in sorted(choices_by_key):
        for choice in choices_by_key[key]:
            params = {**base_params, key: choice}
            params_key = _params_key(params)
            if params_key in seen_history:
                continue
            suggestions.append(params)
            if len(suggestions) >= limit:
                return tuple(suggestions)
    return tuple(suggestions)


def _optuna_library_suggestions(
    signal_name: str,
    signal_grid: Sequence[dict[str, object]],
    history_entries: Sequence[LedgerEntry],
    limit: int,
    optuna_dir: Path,
) -> tuple[dict[str, object], ...]:
    try:
        import optuna  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        return ()
    distributions = {
        key: optuna.distributions.CategoricalDistribution(_grid_values(signal_grid, key))
        for key in _grid_param_keys(signal_grid)
    }
    sampler = optuna.samplers.TPESampler(seed=_stable_seed(signal_name), warn_independent_sampling=False)
    study = optuna.create_study(
        direction="maximize",
        load_if_exists=True,
        sampler=sampler,
        storage=f"sqlite:///{optuna_dir / f'{signal_name}.db'}",
        study_name=f"{signal_name}_return_pct",
    )
    seen_history = {_params_key(entry.spec.signal.params) for entry in history_entries}
    synced_experiment_ids = {
        trial.user_attrs.get("ledger_experiment_id")
        for trial in study.get_trials(deepcopy=False)
        if trial.user_attrs.get("ledger_experiment_id") is not None
    }
    for entry in history_entries:
        if entry.experiment_id in synced_experiment_ids:
            continue
        params = {
            key: entry.spec.signal.params[key]
            for key in distributions
            if key in entry.spec.signal.params
        }
        if not _params_match_distributions(params, distributions):
            continue
        study.add_trial(
            optuna.trial.create_trial(
                params=params,
                distributions=distributions,
                value=entry.metric("return_pct"),
                user_attrs={"ledger_experiment_id": entry.experiment_id},
            )
        )
    suggestions: list[dict[str, object]] = []
    seen_suggestions = {
        _params_key(trial.params)
        for trial in study.get_trials(deepcopy=False)
        if trial.params
    }
    attempts = max(limit * 8, 32)
    for _ in range(attempts):
        trial = study.ask(fixed_distributions=distributions)
        params = dict(trial.params)
        params_key = _params_key(params)
        if params_key in seen_history or params_key in seen_suggestions:
            study.tell(trial, state=optuna.trial.TrialState.FAIL)
            continue
        suggestions.append(params)
        seen_suggestions.add(params_key)
        if len(suggestions) >= limit:
            break
    return tuple(suggestions)


def _params_match_distributions(params: dict[str, object], distributions: dict[str, object]) -> bool:
    if set(params) != set(distributions):
        return False
    for key, distribution in distributions.items():
        choices = getattr(distribution, "choices", None)
        if choices is not None and params[key] not in choices:
            return False
    return True


def _history_density_suggestions(
    signal_grid: Sequence[dict[str, object]],
    history_entries: Sequence[LedgerEntry],
    limit: int,
) -> tuple[dict[str, object], ...]:
    seen_history = {_params_key(entry.spec.signal.params) for entry in history_entries}
    ranked = sorted(
        signal_grid,
        key=lambda params: (
            _predicted_objective(params, history_entries),
            _params_key(params),
        ),
        reverse=True,
    )
    unseen = [params for params in ranked if _params_key(params) not in seen_history]
    pool = unseen or ranked
    return tuple(dict(params) for params in pool[:limit])


def _predicted_objective(params: dict[str, object], history_entries: Sequence[LedgerEntry]) -> float:
    if not history_entries:
        return 0.0
    scores = []
    for entry in history_entries:
        distance = _normalized_param_distance(params, entry.spec.signal.params)
        scores.append(entry.metric("return_pct") * exp(-3.0 * distance))
    return max(scores, default=0.0)


def _normalized_param_distance(left: dict[str, object], right: dict[str, object]) -> float:
    keys = sorted(set(left) & set(right))
    if not keys:
        return 1.0
    distance = 0.0
    for key in keys:
        left_value = left[key]
        right_value = right[key]
        if isinstance(left_value, (int, float)) and isinstance(right_value, (int, float)):
            scale = max(abs(float(left_value)), abs(float(right_value)), 1.0)
            distance += min(abs(float(left_value) - float(right_value)) / scale, 1.0)
        elif left_value != right_value:
            distance += 1.0
    return distance / len(keys)


def _write_study_snapshot(
    signal_name: str,
    optuna_dir: Path,
    history_entries: Sequence[LedgerEntry],
    suggestions: Sequence[dict[str, object]],
) -> None:
    payload = {
        "signal_family": signal_name,
        "sampler": "optuna_tpe",
        "seed_evaluation_threshold": _OPTUNA_SEED_EVALUATION_THRESHOLD,
        "objective": "return_pct",
        "completed_trials": [
            {
                "experiment_id": entry.experiment_id,
                "spec_hash": entry.spec_hash,
                "return_pct": entry.metric("return_pct"),
                "params": dict(sorted(entry.spec.signal.params.items())),
            }
            for entry in history_entries
        ],
        "pending_suggestions": [dict(sorted(params.items())) for params in suggestions],
    }
    path = optuna_dir / f"{signal_name}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")


def _write_multi_signal_study_snapshot(
    shape_key: str,
    optuna_dir: Path,
    history_entries: Sequence[LedgerEntry],
    choices_by_key: dict[str, tuple[object, ...]],
    suggestions: Sequence[dict[str, object]],
) -> None:
    payload = {
        "signal_family": "multi_signal",
        "search_space_version": MULTI_SIGNAL_SEARCH_SPACE_VERSION,
        "shape_key": shape_key,
        "sampler": "optuna_tpe",
        "seed_evaluation_threshold": _OPTUNA_SEED_EVALUATION_THRESHOLD,
        "objective": "return_pct",
        "parameter_choices": {key: list(values) for key, values in sorted(choices_by_key.items())},
        "completed_trials": [
            {
                "experiment_id": entry.experiment_id,
                "spec_hash": entry.spec_hash,
                "return_pct": entry.metric("return_pct"),
                "params": _flatten_multi_signal_params(entry.spec.signal.params, choices_by_key),
            }
            for entry in history_entries
        ],
        "pending_suggestions": [dict(sorted(params.items())) for params in suggestions],
    }
    path = optuna_dir / f"multi_signal_{_stable_seed(shape_key):08x}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")


def _grid_param_keys(signal_grid: Sequence[dict[str, object]]) -> tuple[str, ...]:
    return tuple(sorted({key for params in signal_grid for key in params}))


def _grid_values(signal_grid: Sequence[dict[str, object]], key: str) -> tuple[object, ...]:
    return tuple(sorted({params[key] for params in signal_grid if key in params}, key=lambda value: str(value)))


def _params_key(params: dict[str, object]) -> str:
    return json.dumps(dict(sorted(params.items())), sort_keys=True, separators=(",", ":"), allow_nan=False)


def _stable_seed(value: str) -> int:
    return int(sha256(value.encode("utf-8")).hexdigest()[:8], 16)
