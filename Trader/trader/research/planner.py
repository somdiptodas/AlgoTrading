from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from math import exp
from pathlib import Path
from typing import Sequence

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
    ) -> tuple[PlannedSpec, ...]:
        allowed = tuple(allowed_signal_families or ("ema_cross", "breakout"))
        optuna_buckets = self._optuna_candidate_buckets(
            allowed_signal_families=allowed,
            history_entries=history_entries,
            optuna_dir=optuna_dir,
        )
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
            + optuna_buckets
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

    def _optuna_candidate_buckets(
        self,
        *,
        allowed_signal_families: Sequence[str],
        history_entries: Sequence[LedgerEntry],
        optuna_dir: Path | None,
    ) -> list[list[PlannedSpec]]:
        if optuna_dir is None:
            return []
        completed_by_family = _completed_entries_by_family(history_entries)
        buckets: list[list[PlannedSpec]] = []
        for signal_name in allowed_signal_families:
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


def _completed_entries_by_family(entries: Sequence[LedgerEntry]) -> dict[str, tuple[LedgerEntry, ...]]:
    grouped: dict[str, list[LedgerEntry]] = {}
    for entry in entries:
        if entry.status != "completed":
            continue
        grouped.setdefault(entry.spec.signal.name, []).append(entry)
    return {family: tuple(items) for family, items in grouped.items()}


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
        if set(params) != set(distributions):
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


def _grid_param_keys(signal_grid: Sequence[dict[str, object]]) -> tuple[str, ...]:
    return tuple(sorted({key for params in signal_grid for key in params}))


def _grid_values(signal_grid: Sequence[dict[str, object]], key: str) -> tuple[object, ...]:
    return tuple(sorted({params[key] for params in signal_grid if key in params}, key=lambda value: str(value)))


def _params_key(params: dict[str, object]) -> str:
    return json.dumps(dict(sorted(params.items())), sort_keys=True, separators=(",", ":"), allow_nan=False)


def _stable_seed(value: str) -> int:
    return int(sha256(value.encode("utf-8")).hexdigest()[:8], 16)
