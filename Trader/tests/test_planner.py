from __future__ import annotations

import json
import sys
from types import SimpleNamespace
from pathlib import Path

from trader.ledger.entry import LedgerEntry
from trader.research.planner import (
    MULTI_SIGNAL_SEARCH_SPACE_VERSION,
    DeterministicPlanner,
    MultiSignalRuleShape,
    multi_signal_search_grammar,
)
from trader.strategies.registry import REGISTRY
from trader.strategies.spec import ExecConfig, SignalSpec, SizingSpec, StrategySpec


def test_multi_signal_search_grammar_is_versioned_and_validatable() -> None:
    grammar = multi_signal_search_grammar()

    assert grammar.version == MULTI_SIGNAL_SEARCH_SPACE_VERSION == "multi_signal_v1"
    assert grammar.entry_shapes
    assert grammar.exit_shapes
    assert all(len(shape.predicates) >= 3 for shape in grammar.entry_shapes)
    assert all(len(shape.predicates) >= 3 for shape in grammar.exit_shapes)
    assert all(
        predicate in grammar.predicate_param_grids
        for shape in (*grammar.entry_shapes, *grammar.exit_shapes)
        for predicate in shape.predicates
    )

    REGISTRY.validate_spec(
        StrategySpec(
            name="multi_signal_grammar_sample",
            signal=SignalSpec(
                "multi_signal",
                {
                    "entry_rule": _rule_payload(grammar.entry_shapes[0]),
                    "exit_rule": _rule_payload(grammar.exit_shapes[0]),
                },
            ),
        )
    )


def test_planner_reserves_frontier_and_each_enabled_family_before_truncation() -> None:
    planner = DeterministicPlanner(REGISTRY)
    parent = REGISTRY.validate_spec(
        StrategySpec(
            name="frontier_parent",
            signal=SignalSpec("ema_cross", {"fast_length": 20, "slow_length": 80, "signal_buffer_bps": 0.0}),
        )
    )

    planned = planner.plan(
        batch_size=4,
        frontier_specs=(("frontier_1", parent),),
        allowed_signal_families=("ema_cross", "breakout", "rsi_reversion"),
    )

    assert len(planned) == 4
    assert planned[0].generator_kind == "frontier_neighborhood"
    assert {item.spec.signal.name for item in planned if item.generator_kind == "grid"} == {
        "ema_cross",
        "breakout",
        "rsi_reversion",
    }


def test_planner_does_not_eagerly_expand_grid_combinations(monkeypatch) -> None:
    def lazy_combinations():
        yield (SizingSpec("fixed_fraction", {"fraction": 0.25}), ExecConfig(), ())
        raise AssertionError("planner consumed more combinations than the requested batch needed")

    monkeypatch.setattr("trader.research.planner._parameter_combinations", lazy_combinations)
    planner = DeterministicPlanner(REGISTRY)

    planned = planner.plan(batch_size=1, allowed_signal_families=("ema_cross",))

    assert len(planned) == 1
    assert planned[0].spec.signal.name == "ema_cross"


def test_planner_generates_valid_multi_signal_candidates_with_minimum_rule_sizes() -> None:
    planner = DeterministicPlanner(REGISTRY)

    planned = planner.plan(batch_size=24, allowed_signal_families=("multi_signal",))

    assert planned
    assert {item.generator_kind for item in planned} == {"multi_signal_grid"}
    assert {item.spec.signal.name for item in planned} == {"multi_signal"}
    assert all("multi_signal_v1" in item.spec.tags for item in planned)
    for item in planned:
        validated = REGISTRY.validate_spec(item.spec)
        assert len(validated.signal.params["entry_rule"]["signals"]) >= 3  # type: ignore[index]
        assert len(validated.signal.params["exit_rule"]["signals"]) >= 3  # type: ignore[index]


def test_planner_restart_changes_multi_signal_starting_region() -> None:
    planner = DeterministicPlanner(REGISTRY)

    first = planner.plan(
        batch_size=4,
        allowed_signal_families=("multi_signal",),
        restart_seed="loop_run",
        restart_index=0,
    )
    restarted = planner.plan(
        batch_size=4,
        allowed_signal_families=("multi_signal",),
        restart_seed="loop_run",
        restart_index=1,
    )

    assert first[0].spec.spec_hash() != restarted[0].spec.spec_hash()
    assert all(REGISTRY.validate_spec(item.spec) for item in restarted)


def test_planner_reserves_grid_slot_for_each_enabled_family_before_truncation() -> None:
    planner = DeterministicPlanner(REGISTRY)

    planned = planner.plan(
        batch_size=3,
        allowed_signal_families=("ema_cross", "breakout", "rsi_reversion"),
    )

    assert [item.spec.signal.name for item in planned] == ["ema_cross", "breakout", "rsi_reversion"]
    assert all(item.spec.filters == tuple() for item in planned)


def test_planner_emits_fixed_fraction_sizing_across_enabled_families() -> None:
    planner = DeterministicPlanner(REGISTRY)

    planned = planner.plan(
        batch_size=45,
        allowed_signal_families=("ema_cross", "breakout", "rsi_reversion"),
    )

    assert {
        (item.spec.signal.name, item.spec.sizing.name, item.spec.sizing.params["fraction"])
        for item in planned
    } >= {
        ("ema_cross", "fixed_fraction", 0.25),
        ("ema_cross", "fixed_fraction", 0.50),
        ("ema_cross", "fixed_fraction", 1.00),
        ("breakout", "fixed_fraction", 0.25),
        ("breakout", "fixed_fraction", 0.50),
        ("breakout", "fixed_fraction", 1.00),
        ("rsi_reversion", "fixed_fraction", 0.25),
        ("rsi_reversion", "fixed_fraction", 0.50),
        ("rsi_reversion", "fixed_fraction", 1.00),
    }
    assert all(REGISTRY.validate_spec(item.spec) for item in planned)


def test_planner_emits_session_time_execution_variants_across_enabled_families() -> None:
    planner = DeterministicPlanner(REGISTRY)

    planned = planner.plan(
        batch_size=45,
        allowed_signal_families=("ema_cross", "breakout", "rsi_reversion"),
    )

    assert {
        (
            item.spec.signal.name,
            item.spec.exec_config.entry_session_window,
            item.spec.exec_config.no_new_entry_minutes_before_close,
        )
        for item in planned
    } >= {
        ("ema_cross", "all", None),
        ("ema_cross", "first_30m", None),
        ("ema_cross", "last_30m", None),
        ("ema_cross", "avoid_midday", None),
        ("ema_cross", "all", 30),
        ("breakout", "all", None),
        ("breakout", "first_30m", None),
        ("breakout", "last_30m", None),
        ("breakout", "avoid_midday", None),
        ("breakout", "all", 30),
        ("rsi_reversion", "all", None),
        ("rsi_reversion", "first_30m", None),
        ("rsi_reversion", "last_30m", None),
        ("rsi_reversion", "avoid_midday", None),
        ("rsi_reversion", "all", 30),
    }
    assert all(REGISTRY.validate_spec(item.spec) for item in planned)


def test_planner_frontier_is_not_crowded_out_by_grid_ordering() -> None:
    planner = DeterministicPlanner(REGISTRY)
    parent = REGISTRY.validate_spec(
        StrategySpec(
            name="frontier_parent",
            signal=SignalSpec("ema_cross", {"fast_length": 20, "slow_length": 80, "signal_buffer_bps": 0.0}),
        )
    )

    planned = planner.plan(
        batch_size=2,
        frontier_specs=(("frontier_1", parent),),
        allowed_signal_families=("ema_cross", "breakout"),
    )

    assert any(item.generator_kind == "frontier_neighborhood" for item in planned)


def test_planner_reserves_frontier_slot_per_enabled_family() -> None:
    planner = DeterministicPlanner(REGISTRY)
    parents = (
        (
            "breakout_parent",
            REGISTRY.validate_spec(
                StrategySpec(
                    name="breakout_parent",
                    signal=SignalSpec("breakout", {"entry_window": 20, "exit_window": 10, "buffer_bps": 0.0}),
                )
            ),
        ),
        (
            "ema_parent",
            REGISTRY.validate_spec(
                StrategySpec(
                    name="ema_parent",
                    signal=SignalSpec("ema_cross", {"fast_length": 20, "slow_length": 80, "signal_buffer_bps": 0.0}),
                )
            ),
        ),
        (
            "rsi_parent",
            REGISTRY.validate_spec(
                StrategySpec(
                    name="rsi_parent",
                    signal=SignalSpec(
                        "rsi_reversion",
                        {"rsi_length": 14, "oversold_threshold": 30.0, "overbought_threshold": 70.0},
                    ),
                )
            ),
        ),
    )

    planned = planner.plan(
        batch_size=6,
        frontier_specs=parents,
        allowed_signal_families=("ema_cross", "breakout", "rsi_reversion"),
    )

    assert {
        (item.generator_kind, item.spec.signal.name)
        for item in planned
    } == {
        ("frontier_neighborhood", "ema_cross"),
        ("frontier_neighborhood", "breakout"),
        ("frontier_neighborhood", "rsi_reversion"),
        ("grid", "ema_cross"),
        ("grid", "breakout"),
        ("grid", "rsi_reversion"),
    }


def test_planner_emits_curated_confirmation_specs_before_default_budget_truncation() -> None:
    planner = DeterministicPlanner(REGISTRY)

    planned = planner.plan(
        batch_size=12,
        allowed_signal_families=("rsi_reversion", "vwap_deviation"),
    )
    confirmation_specs = [item.spec for item in planned if item.generator_kind == "confirmation_grid"]

    assert confirmation_specs
    assert any(item.generator_kind == "grid" for item in planned)
    assert all(REGISTRY.validate_spec(spec) for spec in confirmation_specs)
    assert {
        (spec.signal.name, spec.filters[0].name)
        for spec in confirmation_specs
    } >= {
        ("rsi_reversion", "intraday_volatility"),
        ("rsi_reversion", "vwap_distance"),
        ("vwap_deviation", "relative_volume"),
    }
    assert all("confirmation" in spec.tags for spec in confirmation_specs)


def test_planner_emits_curated_composite_specs_before_default_budget_truncation() -> None:
    planner = DeterministicPlanner(REGISTRY)

    planned = planner.plan(
        batch_size=64,
        allowed_signal_families=("ema_cross", "breakout", "rsi_reversion", "vwap_deviation", "composite"),
    )
    composite_specs = [item.spec for item in planned if item.generator_kind == "composite_grid"]

    assert composite_specs
    assert {spec.tags[1] for spec in composite_specs} == {
        "ema_rsi_all",
        "breakout_trend_volume",
        "vwap_rsi_all",
        "ema_breakout_rsi_vote",
        "ema_chop_rsi_all",
        "breakout_open_volume",
    }
    assert all(REGISTRY.validate_spec(spec) for spec in composite_specs)
    assert all("composite" in spec.tags for spec in composite_specs)


def test_planner_caps_composite_bucket_and_represents_canonical_shapes() -> None:
    planner = DeterministicPlanner(REGISTRY)

    planned = planner.plan(
        batch_size=200,
        allowed_signal_families=("composite",),
    )
    composite_plans = [item for item in planned if item.generator_kind == "composite_grid"]
    composite_specs = [item.spec for item in composite_plans]

    assert len(composite_specs) <= 24
    assert {spec.tags[1] for spec in composite_specs} == {
        "ema_rsi_all",
        "breakout_trend_volume",
        "vwap_rsi_all",
        "ema_breakout_rsi_vote",
        "ema_chop_rsi_all",
        "breakout_open_volume",
    }
    assert max(
        sum(1 for spec in composite_specs if spec.tags[1] == recipe)
        for recipe in {spec.tags[1] for spec in composite_specs}
    ) == 3
    assert any(
        spec.signal.name == "composite"
        and spec.signal.params["combiner"] == "all"
        and _child_names(spec) == ("ema_cross", "rsi_reversion")
        for spec in composite_specs
    )
    assert any(
        spec.signal.name == "composite"
        and spec.signal.params["combiner"] == "vote_k_of_n"
        and spec.signal.params["min_agreeing"] == 2
        and _child_names(spec) == ("ema_cross", "breakout", "rsi_reversion")
        for spec in composite_specs
    )
    assert any(
        spec.signal.name == "breakout"
        and {filter_spec.name for filter_spec in spec.filters} == {"relative_volume", "day_type"}
        and any(filter_spec.params.get("mode") == "trend" for filter_spec in spec.filters)
        for spec in composite_specs
    )
    assert any(
        spec.signal.name == "ema_cross"
        and any(filter_spec.name == "day_type" and filter_spec.params.get("mode") == "mean_reversion" for filter_spec in spec.filters)
        for spec in composite_specs
    )
    assert any(
        spec.signal.name == "breakout"
        and spec.exec_config.entry_session_window == "first_30m"
        and any(filter_spec.name == "relative_volume" for filter_spec in spec.filters)
        for spec in composite_specs
    )
    assert all(REGISTRY.validate_spec(spec) for spec in composite_specs)


def test_planner_waits_for_seed_history_before_optuna_bucket(tmp_path: Path) -> None:
    planner = DeterministicPlanner(REGISTRY)
    grid = REGISTRY.parameter_grid("rsi_reversion")
    history = tuple(
        _entry(f"rsi_{index}", "rsi_reversion", grid[index], return_pct=float(index))
        for index in range(9)
    )

    planned = planner.plan(
        batch_size=16,
        allowed_signal_families=("rsi_reversion",),
        history_entries=history,
        optuna_dir=tmp_path,
    )

    assert not any(item.generator_kind == "optuna_tpe" for item in planned)
    assert not (tmp_path / "rsi_reversion.json").exists()


def test_planner_adds_optuna_bucket_after_seed_history_and_persists_state(tmp_path: Path) -> None:
    planner = DeterministicPlanner(REGISTRY)
    grid = REGISTRY.parameter_grid("rsi_reversion")
    history = tuple(
        _entry(f"rsi_{index}", "rsi_reversion", grid[index], return_pct=float(index))
        for index in range(10)
    )

    planned = planner.plan(
        batch_size=20,
        allowed_signal_families=("rsi_reversion",),
        history_entries=history,
        optuna_dir=tmp_path,
    )
    optuna_plans = [item for item in planned if item.generator_kind == "optuna_tpe"]
    payload = json.loads((tmp_path / "rsi_reversion.json").read_text(encoding="utf-8"))

    assert optuna_plans
    assert all(REGISTRY.validate_spec(item.spec) for item in optuna_plans)
    assert {item.spec.signal.name for item in optuna_plans} == {"rsi_reversion"}
    assert all(item.parent_experiment_ids == ("rsi_9",) for item in optuna_plans)
    assert all(item.spec.tags == ("optuna_tpe",) for item in optuna_plans)
    assert payload["signal_family"] == "rsi_reversion"
    assert payload["objective"] == "return_pct"
    assert len(payload["completed_trials"]) == 10
    assert payload["pending_suggestions"]


def test_planner_loads_existing_optuna_study_without_fake_objectives(tmp_path: Path, monkeypatch) -> None:
    planner = DeterministicPlanner(REGISTRY)
    grid = REGISTRY.parameter_grid("rsi_reversion")
    history = tuple(
        _entry(f"rsi_{index}", "rsi_reversion", grid[index], return_pct=float(index))
        for index in range(10)
    )
    create_study_calls = []
    tell_calls = []
    added_trials = []

    class FakeDistribution:
        def __init__(self, choices):
            self.choices = tuple(choices)

    class FakeTrial:
        def __init__(self, params, user_attrs=None):
            self.params = params
            self.user_attrs = user_attrs or {}

    class FakeStudy:
        def __init__(self):
            self.trials = [
                FakeTrial(
                    dict(history[0].spec.signal.params),
                    user_attrs={"ledger_experiment_id": history[0].experiment_id},
                )
            ]
            self.ask_count = 0

        def get_trials(self, deepcopy=False):
            return list(self.trials)

        def add_trial(self, trial):
            self.trials.append(trial)
            added_trials.append(trial)

        def ask(self, fixed_distributions):
            self.ask_count += 1
            if self.ask_count == 1:
                params = dict(history[0].spec.signal.params)
            else:
                params = {
                    key: distribution.choices[-1]
                    for key, distribution in fixed_distributions.items()
                }
            trial = FakeTrial(params)
            self.trials.append(trial)
            return trial

        def tell(self, trial, value=None, state=None):
            tell_calls.append((trial, value, state))

    def create_trial(*, params, distributions, value, user_attrs):
        return FakeTrial(params, user_attrs=user_attrs)

    fake_optuna = SimpleNamespace(
        create_study=lambda **kwargs: create_study_calls.append(kwargs) or FakeStudy(),
        distributions=SimpleNamespace(CategoricalDistribution=FakeDistribution),
        samplers=SimpleNamespace(TPESampler=lambda **kwargs: object()),
        trial=SimpleNamespace(create_trial=create_trial, TrialState=SimpleNamespace(FAIL="FAIL")),
    )
    monkeypatch.setitem(sys.modules, "optuna", fake_optuna)

    planned = planner.plan(
        batch_size=20,
        allowed_signal_families=("rsi_reversion",),
        history_entries=history,
        optuna_dir=tmp_path,
    )

    assert any(item.generator_kind == "optuna_tpe" for item in planned)
    assert create_study_calls[0]["load_if_exists"] is True
    assert create_study_calls[0]["storage"].endswith("/rsi_reversion.db")
    assert {trial.user_attrs["ledger_experiment_id"] for trial in added_trials} == {
        entry.experiment_id for entry in history[1:]
    }
    assert all(value is None for _, value, _ in tell_calls)
    assert any(state == "FAIL" for _, _, state in tell_calls)


def test_planner_skips_optuna_seed_rows_outside_current_grid(tmp_path: Path, monkeypatch) -> None:
    planner = DeterministicPlanner(REGISTRY)
    grid = REGISTRY.parameter_grid("rsi_reversion")
    history = tuple(
        _entry(f"rsi_{index}", "rsi_reversion", grid[index], return_pct=float(index))
        for index in range(10)
    ) + (
        _entry(
            "rsi_stale",
            "rsi_reversion",
            {**grid[0], "oversold_threshold": 25.0},
            return_pct=99.0,
        ),
    )
    added_trials = []

    class FakeDistribution:
        def __init__(self, choices):
            self.choices = tuple(choices)

    class FakeTrial:
        def __init__(self, params, user_attrs=None):
            self.params = params
            self.user_attrs = user_attrs or {}

    class FakeStudy:
        def __init__(self):
            self.ask_count = 0

        def get_trials(self, deepcopy=False):
            return []

        def add_trial(self, trial):
            added_trials.append(trial)

        def ask(self, fixed_distributions):
            self.ask_count += 1
            params = {
                key: distribution.choices[self.ask_count % len(distribution.choices)]
                for key, distribution in fixed_distributions.items()
            }
            return FakeTrial(params)

        def tell(self, trial, value=None, state=None):
            pass

    fake_optuna = SimpleNamespace(
        create_study=lambda **kwargs: FakeStudy(),
        distributions=SimpleNamespace(CategoricalDistribution=FakeDistribution),
        samplers=SimpleNamespace(TPESampler=lambda **kwargs: object()),
        trial=SimpleNamespace(
            create_trial=lambda *, params, distributions, value, user_attrs: FakeTrial(params, user_attrs=user_attrs),
            TrialState=SimpleNamespace(FAIL="FAIL"),
        ),
    )
    monkeypatch.setitem(sys.modules, "optuna", fake_optuna)

    planner.plan(
        batch_size=20,
        allowed_signal_families=("rsi_reversion",),
        history_entries=history,
        optuna_dir=tmp_path,
    )

    assert {trial.user_attrs["ledger_experiment_id"] for trial in added_trials} == {
        entry.experiment_id for entry in history[:-1]
    }


def _entry(
    experiment_id: str,
    signal_name: str,
    params: dict[str, object],
    *,
    return_pct: float,
) -> LedgerEntry:
    spec = REGISTRY.validate_spec(
        StrategySpec(
            name=experiment_id,
            signal=SignalSpec(signal_name, params),
        )
    )
    return LedgerEntry(
        experiment_id=experiment_id,
        evaluation_key=f"{experiment_id}_key",
        status="completed",
        spec=spec,
        spec_hash=spec.spec_hash(),
        data_snapshot_id="snapshot",
        split_plan_id="split",
        cost_model_id="cost",
        aggregate_metrics={
            "return_pct": return_pct,
            "sharpe_like": 0.0,
            "max_drawdown_pct": 0.0,
        },
        fold_results=(),
        robustness_checks={},
        promotion_stage="exploratory",
        completed_at_utc="2026-01-01T00:00:00+00:00",
    )


def _child_names(spec: StrategySpec) -> tuple[str, ...]:
    children = spec.signal.params["children"]
    assert isinstance(children, list)
    return tuple(str(child["name"]) for child in children)


def _rule_payload(shape: MultiSignalRuleShape) -> dict[str, object]:
    grammar = multi_signal_search_grammar()
    rule: dict[str, object] = {
        "combiner": shape.combiner,
        "signals": [
            {"name": predicate, "params": grammar.predicate_param_grids[predicate][0]}
            for predicate in shape.predicates
        ],
    }
    if shape.k is not None:
        rule["k"] = shape.k
    return rule
