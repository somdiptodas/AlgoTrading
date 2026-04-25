from __future__ import annotations

from trader.research.planner import DeterministicPlanner
from trader.strategies.registry import REGISTRY
from trader.strategies.spec import SignalSpec, StrategySpec


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


def _child_names(spec: StrategySpec) -> tuple[str, ...]:
    children = spec.signal.params["children"]
    assert isinstance(children, list)
    return tuple(str(child["name"]) for child in children)
