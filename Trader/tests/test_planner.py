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
