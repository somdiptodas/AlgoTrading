from __future__ import annotations

from types import SimpleNamespace

from trader.research.suppressor import RegionSuppressor, WithinRunStageAFailureMemory, _parameter_scales, spec_distance
from trader.strategies.registry import REGISTRY
from trader.strategies.spec import SignalSpec, StrategySpec


def _spec(name: str, signal_name: str, params: dict[str, int | float]) -> StrategySpec:
    return StrategySpec(name=name, signal=SignalSpec(signal_name, params))


def _failure(
    experiment_id: str,
    spec: StrategySpec,
    robustness_checks: dict[str, bool],
) -> SimpleNamespace:
    return SimpleNamespace(
        experiment_id=experiment_id,
        spec=spec,
        robustness_checks=robustness_checks,
    )


def _stage_a_result(experiment_id: str, spec: StrategySpec) -> SimpleNamespace:
    return SimpleNamespace(
        experiment_id=experiment_id,
        spec=spec,
        robustness_checks={
            "stage_a_pass": False,
            "stage_a_reject_non_positive_return": True,
        },
    )


def test_spec_distance_uses_registry_parameter_ranges() -> None:
    left = _spec(
        "left",
        "ema_cross",
        {"fast_length": 20, "slow_length": 80, "signal_buffer_bps": 0.0},
    ).to_payload(include_name=False)
    right = _spec(
        "right",
        "ema_cross",
        {"fast_length": 22, "slow_length": 80, "signal_buffer_bps": 0.0},
    ).to_payload(include_name=False)

    scales = _parameter_scales(REGISTRY)["ema_cross"]

    assert round(spec_distance(left, right, parameter_scales=scales), 6) == round((2.0 / 26.0) / 3.0, 6)


def test_spec_distance_keeps_legacy_scaling_without_geometry() -> None:
    left = _spec(
        "left",
        "ema_cross",
        {"fast_length": 20, "slow_length": 80, "signal_buffer_bps": 0.0},
    ).to_payload(include_name=False)
    right = _spec(
        "right",
        "ema_cross",
        {"fast_length": 22, "slow_length": 80, "signal_buffer_bps": 0.0},
    ).to_payload(include_name=False)

    assert round(spec_distance(left, right), 6) == round((2.0 / 22.0) / 3.0, 6)


def test_suppressor_weights_failure_types() -> None:
    spec = _spec("candidate", "breakout", {"entry_window": 20, "exit_window": 10, "buffer_bps": 0.0})
    neighborhood = RegionSuppressor(
        (
            _failure(
                "neighborhood_failure",
                spec,
                {"neighborhood_pass": False, "fold_consistency_pass": True},
            ),
        ),
        radius=0.2,
        weight_cap=40.0,
    )
    drawdown = RegionSuppressor(
        (
            _failure(
                "drawdown_failure",
                spec,
                {"drawdown_pass": False, "fold_consistency_pass": True},
            ),
        ),
        radius=0.2,
        weight_cap=40.0,
    )

    assert neighborhood.suppression_weight(spec) > drawdown.suppression_weight(spec)


def test_large_suppression_requires_repeated_nearby_failures() -> None:
    spec = _spec("candidate", "rsi_reversion", {"rsi_length": 14, "oversold_threshold": 30.0, "overbought_threshold": 70.0})
    first_failure = _failure("failure_1", spec, {"neighborhood_pass": False})
    second_failure = _failure("failure_2", spec, {"neighborhood_pass": False})

    single = RegionSuppressor((first_failure,), radius=0.2, weight_cap=40.0)
    repeated = RegionSuppressor((first_failure, second_failure), radius=0.2, weight_cap=40.0)

    assert single.suppression_weight(spec) == 20.0
    assert repeated.suppression_weight(spec) == 40.0


def test_suppressor_keeps_cross_family_failures_isolated() -> None:
    ema_failure = _failure(
        "failure_1",
        _spec("ema_failure", "ema_cross", {"fast_length": 20, "slow_length": 80, "signal_buffer_bps": 0.0}),
        {"neighborhood_pass": False},
    )
    rsi_candidate = _spec(
        "candidate",
        "rsi_reversion",
        {"rsi_length": 14, "oversold_threshold": 30.0, "overbought_threshold": 70.0},
    )

    assert RegionSuppressor((ema_failure,), radius=0.2, weight_cap=40.0).assess(rsi_candidate) is None


def test_within_run_stage_a_memory_suppresses_after_two_nearby_failures() -> None:
    spec = _spec("candidate", "ema_cross", {"fast_length": 20, "slow_length": 80, "signal_buffer_bps": 0.0})
    memory = WithinRunStageAFailureMemory(radius=0.2, weight_cap=40.0)

    memory.record(_stage_a_result("failure_1", spec))
    assert memory.assess(spec) is None

    memory.record(_stage_a_result("failure_2", spec))
    record = memory.assess(spec)

    assert record is not None
    assert record.failure_count_in_radius == 2
    assert record.failed_check_names == ("stage_a_reject_non_positive_return",)


def test_within_run_stage_a_memory_keeps_far_and_cross_family_failures_isolated() -> None:
    failed = _spec("failed", "ema_cross", {"fast_length": 20, "slow_length": 80, "signal_buffer_bps": 0.0})
    far = _spec("far", "ema_cross", {"fast_length": 34, "slow_length": 144, "signal_buffer_bps": 3.0})
    breakout = _spec("breakout", "breakout", {"entry_window": 20, "exit_window": 10, "buffer_bps": 0.0})
    memory = WithinRunStageAFailureMemory(radius=0.1, weight_cap=40.0)
    memory.record(_stage_a_result("failure_1", failed))
    memory.record(_stage_a_result("failure_2", failed))

    assert memory.assess(far) is None
    assert memory.assess(breakout) is None
