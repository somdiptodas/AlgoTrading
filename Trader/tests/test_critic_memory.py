from __future__ import annotations

import json

from trader.ledger.entry import LedgerEntry
from trader.research.critic_memory import CriticRegionMemory
from trader.strategies.spec import SignalSpec, StrategySpec


def _spec(name: str, signal_name: str, params: dict[str, int | float]) -> StrategySpec:
    return StrategySpec(name=name, signal=SignalSpec(signal_name, params))


def _entry(
    experiment_id: str,
    spec: StrategySpec,
    *,
    critique: dict[str, object] | None,
) -> LedgerEntry:
    return LedgerEntry(
        experiment_id=experiment_id,
        evaluation_key=f"{experiment_id}_key",
        status="completed",
        spec=spec,
        spec_hash=spec.spec_hash(),
        data_snapshot_id="snapshot",
        split_plan_id="split",
        cost_model_id="cost",
        aggregate_metrics={"return_pct": -1.0},
        fold_results=(),
        robustness_checks={},
        promotion_stage="exploratory",
        critique=critique,
        completed_at_utc="2026-01-01T00:00:00+00:00",
    )


def test_critic_region_memory_penalizes_nearby_same_family_only() -> None:
    failed = _spec("failed", "ema_cross", {"fast_length": 20, "slow_length": 80, "signal_buffer_bps": 0.0})
    memory = CriticRegionMemory.from_entries(
        (
            _entry(
                "failed_1",
                failed,
                critique={"planning_penalties": {"benchmark_failure": 10.0}},
            ),
        )
    )

    near = _spec("near", "ema_cross", {"fast_length": 22, "slow_length": 80, "signal_buffer_bps": 0.0})
    far = _spec("far", "ema_cross", {"fast_length": 34, "slow_length": 144, "signal_buffer_bps": 3.0})
    cross_family = _spec("breakout", "breakout", {"entry_window": 20, "exit_window": 10, "buffer_bps": 0.0})

    assert memory.penalty(near) > 0.0
    assert memory.penalty(far) == 0.0
    assert memory.penalty(cross_family) == 0.0


def test_critic_region_memory_round_trips_stable_sorted_json(tmp_path) -> None:
    ema = _spec("ema", "ema_cross", {"fast_length": 20, "slow_length": 80, "signal_buffer_bps": 0.0})
    breakout = _spec("breakout", "breakout", {"entry_window": 20, "exit_window": 10, "buffer_bps": 0.0})
    memory = CriticRegionMemory.from_entries(
        (
            _entry("z_ema", ema, critique={"planning_penalties": {"non_positive_return": 12.0}}),
            _entry("a_breakout", breakout, critique={"planning_penalties": {"benchmark_failure": 10.0}}),
            _entry("ignored", ema, critique=None),
        )
    )
    path = tmp_path / "critic_memory.json"

    memory.write(path)
    loaded = CriticRegionMemory.load(path)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert [record["experiment_id"] for record in payload["records"]] == ["a_breakout", "z_ema"]
    assert loaded.record_count == 2
    assert loaded.penalty(ema) == memory.penalty(ema)


def test_missing_or_malformed_critic_region_memory_is_empty(tmp_path) -> None:
    missing = CriticRegionMemory.load(tmp_path / "missing.json")
    malformed_path = tmp_path / "malformed.json"
    malformed_path.write_text("{not valid", encoding="utf-8")
    malformed = CriticRegionMemory.load(malformed_path)

    spec = _spec("ema", "ema_cross", {"fast_length": 20, "slow_length": 80, "signal_buffer_bps": 0.0})
    assert missing.record_count == 0
    assert malformed.record_count == 0
    assert missing.penalty(spec) == 0.0
    assert malformed.penalty(spec) == 0.0
