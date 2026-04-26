from __future__ import annotations

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from trader.data.models import MarketBar
from trader.strategies.registry import REGISTRY
from trader.strategies.spec import SignalSpec, StrategySpec


NEW_YORK = ZoneInfo("America/New_York")


def _bar(index: int, close: float, *, volume: float = 1_000.0, vwap: float | None = 100.0) -> MarketBar:
    timestamp = datetime(2026, 1, 5, 9, 30, tzinfo=NEW_YORK) + timedelta(minutes=index)
    timestamp_utc = timestamp.astimezone(timezone.utc)
    return MarketBar(
        timestamp_ms=int(timestamp_utc.timestamp() * 1000),
        timestamp_utc=timestamp_utc.isoformat(),
        open=close,
        high=close + 0.25,
        low=close - 0.25,
        close=close,
        volume=volume,
        vwap=vwap,
    )


def _multi_signal_params() -> dict[str, object]:
    return {
        "entry_rule": {
            "combiner": "all",
            "signals": [
                {"name": "rsi_below", "params": {"length": "2", "threshold": "30"}},
                {"name": "ema_trend_up", "params": {"fast": "2", "slow": "3"}},
                {"name": "vwap_distance", "params": {"side": "below", "min_bps": "10"}},
            ],
        },
        "exit_rule": {
            "combiner": "any",
            "signals": [
                {"name": "rsi_above", "params": {"length": "2", "threshold": "70"}},
                {"name": "ema_trend_down", "params": {"fast": "2", "slow": "3"}},
                {"name": "vwap_reclaimed", "params": {"min_bps": "0"}},
            ],
        },
    }


def test_multi_signal_normalizes_rules_and_child_predicate_params() -> None:
    validated = REGISTRY.validate_spec(
        StrategySpec(
            name="multi_signal_validation",
            signal=SignalSpec("multi_signal", _multi_signal_params()),
        )
    )

    assert validated.signal.params == {
        "entry_rule": {
            "combiner": "all",
            "signals": [
                {"name": "rsi_below", "params": {"length": 2, "threshold": 30.0}},
                {"name": "ema_trend_up", "params": {"fast": 2, "slow": 3, "buffer_bps": 0.0}},
                {
                    "name": "vwap_distance",
                    "params": {"side": "below", "min_bps": 10.0, "max_bps": 100_000.0},
                },
            ],
        },
        "exit_rule": {
            "combiner": "any",
            "signals": [
                {"name": "rsi_above", "params": {"length": 2, "threshold": 70.0}},
                {"name": "ema_trend_down", "params": {"fast": 2, "slow": 3, "buffer_bps": 0.0}},
                {"name": "vwap_reclaimed", "params": {"min_bps": 0.0}},
            ],
        },
    }


def test_multi_signal_validation_rejects_bad_rule_shapes() -> None:
    with pytest.raises(ValueError, match="multi_signal.entry_rule must be a rule payload"):
        REGISTRY.validate_spec(
            StrategySpec(
                name="bad_entry_rule",
                signal=SignalSpec("multi_signal", {"exit_rule": _multi_signal_params()["exit_rule"]}),
            )
        )

    params = _multi_signal_params()
    entry_rule = dict(params["entry_rule"])  # type: ignore[arg-type]
    entry_rule["signals"] = [
        {"name": "missing", "params": {}},
        {"name": "ema_trend_up", "params": {"fast": 2, "slow": 3}},
        {"name": "vwap_distance", "params": {"side": "below"}},
    ]
    params["entry_rule"] = entry_rule
    with pytest.raises(ValueError, match="Unknown atomic predicate"):
        REGISTRY.validate_spec(
            StrategySpec(
                name="bad_child_predicate",
                signal=SignalSpec("multi_signal", params),
            )
        )


def test_multi_signal_rejects_entry_rules_with_fewer_than_three_signals() -> None:
    params = _multi_signal_params()
    entry_rule = dict(params["entry_rule"])  # type: ignore[arg-type]
    entry_rule["signals"] = entry_rule["signals"][:2]  # type: ignore[index]
    params["entry_rule"] = entry_rule

    with pytest.raises(ValueError, match="entry_rule\\.signals must contain at least 3"):
        REGISTRY.validate_spec(
            StrategySpec(
                name="short_entry_rule",
                signal=SignalSpec("multi_signal", params),
            )
        )


def test_registry_generate_decisions_wraps_legacy_regimes() -> None:
    bars = tuple(_bar(index, 100.0 + index) for index in range(6))
    decisions = REGISTRY.generate_decisions(
        StrategySpec(
            name="legacy_ema_decisions",
            signal=SignalSpec("ema_cross", {"fast_length": 2, "slow_length": 3}),
        ),
        bars[:3],
        bars[3:],
    )

    assert len(decisions) == 3
    assert decisions[0].entry.votes[0].name == "legacy_regime"
