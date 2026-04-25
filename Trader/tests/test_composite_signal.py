from __future__ import annotations

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from trader.data.models import MarketBar
from trader.strategies.registry import REGISTRY
from trader.strategies.signals import composite
from trader.strategies.spec import SignalSpec, StrategySpec


NEW_YORK = ZoneInfo("America/New_York")


def _bar(index: int, close: float, *, vwap: float | None = None) -> MarketBar:
    timestamp = datetime(2026, 1, 5, 9, 30, tzinfo=NEW_YORK) + timedelta(minutes=index)
    timestamp_utc = timestamp.astimezone(timezone.utc)
    return MarketBar(
        timestamp_ms=int(timestamp_utc.timestamp() * 1000),
        timestamp_utc=timestamp_utc.isoformat(),
        open=close,
        high=close + 0.25,
        low=close - 0.25,
        close=close,
        volume=1_000.0,
        vwap=vwap if vwap is not None else close,
    )


def _fake_handler(regime: list[bool], *, required_history: int = 0) -> dict[str, object]:
    return {
        "normalize_params": lambda params: dict(params),
        "required_history": lambda params: required_history,
        "generate_regime": lambda history, test, params: regime[:len(test)],
        "neighbors": lambda params: tuple(),
    }


@pytest.mark.parametrize(
        ("params", "expected"),
        (
            ({"combiner": "all"}, [True, False, False]),
            ({"combiner": "any"}, [True, True, True]),
            ({"combiner": "vote_k_of_n", "min_agreeing": 2}, [True, True, True]),
            ({"combiner": "vote_k_of_n", "min_agreeing": 3}, [True, False, False]),
            ({"combiner": "primary_plus_confirmations", "primary_index": 1}, [True, False, False]),
        ),
    )
def test_composite_combines_child_regimes(monkeypatch: pytest.MonkeyPatch, params: dict[str, object], expected: list[bool]) -> None:
    monkeypatch.setattr(
        composite,
        "_SIGNAL_HANDLERS",
            {
                "a": _fake_handler([True, False, True]),
                "b": _fake_handler([True, True, False]),
                "c": _fake_handler([True, True, True]),
            },
        )
    normalized = composite.normalize_params(
        {
            **params,
            "children": (
                {"name": "a", "params": {}},
                {"name": "b", "params": {}},
                {"name": "c", "params": {}},
            ),
        }
    )

    assert composite.generate_regime((), (_bar(0, 100.0), _bar(1, 101.0), _bar(2, 102.0)), normalized) == expected


def test_composite_required_history_uses_max_child_history(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        composite,
        "_SIGNAL_HANDLERS",
        {
            "a": _fake_handler([], required_history=5),
            "b": _fake_handler([], required_history=13),
        },
    )
    params = composite.normalize_params(
        {
            "combiner": "all",
            "children": (
                {"name": "a", "params": {}},
                {"name": "b", "params": {}},
            ),
        }
    )

    assert composite.required_history(params) == 13


def test_registry_validates_and_hashes_composite_specs() -> None:
    spec = StrategySpec(
        name="ema_and_breakout",
        signal=SignalSpec(
            "composite",
            {
                "combiner": "all",
                "children": [
                    {"name": "ema_cross", "params": {"fast_length": 2, "slow_length": 3}},
                    {"name": "breakout", "params": {"entry_window": 4, "exit_window": 2}},
                ],
            },
        ),
    )

    validated = REGISTRY.validate_spec(spec)
    round_tripped = REGISTRY.validate_spec(StrategySpec.from_payload(validated.to_payload()))

    assert REGISTRY.required_history(validated) == 4
    assert validated.spec_hash() == round_tripped.spec_hash()


def test_registry_generates_composite_regime_from_real_children() -> None:
    bars = tuple(_bar(index, 100.0 + index) for index in range(12))
    spec = REGISTRY.validate_spec(
        StrategySpec(
            name="ema_or_breakout",
            signal=SignalSpec(
                "composite",
                {
                    "combiner": "any",
                    "children": [
                        {"name": "ema_cross", "params": {"fast_length": 2, "slow_length": 3}},
                        {"name": "breakout", "params": {"entry_window": 4, "exit_window": 2}},
                    ],
                },
            ),
        )
    )

    regime = REGISTRY.generate_regime(spec, bars[:4], bars[4:])

    assert len(regime) == len(bars[4:])
    assert any(regime)


def test_composite_neighbors_vary_one_child_and_are_bounded() -> None:
    params = REGISTRY.validate_spec(
        StrategySpec(
            name="composite_neighbors",
            signal=SignalSpec(
                "composite",
                {
                    "combiner": "vote_k_of_n",
                    "min_agreeing": 1,
                    "children": [
                        {"name": "ema_cross", "params": {"fast_length": 8, "slow_length": 34}},
                        {"name": "rsi_reversion", "params": {"rsi_length": 14}},
                    ],
                },
            ),
        )
    ).signal.params

    neighbors = composite.neighbors(params)

    assert 0 < len(neighbors) <= 6
    assert all(neighbor["combiner"] == "vote_k_of_n" for neighbor in neighbors)
    assert all(neighbor["min_agreeing"] == 1 for neighbor in neighbors)


def test_composite_validation_rejects_bad_children_and_nested_non_finite_values() -> None:
    with pytest.raises(ValueError, match="Unknown composite child signal handler"):
        REGISTRY.validate_spec(
            StrategySpec(
                name="bad_child",
                signal=SignalSpec(
                    "composite",
                    {
                        "combiner": "all",
                        "children": [
                            {"name": "ema_cross", "params": {}},
                            {"name": "missing", "params": {}},
                        ],
                    },
                ),
            )
        )
    with pytest.raises(ValueError, match="composite.children\\[0\\].params.signal_buffer_bps must be finite"):
        REGISTRY.validate_spec(
            StrategySpec(
                name="bad_nested_nan",
                signal=SignalSpec(
                    "composite",
                    {
                        "combiner": "all",
                        "children": [
                            {"name": "ema_cross", "params": {"signal_buffer_bps": float("nan")}},
                            {"name": "breakout", "params": {}},
                        ],
                    },
                ),
            )
        )


def test_composite_signal_does_not_peek_forward() -> None:
    history_bars = tuple(_bar(index, 100.0 + index) for index in range(20))
    base_test_bars = tuple(_bar(20 + index, 120.0 + index) for index in range(3))
    altered_test_bars = list(base_test_bars)
    altered_test_bars[-1] = _bar(22, 1_000.0)
    params = REGISTRY.validate_spec(
        StrategySpec(
            name="no_peek_composite",
            signal=SignalSpec(
                "composite",
                {
                    "combiner": "all",
                    "children": [
                        {"name": "ema_cross", "params": {"fast_length": 2, "slow_length": 3}},
                        {"name": "breakout", "params": {"entry_window": 4, "exit_window": 2}},
                    ],
                },
            ),
        )
    ).signal.params

    base_regime = composite.generate_regime(history_bars, base_test_bars, params)
    altered_regime = composite.generate_regime(history_bars, tuple(altered_test_bars), params)

    assert base_regime[0] == altered_regime[0]
