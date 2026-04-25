from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from trader.data.models import AggregateBar, MarketBar
from trader.data.storage import SQLiteBarStore
from trader.data.view import DataView
from trader.ledger.entry import market_bar_from_payload, market_bar_to_payload
from trader.research.planner import DeterministicPlanner
from trader.strategies.registry import REGISTRY
from trader.strategies.signals import vwap_deviation
from trader.strategies.spec import SignalSpec, StrategySpec


NEW_YORK = ZoneInfo("America/New_York")


def _bar(index: int, close: float, *, vwap: float | None = 100.0) -> MarketBar:
    timestamp = (datetime(2026, 1, 5, 9, 30, tzinfo=NEW_YORK) + timedelta(minutes=index)).astimezone(timezone.utc)
    return MarketBar(
        timestamp_ms=int(timestamp.timestamp() * 1000),
        timestamp_utc=timestamp.isoformat(),
        open=close,
        high=close + 0.25,
        low=close - 0.25,
        close=close,
        volume=1_000.0,
        vwap=vwap,
    )


def _session_bar(day: int, minute: int, close: float, *, vwap: float | None = 100.0) -> MarketBar:
    timestamp = (
        datetime(2026, 1, 5 + day, 9, 30, tzinfo=NEW_YORK) + timedelta(minutes=minute)
    ).astimezone(timezone.utc)
    return MarketBar(
        timestamp_ms=int(timestamp.timestamp() * 1000),
        timestamp_utc=timestamp.isoformat(),
        open=close,
        high=close + 0.25,
        low=close - 0.25,
        close=close,
        volume=1_000.0,
        vwap=vwap,
    )


def test_data_view_preserves_vwap_and_uses_it_in_snapshot_hash(tmp_path: Path) -> None:
    timestamp = datetime(2026, 1, 5, 9, 30, tzinfo=NEW_YORK).astimezone(timezone.utc)
    store = SQLiteBarStore(tmp_path / "market.db")
    store.upsert_bars(
        (
            AggregateBar(
                ticker="SPY",
                multiplier=1,
                timespan="minute",
                timestamp_ms=int(timestamp.timestamp() * 1000),
                open=100.0,
                high=101.0,
                low=99.0,
                close=99.5,
                volume=1_000.0,
                vwap=100.0,
                transactions=10,
            ),
        )
    )
    bars = DataView(tmp_path / "market.db").bars("SPY", 1, "minute", regular_session_only=True)

    assert bars[0].vwap == 100.0
    assert DataView.snapshot_hash(bars) != DataView.snapshot_hash((replace(bars[0], vwap=100.5),))


def test_market_bar_payload_round_trips_vwap() -> None:
    bar = _bar(0, 99.5, vwap=100.0)

    assert market_bar_from_payload(market_bar_to_payload(bar)) == bar


def test_vwap_deviation_enters_below_vwap_and_exits_on_reversion() -> None:
    params = vwap_deviation.normalize_params(
        {"entry_deviation_bps": 25.0, "exit_deviation_bps": 0.0, "max_hold_bars": 10}
    )
    bars = (
        _bar(0, 99.70),
        _bar(1, 99.80),
        _bar(2, 100.00),
    )

    assert vwap_deviation.generate_regime((), bars, params) == [True, True, False]


def test_vwap_deviation_defaults_to_churn_control_params() -> None:
    params = vwap_deviation.normalize_params(
        {"entry_deviation_bps": 25.0, "exit_deviation_bps": 0.0, "max_hold_bars": 10}
    )

    assert params["min_bars_between_entries"] == 30
    assert params["max_entries_per_session"] == 4


def test_vwap_deviation_exits_after_timeout() -> None:
    params = vwap_deviation.normalize_params(
        {
            "entry_deviation_bps": 25.0,
            "exit_deviation_bps": 0.0,
            "max_hold_bars": 2,
            "min_bars_between_entries": 0,
        }
    )
    bars = (
        _bar(0, 99.70),
        _bar(1, 99.80),
        _bar(2, 99.85),
        _bar(3, 99.70),
    )

    assert vwap_deviation.generate_regime((), bars, params) == [True, True, False, True]


def test_vwap_deviation_waits_min_bars_between_entries() -> None:
    params = vwap_deviation.normalize_params(
        {
            "entry_deviation_bps": 25.0,
            "exit_deviation_bps": 0.0,
            "max_hold_bars": 1,
            "min_bars_between_entries": 2,
            "max_entries_per_session": 4,
        }
    )
    bars = (
        _bar(0, 99.70),
        _bar(1, 100.00),
        _bar(2, 99.70),
        _bar(3, 99.70),
        _bar(4, 99.70),
    )

    assert vwap_deviation.generate_regime((), bars, params) == [True, False, False, False, True]


def test_vwap_deviation_caps_entries_per_session_and_resets_next_session() -> None:
    params = vwap_deviation.normalize_params(
        {
            "entry_deviation_bps": 25.0,
            "exit_deviation_bps": 0.0,
            "max_hold_bars": 1,
            "min_bars_between_entries": 0,
            "max_entries_per_session": 2,
        }
    )
    bars = (
        _session_bar(0, 0, 99.70),
        _session_bar(0, 1, 100.00),
        _session_bar(0, 2, 99.70),
        _session_bar(0, 3, 100.00),
        _session_bar(0, 4, 99.70),
        _session_bar(1, 0, 99.70),
    )

    assert vwap_deviation.generate_regime((), bars, params) == [True, False, True, False, False, True]


def test_vwap_deviation_missing_vwap_disables_signal() -> None:
    params = vwap_deviation.normalize_params(
        {"entry_deviation_bps": 25.0, "exit_deviation_bps": 0.0, "max_hold_bars": 10}
    )

    assert vwap_deviation.generate_regime((), (_bar(0, 99.50, vwap=None),), params) == [False]


def test_vwap_deviation_missing_vwap_exit_starts_cooldown() -> None:
    params = vwap_deviation.normalize_params(
        {
            "entry_deviation_bps": 25.0,
            "exit_deviation_bps": 0.0,
            "max_hold_bars": 10,
            "min_bars_between_entries": 2,
            "max_entries_per_session": 4,
        }
    )
    bars = (
        _bar(0, 99.70),
        _bar(1, 99.70, vwap=None),
        _bar(2, 99.70),
        _bar(3, 99.70),
        _bar(4, 99.70),
    )

    assert vwap_deviation.generate_regime((), bars, params) == [True, False, False, False, True]


def test_vwap_deviation_does_not_peek_forward() -> None:
    params = vwap_deviation.normalize_params(
        {"entry_deviation_bps": 25.0, "exit_deviation_bps": 0.0, "max_hold_bars": 10}
    )
    base_bars = (_bar(0, 99.70), _bar(1, 99.80), _bar(2, 100.00))
    altered_bars = (base_bars[0], replace(base_bars[1], close=120.0), replace(base_bars[2], close=80.0))

    assert vwap_deviation.generate_regime((), base_bars, params)[0] == vwap_deviation.generate_regime(
        (),
        altered_bars,
        params,
    )[0]


def test_registry_validates_vwap_deviation_specs() -> None:
    spec = REGISTRY.validate_spec(
        StrategySpec(
            name="vwap_test",
            signal=SignalSpec(
                "vwap_deviation",
                {"entry_deviation_bps": 25.0, "exit_deviation_bps": 0.0, "max_hold_bars": 30},
            ),
        )
    )

    assert spec.signal.name == "vwap_deviation"
    assert spec.signal.params["min_bars_between_entries"] == 30
    assert spec.signal.params["max_entries_per_session"] == 4
    assert REGISTRY.required_history(spec) == 0


def test_vwap_deviation_neighbors_preserve_churn_controls() -> None:
    spec = REGISTRY.validate_spec(
        StrategySpec(
            name="vwap_test",
            signal=SignalSpec(
                "vwap_deviation",
                {
                    "entry_deviation_bps": 25.0,
                    "exit_deviation_bps": 0.0,
                    "max_hold_bars": 30,
                    "min_bars_between_entries": 12,
                    "max_entries_per_session": 3,
                },
            ),
        )
    )

    neighbors = REGISTRY.neighbors(spec)

    assert neighbors
    assert {item.signal.params["min_bars_between_entries"] for item in neighbors} == {12}
    assert {item.signal.params["max_entries_per_session"] for item in neighbors} == {3}


@pytest.mark.parametrize(
    ("params", "message"),
    (
        ({"entry_deviation_bps": 0.0}, "entry_deviation_bps"),
        ({"exit_deviation_bps": -1.0}, "exit_deviation_bps"),
        ({"entry_deviation_bps": 10.0, "exit_deviation_bps": 10.0}, "exit_deviation_bps"),
        ({"max_hold_bars": 0}, "max_hold_bars"),
        ({"min_bars_between_entries": -1}, "min_bars_between_entries"),
        ({"max_entries_per_session": 0}, "max_entries_per_session"),
        ({"entry_deviation_bps": float("nan")}, "entry_deviation_bps must be finite"),
    ),
)
def test_registry_rejects_invalid_vwap_deviation_specs(params: dict[str, float], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        REGISTRY.validate_spec(StrategySpec(name="bad_vwap", signal=SignalSpec("vwap_deviation", params)))


def test_planner_emits_vwap_deviation_when_allowed() -> None:
    planned = DeterministicPlanner(REGISTRY).plan(batch_size=5, allowed_signal_families=("vwap_deviation",))

    assert len(planned) == 5
    assert {item.spec.signal.name for item in planned} == {"vwap_deviation"}
    assert all(REGISTRY.validate_spec(item.spec) for item in planned)
    assert all("min_bars_between_entries" in item.spec.signal.params for item in planned)
    assert all("max_entries_per_session" in item.spec.signal.params for item in planned)


def test_vwap_deviation_grid_count_does_not_expand_for_churn_controls() -> None:
    assert len(vwap_deviation.parameter_grid()) == 33
