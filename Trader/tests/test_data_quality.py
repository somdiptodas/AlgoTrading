from __future__ import annotations

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from trader.data.models import AggregateBar, MarketBar
from trader.data.storage import SQLiteBarStore
from trader.data.view import DataView
from trader.evaluation.data_quality import validate_bars
from trader.evaluation.runner import EvaluationRunner
from trader.evaluation.splits import Fold
from trader.strategies.registry import REGISTRY
from trader.strategies.spec import SignalSpec, StrategySpec


NEW_YORK = ZoneInfo("America/New_York")


def _bar(timestamp: datetime, price: float = 100.0, *, volume: float = 1_000.0) -> MarketBar:
    timestamp_utc = timestamp.astimezone(timezone.utc)
    return MarketBar(
        timestamp_ms=int(timestamp_utc.timestamp() * 1000),
        timestamp_utc=timestamp_utc.isoformat(),
        open=price,
        high=price + 0.25,
        low=price - 0.25,
        close=price,
        volume=volume,
    )


def _regular_session_with_missing_bar() -> tuple[MarketBar, ...]:
    start = datetime(2026, 1, 5, 9, 30, tzinfo=NEW_YORK)
    bars = []
    for index in range(390):
        if index == 10:
            continue
        bars.append(_bar(start + timedelta(minutes=index), 100.0 + index))
    return tuple(bars)


def test_validate_bars_detects_core_data_quality_warnings() -> None:
    session_warnings = validate_bars(_regular_session_with_missing_bar())
    assert any(item.startswith("data_quality.missing_bars:") for item in session_warnings)
    assert (
        "data_quality.unexpected_session_length: session 2026-01-05 has 389 regular-session bars; expected 390"
        in session_warnings
    )

    timestamp = datetime(2026, 1, 6, 9, 30, tzinfo=NEW_YORK)
    bad_bars = (
        _bar(timestamp),
        _bar(timestamp),
        MarketBar(
            timestamp_ms=int((timestamp + timedelta(minutes=1)).astimezone(timezone.utc).timestamp() * 1000),
            timestamp_utc=(timestamp + timedelta(minutes=1)).astimezone(timezone.utc).isoformat(),
            open=100.0,
            high=99.0,
            low=101.0,
            close=100.0,
            volume=-1.0,
        ),
    )
    warnings = validate_bars(bad_bars)

    assert any(item.startswith("data_quality.duplicate_timestamp:") for item in warnings)
    assert any(item.startswith("data_quality.ohlc_sanity:") for item in warnings)
    assert any(item.startswith("data_quality.volume_anomaly:") for item in warnings)


def test_data_view_quality_warnings_detect_raw_null_ohlc(tmp_path) -> None:
    timestamp = datetime(2026, 1, 5, 9, 30, tzinfo=NEW_YORK).astimezone(timezone.utc)
    store = SQLiteBarStore(tmp_path / "market.db")
    store.upsert_bars(
        [
            AggregateBar(
                ticker="SPY",
                multiplier=1,
                timespan="minute",
                timestamp_ms=int(timestamp.timestamp() * 1000),
                open=None,
                high=101.0,
                low=99.0,
                close=100.0,
                volume=1_000.0,
                vwap=None,
                transactions=None,
            ),
            AggregateBar(
                ticker="SPY",
                multiplier=1,
                timespan="minute",
                timestamp_ms=int((timestamp + timedelta(minutes=1)).timestamp() * 1000),
                open=100.0,
                high=101.0,
                low=99.0,
                close=100.0,
                volume=None,
                vwap=None,
                transactions=None,
            ),
        ]
    )
    view = DataView(tmp_path / "market.db")

    assert len(view.bars("SPY", 1, "minute", regular_session_only=True)) == 1
    warnings = view.quality_warnings("SPY", 1, "minute", regular_session_only=True)
    assert warnings == (
        f"data_quality.null_ohlc: {timestamp.isoformat()} has null open",
        f"data_quality.volume_anomaly: {(timestamp + timedelta(minutes=1)).isoformat()} has null volume",
    )


def test_runner_quality_warnings_include_raw_null_boundary_rows(tmp_path) -> None:
    start = datetime(2026, 1, 5, 9, 30, tzinfo=NEW_YORK).astimezone(timezone.utc)
    store = SQLiteBarStore(tmp_path / "market.db")
    rows = []
    for index in range(390):
        timestamp = start + timedelta(minutes=index)
        rows.append(
            AggregateBar(
                ticker="SPY",
                multiplier=1,
                timespan="minute",
                timestamp_ms=int(timestamp.timestamp() * 1000),
                open=None if index == 0 else 100.0 + index,
                high=101.0 + index,
                low=99.0 + index,
                close=100.0 + index,
                volume=1_000.0,
                vwap=None,
                transactions=None,
            )
        )
    store.upsert_bars(rows)
    view = DataView(tmp_path / "market.db")
    runner = EvaluationRunner.__new__(EvaluationRunner)
    runner.data_view = view
    bars = view.bars("SPY", 1, "minute", regular_session_only=True)
    spec = REGISTRY.validate_spec(
        StrategySpec(
            name="ema_quality",
            signal=SignalSpec("ema_cross", {"fast_length": 2, "slow_length": 3, "signal_buffer_bps": 0.0}),
        )
    )

    warnings = runner._quality_warnings(spec, bars)

    assert f"data_quality.null_ohlc: {start.isoformat()} has null open" in warnings


def test_runner_quality_warnings_do_not_expand_mid_session_folds(tmp_path) -> None:
    start = datetime(2026, 1, 5, 9, 30, tzinfo=NEW_YORK).astimezone(timezone.utc)
    store = SQLiteBarStore(tmp_path / "market.db")
    rows = []
    for index in range(390):
        timestamp = start + timedelta(minutes=index)
        rows.append(
            AggregateBar(
                ticker="SPY",
                multiplier=1,
                timespan="minute",
                timestamp_ms=int(timestamp.timestamp() * 1000),
                open=None if index == 0 else 100.0 + index,
                high=101.0 + index,
                low=99.0 + index,
                close=100.0 + index,
                volume=1_000.0,
                vwap=None,
                transactions=None,
            )
        )
    store.upsert_bars(rows)
    view = DataView(tmp_path / "market.db")
    runner = EvaluationRunner.__new__(EvaluationRunner)
    runner.data_view = view
    bars = tuple(bar for bar in view.bars("SPY", 1, "minute", regular_session_only=True) if bar.dt_local.hour >= 12)
    spec = REGISTRY.validate_spec(
        StrategySpec(
            name="ema_quality",
            signal=SignalSpec("ema_cross", {"fast_length": 2, "slow_length": 3, "signal_buffer_bps": 0.0}),
        )
    )

    warnings = runner._quality_warnings(spec, bars)

    assert f"data_quality.null_ohlc: {start.isoformat()} has null open" not in warnings


def test_evaluate_fold_propagates_data_quality_warnings() -> None:
    runner = EvaluationRunner.__new__(EvaluationRunner)
    runner.registry = REGISTRY
    runner._fold_result_cache = {}
    bars = tuple(_bar(datetime(2026, 1, 5, 9, 30, tzinfo=NEW_YORK) + timedelta(minutes=index)) for index in range(20))
    bars = bars[:15] + (_bar(datetime(2026, 1, 5, 9, 46, tzinfo=NEW_YORK)),) + bars[16:]
    fold = Fold(
        fold_id="fold_quality",
        train_start_idx=0,
        train_end_idx=9,
        test_start_idx=10,
        test_end_idx=19,
        embargo_bars=0,
        train_start_utc=bars[0].timestamp_utc,
        train_end_utc=bars[9].timestamp_utc,
        test_start_utc=bars[10].timestamp_utc,
        test_end_utc=bars[19].timestamp_utc,
    )
    spec = REGISTRY.validate_spec(
        StrategySpec(
            name="ema_quality",
            signal=SignalSpec("ema_cross", {"fast_length": 2, "slow_length": 3, "signal_buffer_bps": 0.0}),
        )
    )

    result = runner._evaluate_fold(spec, fold, bars)

    assert any(item.startswith("data_quality.missing_bars:") for item in result.warnings)
