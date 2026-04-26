from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from trader.data.models import MarketBar
from trader.features.primitives import ema, rolling_max_exclusive, rolling_min_exclusive, rsi
from trader.strategies.filters.regime import _intraday_realized_volatility_bps, _session_progress_stats


NEW_YORK = ZoneInfo("America/New_York")


def _reference_ema(values: list[float], length: int) -> list[float]:
    alpha = 2.0 / (length + 1)
    result = []
    current = None
    for value in values:
        current = value if current is None else (value * alpha) + (current * (1.0 - alpha))
        result.append(current)
    return result


def _reference_rsi(values: list[float], length: int) -> list[float | None]:
    if len(values) <= length:
        return [None] * len(values)

    def to_rsi(avg_gain: float, avg_loss: float) -> float:
        if avg_loss == 0.0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    result: list[float | None] = [None] * length
    avg_gain = sum(max(values[i] - values[i - 1], 0.0) for i in range(1, length + 1)) / length
    avg_loss = sum(max(values[i - 1] - values[i], 0.0) for i in range(1, length + 1)) / length
    result.append(to_rsi(avg_gain, avg_loss))
    for i in range(length + 1, len(values)):
        gain = max(values[i] - values[i - 1], 0.0)
        loss = max(values[i - 1] - values[i], 0.0)
        avg_gain = (avg_gain * (length - 1) + gain) / length
        avg_loss = (avg_loss * (length - 1) + loss) / length
        result.append(to_rsi(avg_gain, avg_loss))
    return result


def _bar(day: int, minute: int, close: float, *, open_: float | None = None) -> MarketBar:
    timestamp = (
        datetime(2026, 1, 5, 9, 30, tzinfo=NEW_YORK)
        + timedelta(days=day, minutes=minute)
    ).astimezone(timezone.utc)
    base_open = close if open_ is None else open_
    return MarketBar(
        timestamp_ms=int(timestamp.timestamp() * 1000),
        timestamp_utc=timestamp.isoformat(),
        open=base_open,
        high=max(base_open, close) + 0.35,
        low=min(base_open, close) - 0.20,
        close=close,
        volume=1_000.0 + minute,
        vwap=close,
    )


def test_vectorized_feature_primitives_match_reference_outputs() -> None:
    values = [100.0, 101.5, 99.25, 102.0, 102.5, 101.0, 103.75, 104.0, 102.25]

    assert ema(values, 4) == pytest.approx(_reference_ema(values, 4))
    assert rsi(values, 3) == pytest.approx(_reference_rsi(values, 3))
    assert rolling_max_exclusive(values, 3) == [
        None,
        None,
        None,
        101.5,
        102.0,
        102.5,
        102.5,
        103.75,
        104.0,
    ]
    assert rolling_min_exclusive(values, 3) == [
        None,
        None,
        None,
        99.25,
        99.25,
        99.25,
        101.0,
        101.0,
        101.0,
    ]


def test_vectorized_intraday_realized_volatility_resets_at_session_boundaries() -> None:
    bars = (
        _bar(0, 0, 100.0),
        _bar(0, 1, 101.0),
        _bar(0, 2, 99.0),
        _bar(1, 0, 100.0),
        _bar(1, 1, 102.0),
        _bar(1, 2, 103.0),
    )

    realized = _intraday_realized_volatility_bps(bars, 2)

    first_session = math.sqrt(
        (((101.0 / 100.0) - 1.0) * 10_000.0) ** 2
        + (((99.0 / 101.0) - 1.0) * 10_000.0) ** 2
    ) / math.sqrt(2)
    second_session = math.sqrt(
        (((102.0 / 100.0) - 1.0) * 10_000.0) ** 2
        + (((103.0 / 102.0) - 1.0) * 10_000.0) ** 2
    ) / math.sqrt(2)
    assert realized == pytest.approx([None, None, first_session, None, None, second_session])


def test_vectorized_session_progress_stats_match_session_local_state() -> None:
    bars = (
        _bar(0, 0, 100.0, open_=100.0),
        _bar(0, 1, 101.0, open_=100.5),
        _bar(0, 2, 99.0, open_=100.0),
        _bar(1, 0, 50.0, open_=50.0),
        _bar(1, 1, 52.0, open_=51.0),
    )

    stats = _session_progress_stats(bars)

    assert [item[0] for item in stats] == [1, 2, 3, 1, 2]
    assert stats[0][1:] == pytest.approx((0.0, 55.0, 0.0))
    assert stats[2][1:] == pytest.approx((100.0, 255.0, 1.0 / 2.55))
    assert stats[3][1:] == pytest.approx((0.0, 110.0, 0.0))
    assert stats[4][1:] == pytest.approx((400.0, 510.0, 2.0 / 2.55))
