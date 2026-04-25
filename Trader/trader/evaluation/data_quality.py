from __future__ import annotations

import math
from collections import defaultdict
from datetime import time
from typing import Any, Mapping, Sequence

from trader.data.models import REGULAR_SESSION_START, MarketBar

EXPECTED_REGULAR_SESSION_BARS = 390
EXPECTED_MINUTE_MS = 60_000
EXPECTED_REGULAR_SESSION_LAST_BAR = time(15, 59)


def validate_bars(bars: Sequence[MarketBar]) -> tuple[str, ...]:
    warnings: list[str] = []
    seen_timestamps: set[int] = set()
    sessions: defaultdict[str, list[MarketBar]] = defaultdict(list)
    previous: MarketBar | None = None

    for bar in bars:
        if bar.timestamp_ms in seen_timestamps:
            warnings.append(f"data_quality.duplicate_timestamp: duplicate bar at {bar.timestamp_utc}")
        seen_timestamps.add(bar.timestamp_ms)

        if previous is not None and previous.session_date == bar.session_date:
            gap_ms = bar.timestamp_ms - previous.timestamp_ms
            if gap_ms > EXPECTED_MINUTE_MS:
                minutes = gap_ms // EXPECTED_MINUTE_MS
                warnings.append(
                    f"data_quality.missing_bars: gap from {previous.timestamp_utc} to {bar.timestamp_utc} is {minutes} minutes"
                )
        previous = bar

        if not _valid_ohlc(bar):
            warnings.append(f"data_quality.ohlc_sanity: {bar.timestamp_utc} violates OHLC ordering")
        if not _valid_volume(bar):
            warnings.append(f"data_quality.volume_anomaly: {bar.timestamp_utc} has invalid volume")
        sessions[bar.session_date].append(bar)

    for session_date, session_bars in sorted(sessions.items()):
        if _appears_full_regular_session(session_bars) and len(session_bars) != EXPECTED_REGULAR_SESSION_BARS:
            warnings.append(
                f"data_quality.unexpected_session_length: session {session_date} has {len(session_bars)} regular-session bars; expected {EXPECTED_REGULAR_SESSION_BARS}"
            )

    return tuple(warnings)


def validate_raw_bar_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    regular_session_only: bool,
) -> tuple[str, ...]:
    warnings: list[str] = []
    bars: list[MarketBar] = []
    for row in rows:
        timestamp_ms = int(row["timestamp_ms"])
        timestamp_utc = str(row["timestamp_utc"])
        shell_bar = MarketBar(
            timestamp_ms=timestamp_ms,
            timestamp_utc=timestamp_utc,
            open=0.0,
            high=0.0,
            low=0.0,
            close=0.0,
            volume=0.0,
        )
        if regular_session_only and not shell_bar.is_regular_session:
            continue
        null_ohlc = [name for name in ("open", "high", "low", "close") if row[name] is None]
        if null_ohlc:
            warnings.append(f"data_quality.null_ohlc: {timestamp_utc} has null {','.join(null_ohlc)}")
        else:
            if row["volume"] is None:
                warnings.append(f"data_quality.volume_anomaly: {timestamp_utc} has null volume")
            bars.append(
                MarketBar(
                    timestamp_ms=timestamp_ms,
                    timestamp_utc=timestamp_utc,
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row["volume"] or 0.0),
                )
            )
    return tuple(warnings) + validate_bars(tuple(bars))


def _valid_ohlc(bar: MarketBar) -> bool:
    values = (bar.open, bar.high, bar.low, bar.close)
    if any(value is None or not math.isfinite(float(value)) for value in values):
        return False
    return bar.high >= max(bar.open, bar.close) and bar.low <= min(bar.open, bar.close) and bar.high >= bar.low


def _valid_volume(bar: MarketBar) -> bool:
    return bar.volume is not None and math.isfinite(float(bar.volume)) and bar.volume >= 0.0


def _appears_full_regular_session(session_bars: Sequence[MarketBar]) -> bool:
    if not session_bars:
        return False
    start = session_bars[0].dt_local.timetz().replace(tzinfo=None)
    end = session_bars[-1].dt_local.timetz().replace(tzinfo=None)
    return start == REGULAR_SESSION_START and end == EXPECTED_REGULAR_SESSION_LAST_BAR
