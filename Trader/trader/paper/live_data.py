from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time, timezone
from math import isfinite
from zoneinfo import ZoneInfo

from trader.data.models import MarketBar, REGULAR_SESSION_END, REGULAR_SESSION_START


NEW_YORK = ZoneInfo("America/New_York")


@dataclass(frozen=True)
class EarlyClose:
    trading_day: str
    close_time: time
    reason: str = "early_close"

    def __post_init__(self) -> None:
        date.fromisoformat(self.trading_day)
        if not REGULAR_SESSION_START < self.close_time <= REGULAR_SESSION_END:
            raise ValueError("early close_time must be inside regular session hours")


@dataclass(frozen=True)
class MarketCalendar:
    closed_dates: tuple[str, ...] = ()
    early_closes: tuple[EarlyClose, ...] = ()

    def __post_init__(self) -> None:
        for closed_date in self.closed_dates:
            date.fromisoformat(closed_date)

    def is_open_at(self, timestamp_utc: str) -> bool:
        timestamp = _parse_aware_utc(timestamp_utc, "timestamp_utc")
        local = timestamp.astimezone(NEW_YORK)
        bounds = self.session_bounds_utc(local.date().isoformat())
        if bounds is None:
            return False
        session_start, session_end = bounds
        return session_start <= timestamp < session_end

    def session_bounds_utc(self, trading_day: str) -> tuple[datetime, datetime] | None:
        local_date = date.fromisoformat(trading_day)
        if local_date.weekday() >= 5 or trading_day in set(self.closed_dates):
            return None
        session_start = datetime.combine(local_date, REGULAR_SESSION_START, tzinfo=NEW_YORK)
        session_end = datetime.combine(local_date, self._close_time(trading_day), tzinfo=NEW_YORK)
        return session_start.astimezone(timezone.utc), session_end.astimezone(timezone.utc)

    def _close_time(self, trading_day: str) -> time:
        for early_close in self.early_closes:
            if early_close.trading_day == trading_day:
                return early_close.close_time
        return REGULAR_SESSION_END


@dataclass(frozen=True)
class LiveMarketSnapshot:
    symbol: str
    quote_timestamp_utc: str | None
    latest_bar: MarketBar | None
    previous_bar: MarketBar | None = None


@dataclass(frozen=True)
class LiveDataIntegrityConfig:
    max_quote_age_seconds: float = 15.0
    max_bar_age_seconds: float = 120.0
    expected_bar_seconds: float = 60.0
    calendar: MarketCalendar = field(default_factory=MarketCalendar)

    def __post_init__(self) -> None:
        if not _is_positive_finite(self.max_quote_age_seconds):
            raise ValueError("max_quote_age_seconds must be > 0")
        if not _is_positive_finite(self.max_bar_age_seconds):
            raise ValueError("max_bar_age_seconds must be > 0")
        if not _is_positive_finite(self.expected_bar_seconds):
            raise ValueError("expected_bar_seconds must be > 0")


@dataclass(frozen=True)
class LiveDataIntegrityDecision:
    allowed: bool
    reason: str = ""


class LiveDataIntegrityChecker:
    def __init__(self, config: LiveDataIntegrityConfig | None = None) -> None:
        self.config = config or LiveDataIntegrityConfig()

    def assess(
        self,
        snapshot: LiveMarketSnapshot | None,
        *,
        symbol: str,
        now_utc: str,
    ) -> LiveDataIntegrityDecision:
        now = _parse_aware_utc(now_utc, "now_utc")
        if not self.config.calendar.is_open_at(now.isoformat()):
            return LiveDataIntegrityDecision(False, "market_closed")
        if snapshot is None:
            return LiveDataIntegrityDecision(False, "missing_market_data")
        if snapshot.symbol.upper() != symbol.upper():
            return LiveDataIntegrityDecision(False, "symbol_mismatch")
        if snapshot.quote_timestamp_utc is None:
            return LiveDataIntegrityDecision(False, "missing_quote")
        quote_time = _parse_aware_utc(snapshot.quote_timestamp_utc, "quote_timestamp_utc")
        if quote_time > now:
            return LiveDataIntegrityDecision(False, "future_quote")
        if _age_seconds(now, quote_time) > self.config.max_quote_age_seconds:
            return LiveDataIntegrityDecision(False, "stale_quote")
        if snapshot.latest_bar is None:
            return LiveDataIntegrityDecision(False, "missing_bar")
        latest_bar_time = snapshot.latest_bar.dt_utc
        if latest_bar_time > now:
            return LiveDataIntegrityDecision(False, "future_bar")
        if not self.config.calendar.is_open_at(snapshot.latest_bar.timestamp_utc):
            return LiveDataIntegrityDecision(False, "bar_outside_market_hours")
        if _age_seconds(now, latest_bar_time) > self.config.max_bar_age_seconds:
            return LiveDataIntegrityDecision(False, "stale_bar")
        if snapshot.previous_bar is not None:
            gap_seconds = (snapshot.latest_bar.dt_utc - snapshot.previous_bar.dt_utc).total_seconds()
            if gap_seconds <= 0:
                return LiveDataIntegrityDecision(False, "out_of_order_bars")
            if _same_session(snapshot.previous_bar, snapshot.latest_bar) and gap_seconds > self.config.expected_bar_seconds:
                return LiveDataIntegrityDecision(False, "missing_minute")
        return LiveDataIntegrityDecision(True)


def _same_session(previous_bar: MarketBar, latest_bar: MarketBar) -> bool:
    return previous_bar.session_date == latest_bar.session_date


def _age_seconds(now: datetime, timestamp: datetime) -> float:
    return (now - timestamp).total_seconds()


def _is_positive_finite(value: float) -> bool:
    return isfinite(value) and value > 0


def _parse_aware_utc(value: str, field_name: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ValueError(f"{field_name} must be timezone-aware UTC")
    if parsed.utcoffset().total_seconds() != 0:
        raise ValueError(f"{field_name} must be UTC")
    return parsed
