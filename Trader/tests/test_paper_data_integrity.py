from __future__ import annotations

from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from typing import Sequence
from zoneinfo import ZoneInfo

from pytest import raises

from trader.data.models import MarketBar
from trader.paper.gateway import PaperTradingGateway
from trader.paper.live_data import (
    EarlyClose,
    LiveDataIntegrityChecker,
    LiveDataIntegrityConfig,
    LiveMarketSnapshot,
    MarketCalendar,
)
from trader.paper.models import BrokerPosition, Fill, OrderAck, OrderRequest
from trader.paper.store import PaperTradingStore


NEW_YORK = ZoneInfo("America/New_York")


class FakeBroker:
    def __init__(self) -> None:
        self.submitted: list[OrderRequest] = []

    def submit_order(self, request: OrderRequest) -> OrderAck:
        self.submitted.append(request)
        return OrderAck(
            client_order_id=request.client_order_id,
            broker_order_id=f"broker-{len(self.submitted)}",
            status="accepted",
            submitted_at_utc="2026-04-25T14:00:00+00:00",
        )

    def cancel_order(self, client_order_id: str) -> OrderAck:
        return OrderAck(client_order_id, "broker-cancel", "canceled", "2026-04-25T14:00:00+00:00")

    def list_open_orders(self) -> Sequence[OrderAck]:
        return tuple()

    def list_positions(self) -> Sequence[BrokerPosition]:
        return tuple()

    def list_fills(self, *, since_utc: str | None = None) -> Sequence[Fill]:
        return tuple()


def _bar(local_timestamp: datetime, price: float = 100.0) -> MarketBar:
    timestamp = local_timestamp.astimezone(timezone.utc)
    return MarketBar(
        timestamp_ms=int(timestamp.timestamp() * 1000),
        timestamp_utc=timestamp.isoformat(),
        open=price,
        high=price + 0.25,
        low=price - 0.25,
        close=price,
        volume=1_000.0,
    )


def _snapshot(
    *,
    quote_utc: str = "2026-04-27T14:01:20+00:00",
    latest_bar: MarketBar | None = None,
    previous_bar: MarketBar | None = None,
) -> LiveMarketSnapshot:
    return LiveMarketSnapshot(
        symbol="SPY",
        quote_timestamp_utc=quote_utc,
        latest_bar=latest_bar or _bar(datetime(2026, 4, 27, 10, 1, tzinfo=NEW_YORK)),
        previous_bar=previous_bar,
    )


def _gateway(
    tmp_path: Path,
    checker: LiveDataIntegrityChecker,
) -> tuple[PaperTradingGateway, FakeBroker, PaperTradingStore]:
    store = PaperTradingStore(tmp_path / "paper.db")
    store.initialize()
    broker = FakeBroker()
    return PaperTradingGateway(broker, store, data_integrity_checker=checker), broker, store


def _request(client_order_id: str = "client-1") -> OrderRequest:
    return OrderRequest(
        symbol="SPY",
        side="buy",
        quantity=1,
        strategy_id="strategy_1",
        client_order_id=client_order_id,
    )


def test_live_data_integrity_blocks_stale_quote() -> None:
    checker = LiveDataIntegrityChecker(LiveDataIntegrityConfig(max_quote_age_seconds=15.0))

    decision = checker.assess(
        _snapshot(quote_utc="2026-04-27T14:01:00+00:00"),
        symbol="SPY",
        now_utc="2026-04-27T14:01:30+00:00",
    )

    assert not decision.allowed
    assert decision.reason == "stale_quote"


def test_live_data_integrity_blocks_future_quote() -> None:
    checker = LiveDataIntegrityChecker()

    decision = checker.assess(
        _snapshot(quote_utc="2026-04-27T14:02:00+00:00"),
        symbol="SPY",
        now_utc="2026-04-27T14:01:30+00:00",
    )

    assert not decision.allowed
    assert decision.reason == "future_quote"


def test_live_data_integrity_blocks_stale_bar() -> None:
    checker = LiveDataIntegrityChecker(LiveDataIntegrityConfig(max_bar_age_seconds=120.0))

    decision = checker.assess(
        _snapshot(
            quote_utc="2026-04-27T14:00:00+00:00",
            latest_bar=_bar(datetime(2026, 4, 27, 9, 57, tzinfo=NEW_YORK)),
        ),
        symbol="SPY",
        now_utc="2026-04-27T14:00:00+00:00",
    )

    assert not decision.allowed
    assert decision.reason == "stale_bar"


def test_live_data_integrity_blocks_future_bar() -> None:
    checker = LiveDataIntegrityChecker()

    decision = checker.assess(
        _snapshot(
            quote_utc="2026-04-27T14:01:30+00:00",
            latest_bar=_bar(datetime(2026, 4, 27, 10, 2, tzinfo=NEW_YORK)),
        ),
        symbol="SPY",
        now_utc="2026-04-27T14:01:30+00:00",
    )

    assert not decision.allowed
    assert decision.reason == "future_bar"


def test_live_data_integrity_blocks_latest_bar_outside_market_hours() -> None:
    checker = LiveDataIntegrityChecker()

    decision = checker.assess(
        _snapshot(
            quote_utc="2026-04-27T13:30:30+00:00",
            latest_bar=_bar(datetime(2026, 4, 27, 9, 29, tzinfo=NEW_YORK)),
        ),
        symbol="SPY",
        now_utc="2026-04-27T13:30:30+00:00",
    )

    assert not decision.allowed
    assert decision.reason == "bar_outside_market_hours"


def test_live_data_integrity_blocks_missing_minute() -> None:
    checker = LiveDataIntegrityChecker()

    decision = checker.assess(
        _snapshot(
            quote_utc="2026-04-27T13:32:30+00:00",
            previous_bar=_bar(datetime(2026, 4, 27, 9, 30, tzinfo=NEW_YORK)),
            latest_bar=_bar(datetime(2026, 4, 27, 9, 32, tzinfo=NEW_YORK)),
        ),
        symbol="SPY",
        now_utc="2026-04-27T13:32:30+00:00",
    )

    assert not decision.allowed
    assert decision.reason == "missing_minute"


def test_market_calendar_blocks_closed_and_early_close_times() -> None:
    calendar = MarketCalendar(
        closed_dates=("2026-04-27",),
        early_closes=(EarlyClose("2026-07-03", time(13, 0)),),
    )
    checker = LiveDataIntegrityChecker(LiveDataIntegrityConfig(calendar=calendar))

    assert not checker.assess(
        _snapshot(),
        symbol="SPY",
        now_utc="2026-04-27T14:00:00+00:00",
    ).allowed
    assert not checker.assess(
        _snapshot(
            quote_utc="2026-07-03T17:01:00+00:00",
            latest_bar=_bar(datetime(2026, 7, 3, 12, 59, tzinfo=NEW_YORK)),
        ),
        symbol="SPY",
        now_utc="2026-07-03T17:01:00+00:00",
    ).allowed
    assert checker.assess(
        _snapshot(
            quote_utc="2026-07-03T16:59:30+00:00",
            latest_bar=_bar(datetime(2026, 7, 3, 12, 59, tzinfo=NEW_YORK)),
        ),
        symbol="SPY",
        now_utc="2026-07-03T16:59:30+00:00",
    ).allowed


def test_gateway_blocks_order_and_audits_data_integrity_failure(tmp_path: Path) -> None:
    gateway, broker, store = _gateway(tmp_path, LiveDataIntegrityChecker())

    with raises(ValueError, match="data integrity check failed: missing_market_data"):
        gateway.submit_order(
            _request(),
            signal_id="signal-1",
            now_utc="2026-04-27T14:00:00+00:00",
        )

    assert broker.submitted == []
    events = store.audit_events()
    assert events[0].event_type == "data_integrity_blocked"
    assert events[0].payload["reason"] == "missing_market_data"


def test_gateway_idempotent_replay_bypasses_later_data_integrity_failure(tmp_path: Path) -> None:
    gateway, broker, _ = _gateway(tmp_path, LiveDataIntegrityChecker())

    first = gateway.submit_order(
        _request(),
        signal_id="signal-1",
        now_utc="2026-04-27T14:01:30+00:00",
        market_data=_snapshot(),
    )
    second = gateway.submit_order(
        _request(),
        signal_id="signal-1",
        now_utc="2026-04-27T14:02:30+00:00",
    )

    assert first == second
    assert len(broker.submitted) == 1


def test_live_data_integrity_config_rejects_invalid_values() -> None:
    with raises(ValueError, match="max_quote_age_seconds"):
        LiveDataIntegrityConfig(max_quote_age_seconds=0.0)
    with raises(ValueError, match="max_bar_age_seconds"):
        LiveDataIntegrityConfig(max_bar_age_seconds=float("nan"))
    with raises(ValueError, match="expected_bar_seconds"):
        LiveDataIntegrityConfig(expected_bar_seconds=float("inf"))
    with raises(ValueError, match="early close_time"):
        EarlyClose("2026-07-03", time(9, 30))
