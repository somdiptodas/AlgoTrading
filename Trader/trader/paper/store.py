from __future__ import annotations

import sqlite3
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from trader.ledger.entry import json_dumps, json_loads
from trader.paper.audit import AuditEvent
from trader.paper.models import OrderAck, OrderRequest
from trader.paper.reconcile import PositionMismatch, ReconciliationReport


NEW_YORK = ZoneInfo("America/New_York")


SCHEMA = """
CREATE TABLE IF NOT EXISTS paper_orders (
    client_order_id TEXT PRIMARY KEY,
    broker_order_id TEXT NOT NULL,
    status TEXT NOT NULL,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    quantity INTEGER NOT NULL,
    strategy_id TEXT NOT NULL,
    request_json TEXT NOT NULL,
    ack_json TEXT NOT NULL,
    submitted_at_utc TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS paper_audit_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id TEXT NOT NULL UNIQUE,
    event_type TEXT NOT NULL,
    client_order_id TEXT,
    broker_order_id TEXT,
    payload_json TEXT NOT NULL,
    logged_at_utc TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS paper_reconciliation_reports (
    report_id TEXT PRIMARY KEY,
    generated_at_utc TEXT NOT NULL,
    mismatch_count INTEGER NOT NULL,
    report_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS paper_daily_risk (
    trading_day TEXT PRIMARY KEY,
    realized_pnl_cash REAL NOT NULL,
    updated_at_utc TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS paper_kill_switch (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    enabled INTEGER NOT NULL,
    reason TEXT NOT NULL,
    updated_at_utc TEXT NOT NULL
);
"""


class PaperTradingStore:
    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)

    def initialize(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(SCHEMA)

    def get_order_ack(self, client_order_id: str) -> OrderAck | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT ack_json FROM paper_orders WHERE client_order_id = ?",
                (client_order_id,),
            ).fetchone()
        if row is None:
            return None
        return _order_ack_from_payload(dict(json_loads(row["ack_json"], default={})))

    def accepted_order_count_for_trading_day(self, trading_day: str) -> int:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT submitted_at_utc
                FROM paper_orders
                WHERE status IN ('submitted', 'accepted', 'filled')
                """,
            ).fetchall()
        return sum(1 for row in rows if _trading_day(str(row["submitted_at_utc"])) == trading_day)

    def set_daily_realized_pnl(self, trading_day: str, pnl_cash: float, *, updated_at_utc: str) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO paper_daily_risk (trading_day, realized_pnl_cash, updated_at_utc)
                VALUES (?, ?, ?)
                ON CONFLICT(trading_day) DO UPDATE SET
                    realized_pnl_cash = excluded.realized_pnl_cash,
                    updated_at_utc = excluded.updated_at_utc
                """,
                (trading_day, pnl_cash, updated_at_utc),
            )

    def get_daily_realized_pnl(self, trading_day: str) -> float:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT realized_pnl_cash FROM paper_daily_risk WHERE trading_day = ?",
                (trading_day,),
            ).fetchone()
        return 0.0 if row is None else float(row["realized_pnl_cash"])

    def set_kill_switch(self, enabled: bool, *, reason: str, updated_at_utc: str) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO paper_kill_switch (id, enabled, reason, updated_at_utc)
                VALUES (1, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    enabled = excluded.enabled,
                    reason = excluded.reason,
                    updated_at_utc = excluded.updated_at_utc
                """,
                (1 if enabled else 0, reason, updated_at_utc),
            )

    def get_kill_switch(self) -> tuple[bool, str]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT enabled, reason FROM paper_kill_switch WHERE id = 1",
            ).fetchone()
        if row is None:
            return False, ""
        return bool(row["enabled"]), str(row["reason"])

    def get_order_request(self, client_order_id: str) -> OrderRequest | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT request_json FROM paper_orders WHERE client_order_id = ?",
                (client_order_id,),
            ).fetchone()
        if row is None:
            return None
        return _order_request_from_payload(dict(json_loads(row["request_json"], default={})))

    def record_order(self, request: OrderRequest, ack: OrderAck) -> None:
        if ack.client_order_id != request.client_order_id:
            raise ValueError("broker ack client_order_id must match request client_order_id")
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO paper_orders (
                    client_order_id,
                    broker_order_id,
                    status,
                    symbol,
                    side,
                    quantity,
                    strategy_id,
                    request_json,
                    ack_json,
                    submitted_at_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ack.client_order_id,
                    ack.broker_order_id,
                    ack.status,
                    request.symbol.upper(),
                    request.side,
                    request.quantity,
                    request.strategy_id,
                    json_dumps(asdict(request)),
                    json_dumps(asdict(ack)),
                    ack.submitted_at_utc,
                ),
            )

    def append_audit_event(
        self,
        *,
        event_id: str,
        event: AuditEvent,
        client_order_id: str | None = None,
        broker_order_id: str | None = None,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO paper_audit_events (
                    event_id,
                    event_type,
                    client_order_id,
                    broker_order_id,
                    payload_json,
                    logged_at_utc
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    event.event_type,
                    client_order_id,
                    broker_order_id,
                    json_dumps(event.payload),
                    event.occurred_at_utc,
                ),
            )

    def audit_events(self) -> tuple[AuditEvent, ...]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT event_type, payload_json, logged_at_utc
                FROM paper_audit_events
                ORDER BY id
                """
            ).fetchall()
        return tuple(
            AuditEvent(
                event_type=row["event_type"],
                occurred_at_utc=str(row["logged_at_utc"]),
                payload=dict(json_loads(row["payload_json"], default={})),
            )
            for row in rows
        )

    def record_reconciliation_report(self, report: ReconciliationReport) -> None:
        payload = {
            "report_id": report.report_id,
            "generated_at_utc": report.generated_at_utc,
            "mismatches": [asdict(mismatch) for mismatch in report.mismatches],
        }
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO paper_reconciliation_reports (
                    report_id,
                    generated_at_utc,
                    mismatch_count,
                    report_json
                ) VALUES (?, ?, ?, ?)
                """,
                (
                    report.report_id,
                    report.generated_at_utc,
                    len(report.mismatches),
                    json_dumps(payload),
                ),
            )

    def reconciliation_reports(self) -> tuple[ReconciliationReport, ...]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT report_json
                FROM paper_reconciliation_reports
                ORDER BY generated_at_utc, report_id
                """
            ).fetchall()
        return tuple(_reconciliation_report_from_payload(dict(json_loads(row["report_json"], default={}))) for row in rows)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection


def _order_ack_from_payload(payload: dict[str, Any]) -> OrderAck:
    return OrderAck(
        client_order_id=str(payload["client_order_id"]),
        broker_order_id=str(payload["broker_order_id"]),
        status=payload["status"],
        submitted_at_utc=str(payload["submitted_at_utc"]),
        message=str(payload.get("message", "")),
    )


def _order_request_from_payload(payload: dict[str, Any]) -> OrderRequest:
    return OrderRequest(
        symbol=str(payload["symbol"]),
        side=payload["side"],
        quantity=int(payload["quantity"]),
        order_type=payload.get("order_type", "market"),
        time_in_force=payload.get("time_in_force", "day"),
        client_order_id=str(payload.get("client_order_id", "")),
        strategy_id=str(payload.get("strategy_id", "")),
        limit_price=None if payload.get("limit_price") is None else float(payload["limit_price"]),
        reference_price=None if payload.get("reference_price") is None else float(payload["reference_price"]),
    )


def _reconciliation_report_from_payload(payload: dict[str, Any]) -> ReconciliationReport:
    return ReconciliationReport(
        report_id=str(payload["report_id"]),
        generated_at_utc=str(payload["generated_at_utc"]),
        mismatches=tuple(
            PositionMismatch(
                symbol=str(item["symbol"]),
                expected_quantity=int(item["expected_quantity"]),
                actual_quantity=int(item["actual_quantity"]),
                expected_average_price=float(item["expected_average_price"]),
                actual_average_price=float(item["actual_average_price"]),
                reason=str(item["reason"]),
            )
            for item in payload.get("mismatches", ())
        ),
    )


def _trading_day(timestamp_utc: str) -> str:
    return datetime.fromisoformat(timestamp_utc).astimezone(NEW_YORK).date().isoformat()
