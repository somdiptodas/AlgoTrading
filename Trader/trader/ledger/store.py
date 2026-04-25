from __future__ import annotations

import json
import sqlite3
from hashlib import sha256
from pathlib import Path
from typing import Any, Sequence

from trader.evaluation.runner import ExperimentResult
from trader.ledger.entry import LedgerEntry, entry_from_json, entry_to_json, utc_now_iso
from trader.ledger.query import LedgerQueryHelper
from trader.research.suppressor import SuppressedSpec


SCHEMA = """
CREATE TABLE IF NOT EXISTS suppression_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    loop_run_id TEXT NOT NULL,
    spec_hash TEXT NOT NULL,
    signal_family TEXT NOT NULL,
    nearest_failure_experiment_id TEXT NOT NULL,
    nearest_failure_distance REAL NOT NULL,
    failed_check_names TEXT NOT NULL,
    failure_count_in_radius INTEGER NOT NULL DEFAULT 0,
    suppression_weight REAL NOT NULL,
    logged_at_utc TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_suppression_log_run
ON suppression_log (loop_run_id, logged_at_utc DESC);

CREATE INDEX IF NOT EXISTS idx_suppression_log_family
ON suppression_log (signal_family, suppression_weight DESC);

CREATE TABLE IF NOT EXISTS ledger_entries (
    experiment_id TEXT PRIMARY KEY,
    evaluation_key TEXT NOT NULL UNIQUE,
    status TEXT NOT NULL,
    signal_family TEXT NOT NULL,
    promotion_stage TEXT NOT NULL,
    spec_hash TEXT NOT NULL,
    data_snapshot_id TEXT NOT NULL,
    split_plan_id TEXT NOT NULL,
    cost_model_id TEXT NOT NULL,
    generator_kind TEXT NOT NULL,
    return_pct REAL NOT NULL DEFAULT 0.0,
    sharpe_like REAL NOT NULL DEFAULT 0.0,
    max_drawdown_pct REAL NOT NULL DEFAULT 0.0,
    delta_buy_and_hold_return_pct REAL NOT NULL DEFAULT 0.0,
    fold_consistency_pass INTEGER NOT NULL DEFAULT 0,
    neighborhood_pass INTEGER NOT NULL DEFAULT 0,
    created_at_utc TEXT NOT NULL,
    updated_at_utc TEXT NOT NULL,
    completed_at_utc TEXT,
    entry_json TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_ledger_entries_status_completed
ON ledger_entries (status, completed_at_utc DESC);

CREATE INDEX IF NOT EXISTS idx_ledger_entries_stage_metrics
ON ledger_entries (promotion_stage, sharpe_like DESC, return_pct DESC, max_drawdown_pct ASC);

CREATE INDEX IF NOT EXISTS idx_ledger_entries_signal_family
ON ledger_entries (signal_family, completed_at_utc DESC);
"""


class LedgerStore:
    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)
        self.query = LedgerQueryHelper()

    def initialize(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(SCHEMA)

    def evaluation_key(
        self,
        spec_hash: str,
        data_snapshot_id: str,
        split_plan_id: str,
        cost_model_id: str,
    ) -> str:
        payload = "|".join([spec_hash, data_snapshot_id, split_plan_id, cost_model_id])
        return sha256(payload.encode("utf-8")).hexdigest()

    def evaluation_key_for_components(
        self,
        spec_hash: str,
        data_snapshot_id: str,
        split_plan_id: str,
        cost_model_id: str,
    ) -> str:
        return self.evaluation_key(spec_hash, data_snapshot_id, split_plan_id, cost_model_id)

    def get_by_evaluation_key(self, key: str) -> LedgerEntry | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT entry_json FROM ledger_entries WHERE evaluation_key = ?",
                (key,),
            ).fetchone()
        if row is None:
            return None
        return entry_from_json(row["entry_json"])

    def upsert_entry(self, entry: LedgerEntry) -> LedgerEntry:
        existing = self.get_by_evaluation_key(entry.evaluation_key) or self._get_by_experiment_id(entry.experiment_id)
        target = entry if existing is None else LedgerEntry.from_payload(
            {
                **entry.to_payload(),
                "experiment_id": existing.experiment_id,
                "result": {
                    **entry.to_payload()["result"],
                    "experiment_id": existing.experiment_id,
                },
                "created_at_utc": existing.created_at_utc,
            }
        )
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO ledger_entries (
                    experiment_id,
                    evaluation_key,
                    status,
                    signal_family,
                    promotion_stage,
                    spec_hash,
                    data_snapshot_id,
                    split_plan_id,
                    cost_model_id,
                    generator_kind,
                    return_pct,
                    sharpe_like,
                    max_drawdown_pct,
                    delta_buy_and_hold_return_pct,
                    fold_consistency_pass,
                    neighborhood_pass,
                    created_at_utc,
                    updated_at_utc,
                    completed_at_utc,
                    entry_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(experiment_id) DO UPDATE SET
                    evaluation_key = excluded.evaluation_key,
                    status = excluded.status,
                    signal_family = excluded.signal_family,
                    promotion_stage = excluded.promotion_stage,
                    spec_hash = excluded.spec_hash,
                    data_snapshot_id = excluded.data_snapshot_id,
                    split_plan_id = excluded.split_plan_id,
                    cost_model_id = excluded.cost_model_id,
                    generator_kind = excluded.generator_kind,
                    return_pct = excluded.return_pct,
                    sharpe_like = excluded.sharpe_like,
                    max_drawdown_pct = excluded.max_drawdown_pct,
                    delta_buy_and_hold_return_pct = excluded.delta_buy_and_hold_return_pct,
                    fold_consistency_pass = excluded.fold_consistency_pass,
                    neighborhood_pass = excluded.neighborhood_pass,
                    updated_at_utc = excluded.updated_at_utc,
                    completed_at_utc = excluded.completed_at_utc,
                    entry_json = excluded.entry_json
                """,
                self._row_values(target),
            )
        return target

    def record_result(
        self,
        result: ExperimentResult,
        artifact_paths: dict[str, str],
        generator_kind: str,
        parent_experiment_ids: tuple[str, ...] = (),
        critique: dict[str, object] | None = None,
    ) -> LedgerEntry:
        key = self.evaluation_key(
            result.spec_hash,
            result.data_snapshot_id,
            result.split_plan_id,
            result.cost_model_id,
        )
        existing = self.get_by_evaluation_key(key)
        if existing is not None:
            return existing
        entry = LedgerEntry.from_result(
            result,
            evaluation_key=key,
            artifact_paths={str(name): str(path) for name, path in artifact_paths.items()},
            generator_kind=generator_kind,
            parent_experiment_ids=tuple(parent_experiment_ids),
            critique=None if critique is None else dict(critique),
        )
        return self.upsert_entry(entry)

    def list_completed(self, limit: int = 20) -> tuple[LedgerEntry, ...]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT entry_json
                FROM ledger_entries
                WHERE status = 'completed'
                ORDER BY completed_at_utc DESC, experiment_id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return tuple(entry_from_json(row["entry_json"]) for row in rows)

    def top_experiments(self, limit: int = 10) -> tuple[LedgerEntry, ...]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT entry_json
                FROM ledger_entries
                WHERE status = 'completed'
                ORDER BY completed_at_utc DESC, experiment_id DESC
                """
            ).fetchall()
        entries = [entry_from_json(row["entry_json"]) for row in rows]
        return self.query.top_experiments(entries, limit=limit)

    def stats(self) -> dict[str, Any]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT status, COUNT(*) AS count FROM ledger_entries GROUP BY status"
            ).fetchall()
        counts = {str(row["status"]): int(row["count"]) for row in rows}
        return {
            "total": sum(counts.values()),
            "by_status": counts,
        }

    def log_suppression_batch(
        self,
        loop_run_id: str,
        records: Sequence[SuppressedSpec],
    ) -> int:
        """
        Persist suppression audit records for this loop run.
        Returns the number of rows written.
        """
        if not records:
            return 0
        now = utc_now_iso()
        rows = [
            (
                loop_run_id,
                record.spec_hash,
                record.signal_family,
                record.nearest_failure_experiment_id,
                record.nearest_failure_distance,
                json.dumps(list(record.failed_check_names)),
                record.failure_count_in_radius,
                record.suppression_weight,
                now,
            )
            for record in records
        ]
        with self._connect() as connection:
            connection.executemany(
                """
                INSERT INTO suppression_log (
                    loop_run_id, spec_hash, signal_family,
                    nearest_failure_experiment_id, nearest_failure_distance,
                    failed_check_names, failure_count_in_radius,
                    suppression_weight, logged_at_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
        return len(rows)

    def suppression_summary(self, loop_run_id: str) -> dict[str, Any]:
        """Return a summary of suppression decisions for a given loop run."""
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT signal_family, COUNT(*) AS count,
                       AVG(suppression_weight) AS avg_weight,
                       MAX(suppression_weight) AS max_weight
                FROM suppression_log
                WHERE loop_run_id = ?
                GROUP BY signal_family
                ORDER BY signal_family
                """,
                (loop_run_id,),
            ).fetchall()
        return {
            "loop_run_id": loop_run_id,
            "by_family": [
                {
                    "family": str(row["signal_family"]),
                    "suppressed_count": int(row["count"]),
                    "avg_weight": round(float(row["avg_weight"]), 3),
                    "max_weight": round(float(row["max_weight"]), 3),
                }
                for row in rows
            ],
        }

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _get_by_experiment_id(self, experiment_id: str) -> LedgerEntry | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT entry_json FROM ledger_entries WHERE experiment_id = ?",
                (experiment_id,),
            ).fetchone()
        if row is None:
            return None
        return entry_from_json(row["entry_json"])

    def _row_values(self, entry: LedgerEntry) -> tuple[object, ...]:
        return (
            entry.experiment_id,
            entry.evaluation_key,
            entry.status,
            entry.spec.signal.name,
            entry.promotion_stage,
            entry.spec_hash,
            entry.data_snapshot_id,
            entry.split_plan_id,
            entry.cost_model_id,
            entry.generator_kind,
            entry.metric("return_pct"),
            entry.metric("sharpe_like"),
            entry.metric("max_drawdown_pct"),
            entry.metric("delta_buy_and_hold_return_pct"),
            int(bool(entry.robustness_checks.get("fold_consistency_pass"))),
            int(bool(entry.robustness_checks.get("neighborhood_pass"))),
            entry.created_at_utc,
            entry.updated_at_utc,
            entry.completed_at_utc,
            entry_to_json(entry),
        )
