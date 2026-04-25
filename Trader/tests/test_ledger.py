from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from trader.artifacts.store import ArtifactStore
from trader.data.models import MarketBar
from trader.evaluation.runner import ExperimentResult, FoldResult
from trader.execution.engine import BacktestResult
from trader.execution.fills import Trade
from trader.ledger.entry import (
    LedgerEntry,
    entry_from_json,
    experiment_result_to_ledger_payload,
    experiment_result_to_payload,
    json_dumps,
    json_loads,
)
from trader.ledger.query import LedgerQueryHelper
import trader.ledger.store as ledger_store_module
from trader.ledger.store import SCHEMA, LedgerStore
from trader.research.suppressor import SuppressedSpec
from trader.reporting.report import render_experiment_report
from trader.strategies.spec import SignalSpec, StrategySpec


NEW_YORK = ZoneInfo("America/New_York")


def _sample_result() -> ExperimentResult:
    start = datetime(2026, 1, 5, 9, 30, tzinfo=NEW_YORK)
    bars = []
    for index in range(3):
        timestamp = (start + timedelta(minutes=index)).astimezone(timezone.utc)
        price = 100.0 + index
        bars.append(
            MarketBar(
                timestamp_ms=int(timestamp.timestamp() * 1000),
                timestamp_utc=timestamp.isoformat(),
                open=price,
                high=price + 0.25,
                low=price - 0.25,
                close=price,
                volume=1_000.0,
            )
        )
    backtest = BacktestResult(
        bars=tuple(bars),
        trades=(
            Trade("a", "b", 100.0, 101.0, 10, 2, 10.0, 1.0, "signal_flip"),
        ),
        equity_curve=(100_000.0, 100_010.0, 100_015.0),
        initial_cash=100_000.0,
        final_cash=100_015.0,
    )
    fold = FoldResult(
        fold_id="fold_1",
        train_start_utc=bars[0].timestamp_utc,
        train_end_utc=bars[1].timestamp_utc,
        test_start_utc=bars[1].timestamp_utc,
        test_end_utc=bars[2].timestamp_utc,
        metrics={
            "return_pct": 0.015,
            "sharpe_like": 1.2,
            "max_drawdown_pct": 0.0,
            "exposure_pct": 66.0,
            "trade_count": 1.0,
            "win_rate_pct": 100.0,
            "profit_factor": 1.0,
            "avg_trade_pct": 1.0,
        },
        baseline_metrics={
            "always_flat": {"return_pct": 0.0},
            "buy_and_hold": {"return_pct": 0.01},
        },
        baseline_deltas={
            "delta_always_flat_return_pct": 0.015,
            "delta_buy_and_hold_return_pct": 0.005,
            "delta_exposure_adjusted_buy_and_hold_pct": 0.0084,
            "delta_always_flat_sharpe_like": 1.2,
            "delta_buy_and_hold_sharpe_like": 0.2,
        },
        warnings=tuple(),
        backtest=backtest,
    )
    spec = StrategySpec(
        name="ema_roundtrip",
        signal=SignalSpec("ema_cross", {"fast_length": 20, "slow_length": 80, "signal_buffer_bps": 0.0}),
    )
    return ExperimentResult(
        experiment_id="exp_roundtrip",
        status="completed",
        spec=spec,
        spec_hash=spec.spec_hash(),
        data_snapshot_id="snapshot123",
        split_plan_id="split123",
        cost_model_id=spec.exec_config.cost_model_id(),
        aggregate_metrics={
            "return_pct": 0.015,
            "sharpe_like": 1.2,
            "max_drawdown_pct": 0.0,
            "trade_count": 1.0,
            "delta_buy_and_hold_return_pct": 0.005,
            "delta_exposure_adjusted_buy_and_hold_pct": 0.0084,
        },
        fold_results=(fold,),
        robustness_checks={"fold_consistency_pass": True, "neighborhood_pass": True},
        promotion_stage="research_frontier",
    )


def test_ledger_round_trip_and_dedupe(tmp_path: Path) -> None:
    result = _sample_result()
    ledger = LedgerStore(tmp_path / "ledger.db")
    ledger.initialize()
    artifacts = ArtifactStore(tmp_path / "artifacts", tmp_path / "reports")
    report = render_experiment_report(result, critique={"verdict": "ok"})
    artifact_paths = artifacts.write_experiment(result, report_markdown=report, critique={"verdict": "ok"})

    first = ledger.record_result(result, artifact_paths=artifact_paths, generator_kind="grid")
    second = ledger.record_result(result, artifact_paths=artifact_paths, generator_kind="grid")

    assert first.experiment_id == second.experiment_id
    fetched = ledger.get_by_evaluation_key(first.evaluation_key)
    assert fetched is not None
    assert fetched.spec_hash == result.spec_hash
    assert fetched.to_result().aggregate_metrics["sharpe_like"] == result.aggregate_metrics["sharpe_like"]
    assert (
        fetched.to_result().aggregate_metrics["delta_exposure_adjusted_buy_and_hold_pct"]
        == result.aggregate_metrics["delta_exposure_adjusted_buy_and_hold_pct"]
    )
    assert len(ledger.list_completed()) == 1
    assert len(ledger.top_experiments()) == 1
    for path in artifact_paths.values():
        assert Path(path).exists()


def test_ledger_preserves_critique_planning_penalties(tmp_path: Path) -> None:
    result = _sample_result()
    ledger = LedgerStore(tmp_path / "ledger.db")
    ledger.initialize()
    critique = {
        "verdict": "fragile",
        "notes": ["Strategy does not beat buy-and-hold on average OOS return."],
        "next_focus": ["Search for lower-drawdown variants before promoting."],
        "planning_penalties": {"benchmark_failure": 10.0},
    }

    entry = ledger.record_result(result, artifact_paths={}, generator_kind="grid", critique=critique)
    fetched = ledger.get_by_evaluation_key(entry.evaluation_key)

    assert fetched is not None
    assert fetched.critique == critique


def test_suppression_log_separates_audit_types(tmp_path: Path) -> None:
    ledger = LedgerStore(tmp_path / "ledger.db")
    ledger.initialize()
    evaluated = SuppressedSpec(
        spec_hash="evaluated_hash",
        signal_family="ema_cross",
        nearest_failure_experiment_id="failure_1",
        nearest_failure_distance=0.0,
        failed_check_names=("neighborhood_pass",),
        failure_count_in_radius=2,
        suppression_weight=20.0,
    )
    discarded = SuppressedSpec(
        spec_hash="discarded_hash",
        signal_family="ema_cross",
        nearest_failure_experiment_id="failure_2",
        nearest_failure_distance=0.1,
        failed_check_names=("regime_pass",),
        failure_count_in_radius=1,
        suppression_weight=5.0,
    )

    logged = ledger.log_suppression_batch(
        "loop_1",
        (evaluated, discarded),
        audit_type_by_spec_hash={
            "evaluated_hash": "evaluated",
            "discarded_hash": "preview_discarded",
        },
    )
    summary = ledger.suppression_summary("loop_1")

    assert logged == 2
    assert summary["by_type"] == [
        {"audit_type": "evaluated", "suppressed_count": 1},
        {"audit_type": "preview_discarded", "suppressed_count": 1},
    ]


def test_legacy_suppression_log_gets_audit_type_column(tmp_path: Path) -> None:
    db_path = tmp_path / "ledger.db"
    with sqlite3.connect(db_path) as connection:
        connection.executescript(
            """
            CREATE TABLE suppression_log (
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
            """
        )
        connection.execute(
            """
            INSERT INTO suppression_log (
                loop_run_id, spec_hash, signal_family,
                nearest_failure_experiment_id, nearest_failure_distance,
                failed_check_names, failure_count_in_radius,
                suppression_weight, logged_at_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("loop_legacy", "legacy_hash", "ema_cross", "failure_1", 0.0, "[]", 1, 5.0, "2026-01-01T00:00:00+00:00"),
        )

    ledger = LedgerStore(db_path)
    ledger.initialize()

    with sqlite3.connect(db_path) as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(suppression_log)").fetchall()}
        audit_type = connection.execute(
            "SELECT audit_type FROM suppression_log WHERE spec_hash = ?",
            ("legacy_hash",),
        ).fetchone()[0]

    assert "audit_type" in columns
    assert audit_type == "previewed"


def test_ledger_entry_json_is_compact_and_artifacts_keep_full_details(tmp_path: Path) -> None:
    result = _sample_result()
    ledger = LedgerStore(tmp_path / "ledger.db")
    ledger.initialize()
    artifacts = ArtifactStore(tmp_path / "artifacts", tmp_path / "reports")
    artifact_paths = artifacts.write_experiment(result)
    entry = ledger.record_result(result, artifact_paths=artifact_paths, generator_kind="grid")

    with sqlite3.connect(ledger.database_path) as connection:
        raw_entry_json = connection.execute(
            "SELECT entry_json FROM ledger_entries WHERE experiment_id = ?",
            (entry.experiment_id,),
        ).fetchone()[0]

    assert '"bars"' not in raw_entry_json
    assert '"equity_curve"' not in raw_entry_json

    entry_payload = json.loads(raw_entry_json)
    fold_payload = entry_payload["result"]["fold_results"][0]
    assert "backtest" not in fold_payload
    assert "trades" not in fold_payload
    assert fold_payload["backtest_summary"] == {
        "bar_count": 3,
        "final_cash": 100_015.0,
        "initial_cash": 100_000.0,
        "trade_count": 1,
    }

    artifact_result = json.loads(Path(artifact_paths["result"]).read_text(encoding="utf-8"))
    assert len(artifact_result["fold_results"][0]["backtest"]["bars"]) == 3
    assert len(artifact_result["fold_results"][0]["backtest"]["trades"]) == 1
    assert len(artifact_result["fold_results"][0]["backtest"]["equity_curve"]) == 3

    fetched = ledger.get_by_evaluation_key(entry.evaluation_key)
    assert fetched is not None
    assert fetched.aggregate_metrics == result.aggregate_metrics
    assert fetched.fold_results[0].metrics == result.fold_results[0].metrics
    assert fetched.fold_results[0].backtest.bars == tuple()
    assert fetched.fold_results[0].backtest.trades == tuple()
    assert fetched.fold_results[0].backtest.equity_curve == tuple()


def test_legacy_full_ledger_entry_json_still_reads() -> None:
    result = _sample_result()
    legacy_payload = {
        "experiment_id": result.experiment_id,
        "evaluation_key": "legacy_key",
        "status": result.status,
        "result": experiment_result_to_payload(result),
        "artifact_paths": {},
        "generator_kind": "grid",
        "parent_experiment_ids": [],
        "critique": None,
        "created_at_utc": "2026-01-05T00:00:00+00:00",
        "updated_at_utc": "2026-01-05T00:00:00+00:00",
        "completed_at_utc": "2026-01-05T00:00:00+00:00",
    }

    entry = entry_from_json(json_dumps(legacy_payload))

    assert entry.evaluation_key == "legacy_key"
    assert len(entry.fold_results[0].backtest.bars) == 3
    assert len(entry.fold_results[0].backtest.trades) == 1
    assert entry.fold_results[0].backtest.equity_curve == (100_000.0, 100_010.0, 100_015.0)


def test_holdout_result_serializes_separately_from_research_folds() -> None:
    result = _sample_result()
    holdout = replace(result.fold_results[0], fold_id="holdout")
    with_holdout = replace(result, holdout_result=holdout)

    payload = experiment_result_to_payload(with_holdout)
    ledger_payload = experiment_result_to_ledger_payload(with_holdout)
    round_tripped = entry_from_json(
        json_dumps(
            {
                "experiment_id": result.experiment_id,
                "evaluation_key": "holdout_key",
                "status": result.status,
                "result": ledger_payload,
                "artifact_paths": {},
                "generator_kind": "grid",
                "parent_experiment_ids": [],
                "critique": None,
                "created_at_utc": "2026-01-05T00:00:00+00:00",
                "updated_at_utc": "2026-01-05T00:00:00+00:00",
                "completed_at_utc": "2026-01-05T00:00:00+00:00",
            }
        )
    )

    assert len(payload["fold_results"]) == 1
    assert payload["holdout_result"]["fold_id"] == "holdout"
    assert len(ledger_payload["fold_results"]) == 1
    assert "backtest_summary" in ledger_payload["holdout_result"]
    assert round_tripped.holdout_result is not None
    assert round_tripped.holdout_result.fold_id == "holdout"


def test_ledger_round_trip_preserves_holdout_result(tmp_path: Path) -> None:
    result = _sample_result()
    holdout = replace(result.fold_results[0], fold_id="holdout")
    with_holdout = replace(result, holdout_result=holdout)
    ledger = LedgerStore(tmp_path / "ledger.db")
    ledger.initialize()

    entry = ledger.record_result(with_holdout, artifact_paths={}, generator_kind="grid")
    fetched = ledger.get_by_evaluation_key(entry.evaluation_key)

    assert fetched is not None
    assert fetched.to_result().holdout_result is not None
    assert fetched.to_result().holdout_result.fold_id == "holdout"


def test_report_renders_holdout_warnings() -> None:
    result = _sample_result()
    holdout = replace(result.fold_results[0], fold_id="holdout", warnings=("data_quality.missing_bars: gap",))
    report = render_experiment_report(replace(result, holdout_result=holdout))

    assert "## Holdout" in report
    assert "Warnings: data_quality.missing_bars: gap" in report


def test_report_renders_exposure_adjusted_buy_hold_delta() -> None:
    report = render_experiment_report(_sample_result())

    assert "Return vs exposure-adjusted buy and hold" in report


def test_top_experiments_ranks_from_scalar_columns_before_deserializing_winners(
    tmp_path: Path,
    monkeypatch,
) -> None:
    ledger = LedgerStore(tmp_path / "ledger.db")
    ledger.initialize()
    base = _sample_result()

    for index, metrics in enumerate(
        (
            {"return_pct": 1.0, "sharpe_like": 0.1, "max_drawdown_pct": 4.0, "trade_count": 2.0},
            {"return_pct": 4.0, "sharpe_like": 0.8, "max_drawdown_pct": 1.0, "trade_count": 4.0},
            {"return_pct": 2.0, "sharpe_like": 0.3, "max_drawdown_pct": 3.0, "trade_count": 3.0},
        )
    ):
        result = replace(
            base,
            experiment_id=f"exp_{index}",
            aggregate_metrics={
                **base.aggregate_metrics,
                **metrics,
                "delta_buy_and_hold_return_pct": metrics["return_pct"] - 0.5,
            },
        )
        entry = LedgerEntry.from_result(
            result,
            evaluation_key=f"key_{index}",
            artifact_paths={},
            generator_kind="grid",
        )
        ledger.upsert_entry(entry)

    parse_count = 0
    original_entry_from_json = ledger_store_module.entry_from_json

    def counting_entry_from_json(payload: str | bytes | bytearray) -> LedgerEntry:
        nonlocal parse_count
        parse_count += 1
        return original_entry_from_json(payload)

    monkeypatch.setattr(ledger_store_module, "entry_from_json", counting_entry_from_json)

    top = ledger.top_experiments(limit=1)

    assert [entry.experiment_id for entry in top] == ["exp_1"]
    assert parse_count == 1


def test_promoted_experiments_excludes_legacy_frontier_stage() -> None:
    base = _sample_result()
    entries = (
        LedgerEntry.from_result(
            replace(
                base,
                experiment_id="legacy_frontier",
                promotion_stage="frontier",
                aggregate_metrics={
                    **base.aggregate_metrics,
                    "return_pct": -1.0,
                    "delta_buy_and_hold_return_pct": -5.0,
                },
            ),
            evaluation_key="legacy_frontier_key",
            artifact_paths={},
            generator_kind="grid",
        ),
        LedgerEntry.from_result(
            replace(base, experiment_id="research_frontier", promotion_stage="research_frontier"),
            evaluation_key="research_frontier_key",
            artifact_paths={},
            generator_kind="grid",
        ),
        LedgerEntry.from_result(
            replace(base, experiment_id="candidate", promotion_stage="candidate"),
            evaluation_key="candidate_key",
            artifact_paths={},
            generator_kind="grid",
        ),
    )

    promoted = LedgerQueryHelper().promoted_experiments(entries, limit=10)

    assert {entry.experiment_id for entry in promoted} == {"research_frontier", "candidate"}


def test_initialize_migrates_legacy_ledger_trade_count_and_compacts_json(tmp_path: Path) -> None:
    database_path = tmp_path / "legacy_ledger.db"
    result = _sample_result()
    entry = LedgerEntry.from_result(
        result,
        evaluation_key="legacy_key",
        artifact_paths={},
        generator_kind="grid",
    )
    legacy_payload = {
        **entry.to_payload(),
        "result": experiment_result_to_payload(result),
    }
    legacy_entry_json = json_dumps(legacy_payload)
    old_schema = SCHEMA.replace("    trade_count REAL NOT NULL DEFAULT 0.0,\n", "")

    with sqlite3.connect(database_path) as connection:
        connection.executescript(old_schema)
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
            """,
            (
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
                legacy_entry_json,
            ),
        )

    LedgerStore(database_path).initialize()

    with sqlite3.connect(database_path) as connection:
        connection.row_factory = sqlite3.Row
        columns = {
            str(row["name"])
            for row in connection.execute("PRAGMA table_info(ledger_entries)").fetchall()
        }
        row = connection.execute(
            "SELECT trade_count, entry_json FROM ledger_entries WHERE experiment_id = ?",
            (entry.experiment_id,),
        ).fetchone()

    assert "trade_count" in columns
    assert row["trade_count"] == result.aggregate_metrics["trade_count"]
    migrated_payload = json.loads(row["entry_json"])
    fold_payload = migrated_payload["result"]["fold_results"][0]
    assert "backtest" not in fold_payload
    assert fold_payload["backtest_summary"]["bar_count"] == 3


def test_json_dumps_sanitizes_non_finite_values() -> None:
    payload = json_dumps(
        {
            "positive": math.inf,
            "negative": -math.inf,
            "missing": math.nan,
            "nested": [1.0, math.inf],
        },
        pretty=True,
    )

    assert "Infinity" not in payload
    assert "-Infinity" not in payload
    assert "NaN" not in payload
    parsed = json.loads(payload)
    assert parsed == {
        "missing": None,
        "negative": None,
        "nested": [1.0, None],
        "positive": None,
    }
    assert json_loads('{"legacy": Infinity, "missing": NaN}') == {"legacy": None, "missing": None}


def test_persisted_ledger_and_artifacts_use_standard_json_for_non_finite_metrics(tmp_path: Path) -> None:
    result = _result_with_non_finite_metrics()
    artifacts = ArtifactStore(tmp_path / "artifacts", tmp_path / "reports")
    artifact_paths = artifacts.write_experiment(result)

    ledger = LedgerStore(tmp_path / "ledger.db")
    ledger.initialize()
    entry = ledger.record_result(result, artifact_paths=artifact_paths, generator_kind="grid")

    with sqlite3.connect(ledger.database_path) as connection:
        raw_entry_json = connection.execute(
            "SELECT entry_json FROM ledger_entries WHERE experiment_id = ?",
            (entry.experiment_id,),
        ).fetchone()[0]

    serialized_payloads = [
        raw_entry_json,
        Path(artifact_paths["result"]).read_text(encoding="utf-8"),
        Path(artifact_paths["manifest"]).read_text(encoding="utf-8"),
    ]
    for payload in serialized_payloads:
        assert "Infinity" not in payload
        assert "-Infinity" not in payload
        assert "NaN" not in payload
        json.loads(payload)

    result_payload = json.loads(Path(artifact_paths["result"]).read_text(encoding="utf-8"))
    assert result_payload["aggregate_metrics"]["profit_factor"] is None
    assert result_payload["fold_results"][0]["metrics"]["profit_factor"] is None
    assert result_payload["fold_results"][0]["baseline_metrics"]["buy_and_hold"]["profit_factor"] is None

    fetched = ledger.get_by_evaluation_key(entry.evaluation_key)
    assert fetched is not None
    assert "profit_factor" not in fetched.aggregate_metrics


def test_artifact_manifest_records_generator_kind(tmp_path: Path) -> None:
    result = _sample_result()
    artifacts = ArtifactStore(tmp_path / "artifacts", tmp_path / "reports")

    artifact_paths = artifacts.write_experiment(result, generator_kind="frontier_neighborhood")

    manifest = json.loads(Path(artifact_paths["manifest"]).read_text(encoding="utf-8"))
    assert manifest["generator_kind"] == "frontier_neighborhood"


def _result_with_non_finite_metrics() -> ExperimentResult:
    result = _sample_result()
    fold = result.fold_results[0]
    tainted_fold = replace(
        fold,
        metrics={
            **fold.metrics,
            "profit_factor": math.inf,
            "nan_metric": math.nan,
        },
        baseline_metrics={
            **fold.baseline_metrics,
            "buy_and_hold": {
                **fold.baseline_metrics["buy_and_hold"],
                "profit_factor": math.inf,
            },
        },
        baseline_deltas={
            **fold.baseline_deltas,
            "nan_delta": math.nan,
        },
    )
    return replace(
        result,
        aggregate_metrics={
            **result.aggregate_metrics,
            "profit_factor": math.inf,
            "nan_metric": math.nan,
        },
        fold_results=(tainted_fold,),
        robustness_checks={
            **result.robustness_checks,
            "nan_check": math.nan,
        },
    )
