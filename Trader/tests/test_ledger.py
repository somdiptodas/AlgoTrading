from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from trader.artifacts.store import ArtifactStore
from trader.data.models import MarketBar
from trader.evaluation.runner import ExperimentResult, FoldResult
from trader.execution.engine import BacktestResult
from trader.execution.fills import Trade
from trader.ledger.store import LedgerStore
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
        },
        fold_results=(fold,),
        robustness_checks={"fold_consistency_pass": True, "neighborhood_pass": True},
        promotion_stage="frontier",
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
    assert len(ledger.list_completed()) == 1
    assert len(ledger.top_experiments()) == 1
    for path in artifact_paths.values():
        assert Path(path).exists()
