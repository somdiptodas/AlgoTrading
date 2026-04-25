from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from trader.data.models import MarketBar
from trader.evaluation.runner import ExperimentResult, FoldResult
from trader.execution.engine import BacktestResult
from trader.execution.fills import Trade
from trader.strategies.spec import StrategySpec


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def json_dumps(value: object, *, pretty: bool = False) -> str:
    value = _sanitize_json_value(value)
    if pretty:
        return json.dumps(value, indent=2, sort_keys=True, allow_nan=False)
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sanitize_json_value(value: object) -> object:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _sanitize_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_json_value(item) for item in value]
    return value


def json_loads(payload: str | bytes | bytearray | None, default: Any = None) -> Any:
    if payload in (None, ""):
        return default
    return json.loads(payload, parse_constant=lambda _: None)


def market_bar_to_payload(bar: MarketBar) -> dict[str, object]:
    return {
        "timestamp_ms": bar.timestamp_ms,
        "timestamp_utc": bar.timestamp_utc,
        "open": bar.open,
        "high": bar.high,
        "low": bar.low,
        "close": bar.close,
        "volume": bar.volume,
    }


def market_bar_from_payload(payload: dict[str, Any]) -> MarketBar:
    return MarketBar(
        timestamp_ms=int(payload["timestamp_ms"]),
        timestamp_utc=str(payload["timestamp_utc"]),
        open=float(payload["open"]),
        high=float(payload["high"]),
        low=float(payload["low"]),
        close=float(payload["close"]),
        volume=float(payload["volume"]),
    )


def trade_to_payload(trade: Trade) -> dict[str, object]:
    return {
        "entry_timestamp_utc": trade.entry_timestamp_utc,
        "exit_timestamp_utc": trade.exit_timestamp_utc,
        "entry_price": trade.entry_price,
        "exit_price": trade.exit_price,
        "shares": trade.shares,
        "bars_held": trade.bars_held,
        "pnl_cash": trade.pnl_cash,
        "pnl_pct": trade.pnl_pct,
        "exit_reason": trade.exit_reason,
    }


def trade_from_payload(payload: dict[str, Any]) -> Trade:
    return Trade(
        entry_timestamp_utc=str(payload["entry_timestamp_utc"]),
        exit_timestamp_utc=str(payload["exit_timestamp_utc"]),
        entry_price=float(payload["entry_price"]),
        exit_price=float(payload["exit_price"]),
        shares=int(payload["shares"]),
        bars_held=int(payload["bars_held"]),
        pnl_cash=float(payload["pnl_cash"]),
        pnl_pct=float(payload["pnl_pct"]),
        exit_reason=str(payload["exit_reason"]),
    )


def backtest_to_payload(backtest: BacktestResult) -> dict[str, object]:
    return {
        "bars": [market_bar_to_payload(bar) for bar in backtest.bars],
        "trades": [trade_to_payload(trade) for trade in backtest.trades],
        "equity_curve": list(backtest.equity_curve),
        "initial_cash": backtest.initial_cash,
        "final_cash": backtest.final_cash,
    }


def backtest_from_payload(payload: dict[str, Any]) -> BacktestResult:
    return BacktestResult(
        bars=tuple(market_bar_from_payload(item) for item in payload.get("bars", ())),
        trades=tuple(trade_from_payload(item) for item in payload.get("trades", ())),
        equity_curve=tuple(float(value) for value in payload.get("equity_curve", ())),
        initial_cash=float(payload.get("initial_cash", 0.0)),
        final_cash=float(payload.get("final_cash", 0.0)),
    )


def _float_dict(payload: dict[str, Any]) -> dict[str, float]:
    return {str(key): float(value) for key, value in payload.items() if value is not None}


def _robustness_dict(payload: dict[str, Any]) -> dict[str, float | bool]:
    normalized: dict[str, float | bool] = {}
    for key, value in payload.items():
        if value is None:
            continue
        normalized[str(key)] = bool(value) if isinstance(value, bool) else float(value)
    return normalized


def fold_result_to_payload(fold_result: FoldResult) -> dict[str, object]:
    return {
        "fold_id": fold_result.fold_id,
        "train_start_utc": fold_result.train_start_utc,
        "train_end_utc": fold_result.train_end_utc,
        "test_start_utc": fold_result.test_start_utc,
        "test_end_utc": fold_result.test_end_utc,
        "metrics": dict(fold_result.metrics),
        "baseline_metrics": {
            name: dict(metrics) for name, metrics in sorted(fold_result.baseline_metrics.items())
        },
        "baseline_deltas": dict(fold_result.baseline_deltas),
        "warnings": list(fold_result.warnings),
        "backtest": backtest_to_payload(fold_result.backtest),
    }


def fold_result_to_ledger_payload(fold_result: FoldResult) -> dict[str, object]:
    return {
        "fold_id": fold_result.fold_id,
        "train_start_utc": fold_result.train_start_utc,
        "train_end_utc": fold_result.train_end_utc,
        "test_start_utc": fold_result.test_start_utc,
        "test_end_utc": fold_result.test_end_utc,
        "metrics": dict(fold_result.metrics),
        "baseline_metrics": {
            name: dict(metrics) for name, metrics in sorted(fold_result.baseline_metrics.items())
        },
        "baseline_deltas": dict(fold_result.baseline_deltas),
        "warnings": list(fold_result.warnings),
        "backtest_summary": {
            "bar_count": len(fold_result.backtest.bars),
            "trade_count": len(fold_result.backtest.trades),
            "initial_cash": fold_result.backtest.initial_cash,
            "final_cash": fold_result.backtest.final_cash,
        },
    }


def fold_result_from_payload(payload: dict[str, Any]) -> FoldResult:
    baseline_metrics = {
        str(name): _float_dict(dict(metrics))
        for name, metrics in dict(payload.get("baseline_metrics", {})).items()
    }
    if "backtest" in payload:
        backtest = backtest_from_payload(dict(payload.get("backtest", {})))
    else:
        summary = dict(payload.get("backtest_summary", {}))
        backtest = BacktestResult(
            bars=tuple(),
            trades=tuple(),
            equity_curve=tuple(),
            initial_cash=float(summary.get("initial_cash", 0.0)),
            final_cash=float(summary.get("final_cash", 0.0)),
        )
    return FoldResult(
        fold_id=str(payload["fold_id"]),
        train_start_utc=str(payload["train_start_utc"]),
        train_end_utc=str(payload["train_end_utc"]),
        test_start_utc=str(payload["test_start_utc"]),
        test_end_utc=str(payload["test_end_utc"]),
        metrics=_float_dict(dict(payload.get("metrics", {}))),
        baseline_metrics=baseline_metrics,
        baseline_deltas=_float_dict(dict(payload.get("baseline_deltas", {}))),
        warnings=tuple(str(item) for item in payload.get("warnings", ())),
        backtest=backtest,
    )


def experiment_result_to_payload(result: ExperimentResult) -> dict[str, object]:
    payload = {
        "experiment_id": result.experiment_id,
        "status": result.status,
        "spec": result.spec.to_payload(),
        "spec_hash": result.spec_hash,
        "data_snapshot_id": result.data_snapshot_id,
        "split_plan_id": result.split_plan_id,
        "cost_model_id": result.cost_model_id,
        "aggregate_metrics": dict(result.aggregate_metrics),
        "fold_results": [fold_result_to_payload(item) for item in result.fold_results],
        "robustness_checks": dict(result.robustness_checks),
        "promotion_stage": result.promotion_stage,
    }
    if result.holdout_result is not None:
        payload["holdout_result"] = fold_result_to_payload(result.holdout_result)
    return payload


def experiment_result_to_ledger_payload(result: ExperimentResult) -> dict[str, object]:
    payload = {
        "experiment_id": result.experiment_id,
        "status": result.status,
        "spec": result.spec.to_payload(),
        "spec_hash": result.spec_hash,
        "data_snapshot_id": result.data_snapshot_id,
        "split_plan_id": result.split_plan_id,
        "cost_model_id": result.cost_model_id,
        "aggregate_metrics": dict(result.aggregate_metrics),
        "fold_results": [fold_result_to_ledger_payload(item) for item in result.fold_results],
        "robustness_checks": dict(result.robustness_checks),
        "promotion_stage": result.promotion_stage,
    }
    if result.holdout_result is not None:
        payload["holdout_result"] = fold_result_to_ledger_payload(result.holdout_result)
    return payload


def experiment_result_from_payload(payload: dict[str, Any]) -> ExperimentResult:
    holdout_payload = payload.get("holdout_result")
    return ExperimentResult(
        experiment_id=str(payload["experiment_id"]),
        status=str(payload.get("status", "completed")),
        spec=StrategySpec.from_payload(dict(payload.get("spec", {}))),
        spec_hash=str(payload["spec_hash"]),
        data_snapshot_id=str(payload["data_snapshot_id"]),
        split_plan_id=str(payload["split_plan_id"]),
        cost_model_id=str(payload["cost_model_id"]),
        aggregate_metrics=_float_dict(dict(payload.get("aggregate_metrics", {}))),
        fold_results=tuple(fold_result_from_payload(item) for item in payload.get("fold_results", ())),
        robustness_checks=_robustness_dict(dict(payload.get("robustness_checks", {}))),
        promotion_stage=str(payload.get("promotion_stage", "exploratory")),
        holdout_result=None if holdout_payload is None else fold_result_from_payload(dict(holdout_payload)),
    )


def result_to_json(result: ExperimentResult, *, pretty: bool = False) -> str:
    return json_dumps(experiment_result_to_payload(result), pretty=pretty)


def result_from_json(payload: str | bytes | bytearray) -> ExperimentResult:
    return experiment_result_from_payload(dict(json_loads(payload, default={})))


@dataclass(frozen=True)
class LedgerEntry:
    experiment_id: str
    evaluation_key: str
    status: str
    spec: StrategySpec
    spec_hash: str
    data_snapshot_id: str
    split_plan_id: str
    cost_model_id: str
    aggregate_metrics: dict[str, float]
    fold_results: tuple[FoldResult, ...]
    robustness_checks: dict[str, float | bool]
    promotion_stage: str
    holdout_result: FoldResult | None = None
    artifact_paths: dict[str, str] = field(default_factory=dict)
    generator_kind: str = "unknown"
    parent_experiment_ids: tuple[str, ...] = ()
    critique: dict[str, Any] | None = None
    created_at_utc: str = field(default_factory=utc_now_iso)
    updated_at_utc: str = field(default_factory=utc_now_iso)
    completed_at_utc: str | None = None

    @classmethod
    def from_result(
        cls,
        result: ExperimentResult,
        *,
        evaluation_key: str,
        artifact_paths: dict[str, str],
        generator_kind: str,
        parent_experiment_ids: tuple[str, ...] = (),
        critique: dict[str, Any] | None = None,
        created_at_utc: str | None = None,
        updated_at_utc: str | None = None,
        completed_at_utc: str | None = None,
    ) -> "LedgerEntry":
        created = created_at_utc or utc_now_iso()
        updated = updated_at_utc or created
        completed = completed_at_utc or updated
        return cls(
            experiment_id=result.experiment_id,
            evaluation_key=evaluation_key,
            status=result.status,
            spec=result.spec,
            spec_hash=result.spec_hash,
            data_snapshot_id=result.data_snapshot_id,
            split_plan_id=result.split_plan_id,
            cost_model_id=result.cost_model_id,
            aggregate_metrics=dict(result.aggregate_metrics),
            fold_results=tuple(result.fold_results),
            robustness_checks=dict(result.robustness_checks),
            promotion_stage=result.promotion_stage,
            holdout_result=result.holdout_result,
            artifact_paths=dict(sorted((str(key), str(value)) for key, value in artifact_paths.items())),
            generator_kind=generator_kind,
            parent_experiment_ids=tuple(parent_experiment_ids),
            critique=critique,
            created_at_utc=created,
            updated_at_utc=updated,
            completed_at_utc=completed,
        )

    def metric(self, name: str, default: float = 0.0) -> float:
        return float(self.aggregate_metrics.get(name, default))

    def to_result(self) -> ExperimentResult:
        return ExperimentResult(
            experiment_id=self.experiment_id,
            status=self.status,
            spec=self.spec,
            spec_hash=self.spec_hash,
            data_snapshot_id=self.data_snapshot_id,
            split_plan_id=self.split_plan_id,
            cost_model_id=self.cost_model_id,
            aggregate_metrics=dict(self.aggregate_metrics),
            fold_results=tuple(self.fold_results),
            robustness_checks=dict(self.robustness_checks),
            promotion_stage=self.promotion_stage,
            holdout_result=self.holdout_result,
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "experiment_id": self.experiment_id,
            "evaluation_key": self.evaluation_key,
            "status": self.status,
            "result": experiment_result_to_ledger_payload(self.to_result()),
            "artifact_paths": dict(self.artifact_paths),
            "generator_kind": self.generator_kind,
            "parent_experiment_ids": list(self.parent_experiment_ids),
            "critique": self.critique,
            "created_at_utc": self.created_at_utc,
            "updated_at_utc": self.updated_at_utc,
            "completed_at_utc": self.completed_at_utc,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "LedgerEntry":
        result = experiment_result_from_payload(dict(payload.get("result", {})))
        return cls.from_result(
            result,
            evaluation_key=str(payload["evaluation_key"]),
            artifact_paths={
                str(key): str(value) for key, value in dict(payload.get("artifact_paths", {})).items()
            },
            generator_kind=str(payload.get("generator_kind", "unknown")),
            parent_experiment_ids=tuple(str(item) for item in payload.get("parent_experiment_ids", ())),
            critique=payload.get("critique"),
            created_at_utc=str(payload.get("created_at_utc", utc_now_iso())),
            updated_at_utc=str(payload.get("updated_at_utc", utc_now_iso())),
            completed_at_utc=(
                None if payload.get("completed_at_utc") is None else str(payload.get("completed_at_utc"))
            ),
        )


def entry_to_json(entry: LedgerEntry, *, pretty: bool = False) -> str:
    return json_dumps(entry.to_payload(), pretty=pretty)


def entry_from_json(payload: str | bytes | bytearray) -> LedgerEntry:
    return LedgerEntry.from_payload(dict(json_loads(payload, default={})))
