from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

from trader.evaluation.runner import DEFAULT_LOCKED_HOLDOUT_MONTHS, EvaluationRunner
from trader.ledger.entry import LedgerEntry
from trader.ledger.query import PROMOTED_PROMOTION_STAGES
from trader.ledger.store import LedgerStore


DECAY_MONITOR_GENERATOR_KIND = "decay_monitor"


@dataclass(frozen=True)
class StrategyDecayItem:
    spec_hash: str
    source_experiment_id: str
    latest_experiment_id: str
    source_data_snapshot_id: str
    latest_data_snapshot_id: str
    source_promotion_stage: str
    latest_promotion_stage: str
    evaluation_count: int
    return_delta_pct: float
    sharpe_delta: float
    drawdown_delta_pct: float
    rolling_return_delta_pct: float
    rolling_sharpe_delta: float
    rolling_drawdown_delta_pct: float
    rolling_degradation_count: int
    status: str

    def to_payload(self) -> dict[str, object]:
        return asdict(self)


def reevaluate_promoted_specs(
    ledger: LedgerStore,
    runner: EvaluationRunner,
    *,
    limit: int = 10,
    history_limit: int = 10_000,
    num_folds: int = 3,
    embargo_bars: int = 1,
    locked_holdout_months: int | None = DEFAULT_LOCKED_HOLDOUT_MONTHS,
    include_robustness: bool = True,
) -> tuple[LedgerEntry, ...]:
    history = ledger.list_completed(limit=history_limit)
    promoted = ledger.query.promoted_experiments(history, limit=limit)
    recorded: list[LedgerEntry] = []
    seen_spec_hashes: set[str] = set()
    for source in promoted:
        if source.spec_hash in seen_spec_hashes:
            continue
        seen_spec_hashes.add(source.spec_hash)
        current_key = runner.evaluation_key_for_spec(
            source.spec,
            num_folds=num_folds,
            embargo_bars=embargo_bars,
            locked_holdout_months=locked_holdout_months,
        )
        if ledger.get_by_evaluation_key(current_key) is not None:
            continue
        result = runner.evaluate_walk_forward(
            source.spec,
            num_folds=num_folds,
            embargo_bars=embargo_bars,
            locked_holdout_months=locked_holdout_months,
            include_robustness=include_robustness,
        )
        recorded.append(
            ledger.record_result(
                result,
                artifact_paths={},
                generator_kind=DECAY_MONITOR_GENERATOR_KIND,
                parent_experiment_ids=(source.experiment_id,),
            )
        )
    return tuple(recorded)


def build_decay_report(
    entries: Iterable[LedgerEntry],
    *,
    current_snapshot_id: str | None = None,
    limit: int = 10,
    return_drop_warn_pct: float = 1.0,
    sharpe_drop_warn: float = 0.25,
    drawdown_increase_warn_pct: float = 1.0,
) -> tuple[StrategyDecayItem, ...]:
    completed = [entry for entry in entries if entry.status == "completed"]
    promoted_by_hash: dict[str, list[LedgerEntry]] = {}
    entries_by_hash: dict[str, list[LedgerEntry]] = {}
    for entry in completed:
        entries_by_hash.setdefault(entry.spec_hash, []).append(entry)
        if entry.promotion_stage in PROMOTED_PROMOTION_STAGES:
            promoted_by_hash.setdefault(entry.spec_hash, []).append(entry)

    report: list[StrategyDecayItem] = []
    for spec_hash, promoted_entries in promoted_by_hash.items():
        history = sorted(entries_by_hash.get(spec_hash, ()), key=_entry_order_key)
        if not history:
            continue
        source = sorted(promoted_entries, key=_entry_order_key)[0]
        latest = history[-1]
        report.append(
            _decay_item(
                source,
                latest,
                evaluation_count=len(history),
                current_snapshot_id=current_snapshot_id,
                return_drop_warn_pct=return_drop_warn_pct,
                sharpe_drop_warn=sharpe_drop_warn,
                drawdown_increase_warn_pct=drawdown_increase_warn_pct,
                history=history,
            )
        )

    report.sort(
        key=lambda item: (
            _STATUS_RANK.get(item.status, 0),
            -item.return_delta_pct,
            -item.sharpe_delta,
            item.spec_hash,
        ),
        reverse=True,
    )
    return tuple(report[:limit])


def decay_report_to_payload(items: Iterable[StrategyDecayItem]) -> list[dict[str, object]]:
    return [item.to_payload() for item in items]


def _decay_item(
    source: LedgerEntry,
    latest: LedgerEntry,
    *,
    evaluation_count: int,
    current_snapshot_id: str | None,
    return_drop_warn_pct: float,
    sharpe_drop_warn: float,
    drawdown_increase_warn_pct: float,
    history: list[LedgerEntry],
) -> StrategyDecayItem:
    return_delta = latest.metric("return_pct") - source.metric("return_pct")
    sharpe_delta = latest.metric("annualized_sharpe", latest.metric("sharpe_like")) - source.metric(
        "annualized_sharpe",
        source.metric("sharpe_like"),
    )
    drawdown_delta = latest.metric("max_drawdown_pct") - source.metric("max_drawdown_pct")
    previous = history[-2] if len(history) >= 2 else source
    rolling_return_delta = latest.metric("return_pct") - previous.metric("return_pct")
    rolling_sharpe_delta = latest.metric("annualized_sharpe", latest.metric("sharpe_like")) - previous.metric(
        "annualized_sharpe",
        previous.metric("sharpe_like"),
    )
    rolling_drawdown_delta = latest.metric("max_drawdown_pct") - previous.metric("max_drawdown_pct")
    rolling_degradation_count = _rolling_degradation_count(
        history,
        return_drop_warn_pct=return_drop_warn_pct,
        sharpe_drop_warn=sharpe_drop_warn,
        drawdown_increase_warn_pct=drawdown_increase_warn_pct,
    )
    status = _decay_status(
        latest,
        current_snapshot_id=current_snapshot_id,
        return_delta_pct=return_delta,
        sharpe_delta=sharpe_delta,
        drawdown_delta_pct=drawdown_delta,
        rolling_degradation_count=rolling_degradation_count,
        return_drop_warn_pct=return_drop_warn_pct,
        sharpe_drop_warn=sharpe_drop_warn,
        drawdown_increase_warn_pct=drawdown_increase_warn_pct,
    )
    return StrategyDecayItem(
        spec_hash=source.spec_hash,
        source_experiment_id=source.experiment_id,
        latest_experiment_id=latest.experiment_id,
        source_data_snapshot_id=source.data_snapshot_id,
        latest_data_snapshot_id=latest.data_snapshot_id,
        source_promotion_stage=source.promotion_stage,
        latest_promotion_stage=latest.promotion_stage,
        evaluation_count=evaluation_count,
        return_delta_pct=return_delta,
        sharpe_delta=sharpe_delta,
        drawdown_delta_pct=drawdown_delta,
        rolling_return_delta_pct=rolling_return_delta,
        rolling_sharpe_delta=rolling_sharpe_delta,
        rolling_drawdown_delta_pct=rolling_drawdown_delta,
        rolling_degradation_count=rolling_degradation_count,
        status=status,
    )


def _decay_status(
    latest: LedgerEntry,
    *,
    current_snapshot_id: str | None,
    return_delta_pct: float,
    sharpe_delta: float,
    drawdown_delta_pct: float,
    rolling_degradation_count: int,
    return_drop_warn_pct: float,
    sharpe_drop_warn: float,
    drawdown_increase_warn_pct: float,
) -> str:
    if current_snapshot_id is not None and latest.data_snapshot_id != current_snapshot_id:
        return "pending_recheck"
    if latest.promotion_stage not in PROMOTED_PROMOTION_STAGES:
        return "demoted"
    if (
        return_delta_pct <= -abs(return_drop_warn_pct)
        or sharpe_delta <= -abs(sharpe_drop_warn)
        or drawdown_delta_pct >= abs(drawdown_increase_warn_pct)
        or rolling_degradation_count > 0
    ):
        return "degrading"
    return "stable"


def _rolling_degradation_count(
    history: list[LedgerEntry],
    *,
    return_drop_warn_pct: float,
    sharpe_drop_warn: float,
    drawdown_increase_warn_pct: float,
) -> int:
    count = 0
    for previous, current in reversed(tuple(zip(history, history[1:]))):
        if not _is_degraded_step(
            previous,
            current,
            return_drop_warn_pct=return_drop_warn_pct,
            sharpe_drop_warn=sharpe_drop_warn,
            drawdown_increase_warn_pct=drawdown_increase_warn_pct,
        ):
            break
        count += 1
    return count


def _is_degraded_step(
    previous: LedgerEntry,
    current: LedgerEntry,
    *,
    return_drop_warn_pct: float,
    sharpe_drop_warn: float,
    drawdown_increase_warn_pct: float,
) -> bool:
    return (
        current.metric("return_pct") - previous.metric("return_pct") <= -abs(return_drop_warn_pct)
        or current.metric("annualized_sharpe", current.metric("sharpe_like"))
        - previous.metric("annualized_sharpe", previous.metric("sharpe_like"))
        <= -abs(sharpe_drop_warn)
        or current.metric("max_drawdown_pct") - previous.metric("max_drawdown_pct")
        >= abs(drawdown_increase_warn_pct)
    )


def _entry_order_key(entry: LedgerEntry) -> tuple[str, str]:
    return (entry.completed_at_utc or entry.updated_at_utc, entry.experiment_id)


_STATUS_RANK = {
    "demoted": 4,
    "degrading": 3,
    "pending_recheck": 2,
    "stable": 1,
}
