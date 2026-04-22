from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from trader.ledger.entry import LedgerEntry


@dataclass(frozen=True)
class RankedLedgerEntry:
    entry: LedgerEntry
    composite_score: float
    pareto_optimal: bool


class LedgerQueryHelper:
    def pareto_frontier(self, entries: Iterable[LedgerEntry]) -> tuple[LedgerEntry, ...]:
        completed = [entry for entry in entries if entry.status == "completed"]
        frontier = [entry for entry in completed if not any(self._dominates(other, entry) for other in completed if other != entry)]
        return tuple(sorted(frontier, key=self._sort_key, reverse=True))

    def rank_entries(self, entries: Iterable[LedgerEntry], *, limit: int = 10) -> tuple[RankedLedgerEntry, ...]:
        completed = [entry for entry in entries if entry.status == "completed"]
        frontier_ids = {entry.experiment_id for entry in self.pareto_frontier(completed)}
        ranked = [
            RankedLedgerEntry(
                entry=entry,
                composite_score=self._composite_score(entry),
                pareto_optimal=entry.experiment_id in frontier_ids,
            )
            for entry in completed
        ]
        ranked.sort(
            key=lambda item: (
                item.pareto_optimal,
                item.composite_score,
                item.entry.completed_at_utc or "",
                item.entry.experiment_id,
            ),
            reverse=True,
        )
        return tuple(ranked[:limit])

    def top_experiments(self, entries: Sequence[LedgerEntry], *, limit: int = 10) -> tuple[LedgerEntry, ...]:
        frontier = list(self.pareto_frontier(entries))
        frontier_ids = {entry.experiment_id for entry in frontier}
        remaining = [entry for entry in entries if entry.status == "completed" and entry.experiment_id not in frontier_ids]
        frontier.sort(key=self._sort_key, reverse=True)
        remaining.sort(key=self._sort_key, reverse=True)
        ordered = frontier + remaining
        return tuple(ordered[:limit])

    def _composite_score(self, entry: LedgerEntry) -> float:
        stage_rank = {"candidate": 3.0, "frontier": 2.0, "exploratory": 1.0}.get(entry.promotion_stage, 0.0)
        fold_consistency = 1.0 if bool(entry.robustness_checks.get("fold_consistency_pass")) else 0.0
        neighborhood = 1.0 if bool(entry.robustness_checks.get("neighborhood_pass")) else 0.0
        return (
            stage_rank * 1_000.0
            + fold_consistency * 100.0
            + neighborhood * 50.0
            + entry.metric("delta_buy_and_hold_return_pct") * 4.0
            + entry.metric("sharpe_like") * 20.0
            + entry.metric("return_pct")
            - entry.metric("max_drawdown_pct") * 0.5
            + min(entry.metric("trade_count"), 25.0)
        )

    def _dominates(self, left: LedgerEntry, right: LedgerEntry) -> bool:
        left_return = left.metric("return_pct")
        right_return = right.metric("return_pct")
        left_sharpe = left.metric("sharpe_like")
        right_sharpe = right.metric("sharpe_like")
        left_drawdown = left.metric("max_drawdown_pct")
        right_drawdown = right.metric("max_drawdown_pct")
        no_worse = (
            left_return >= right_return
            and left_sharpe >= right_sharpe
            and left_drawdown <= right_drawdown
        )
        strictly_better = (
            left_return > right_return
            or left_sharpe > right_sharpe
            or left_drawdown < right_drawdown
        )
        return no_worse and strictly_better

    def _sort_key(self, entry: LedgerEntry) -> tuple[float, float, float, float, str, str]:
        return (
            self._composite_score(entry),
            entry.metric("sharpe_like"),
            entry.metric("return_pct"),
            -entry.metric("max_drawdown_pct"),
            entry.completed_at_utc or "",
            entry.experiment_id,
        )
