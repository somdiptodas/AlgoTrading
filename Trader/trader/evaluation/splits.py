from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Sequence

from trader.data.models import MarketBar


@dataclass(frozen=True)
class Fold:
    fold_id: str
    train_start_idx: int
    train_end_idx: int
    test_start_idx: int
    test_end_idx: int
    embargo_bars: int
    train_start_utc: str
    train_end_utc: str
    test_start_utc: str
    test_end_utc: str


def build_walk_forward_folds(
    bars: Sequence[MarketBar],
    *,
    num_folds: int,
    min_train_bars: int,
    embargo_bars: int = 1,
) -> tuple[str, tuple[Fold, ...]]:
    if num_folds < 1:
        raise ValueError("num_folds must be >= 1")
    if min_train_bars < 1:
        raise ValueError("min_train_bars must be >= 1")
    if embargo_bars < 0:
        raise ValueError("embargo_bars must be >= 0")
    total_bars = len(bars)
    available_test_bars = total_bars - min_train_bars - embargo_bars
    if available_test_bars < num_folds:
        raise ValueError("Not enough bars to build the requested walk-forward folds")
    fold_test_size = available_test_bars // num_folds
    folds: list[Fold] = []
    for fold_index in range(num_folds):
        test_start_idx = min_train_bars + (fold_index * fold_test_size)
        if fold_index == num_folds - 1:
            test_end_idx = total_bars - 1
        else:
            test_end_idx = test_start_idx + fold_test_size - 1
        train_start_idx = 0
        train_end_idx = test_start_idx - embargo_bars - 1
        if train_end_idx < train_start_idx:
            raise ValueError("Embargo leaves no training data")
        fold = Fold(
            fold_id=f"fold_{fold_index + 1}",
            train_start_idx=train_start_idx,
            train_end_idx=train_end_idx,
            test_start_idx=test_start_idx,
            test_end_idx=test_end_idx,
            embargo_bars=embargo_bars,
            train_start_utc=bars[train_start_idx].timestamp_utc,
            train_end_utc=bars[train_end_idx].timestamp_utc,
            test_start_utc=bars[test_start_idx].timestamp_utc,
            test_end_utc=bars[test_end_idx].timestamp_utc,
        )
        folds.append(fold)
    validate_folds(folds)
    plan_payload = "|".join(
        [
            "walk_forward",
            str(num_folds),
            str(min_train_bars),
            str(embargo_bars),
            str(total_bars),
            bars[0].timestamp_utc,
            bars[-1].timestamp_utc,
        ]
    )
    return sha256(plan_payload.encode("utf-8")).hexdigest(), tuple(folds)


def validate_folds(folds: Sequence[Fold]) -> None:
    previous_test_end = -1
    for fold in folds:
        if fold.train_end_idx >= fold.test_start_idx:
            raise ValueError(f"{fold.fold_id} train/test ranges overlap")
        if fold.test_start_idx - fold.train_end_idx - 1 < fold.embargo_bars:
            raise ValueError(f"{fold.fold_id} embargo violation")
        if fold.test_start_idx <= previous_test_end:
            raise ValueError("Test folds overlap or are out of order")
        previous_test_end = fold.test_end_idx
