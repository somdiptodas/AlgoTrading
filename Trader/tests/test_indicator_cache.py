from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from trader.data.models import MarketBar
from trader.evaluation.runner import EvaluationRunner
from trader.evaluation.splits import Fold
from trader.features.pipeline import FeaturePipeline, feature_cache_context
from trader.strategies.registry import REGISTRY
from trader.strategies.spec import SignalSpec, StrategySpec


NEW_YORK = ZoneInfo("America/New_York")


def _bars(count: int) -> tuple[MarketBar, ...]:
    bars = []
    start = datetime(2026, 1, 5, 9, 30, tzinfo=NEW_YORK)
    for index in range(count):
        timestamp = (start + timedelta(minutes=index)).astimezone(timezone.utc)
        price = 100.0 + index
        bars.append(
            MarketBar(
                timestamp_ms=int(timestamp.timestamp() * 1000),
                timestamp_utc=timestamp.isoformat(),
                open=price,
                high=price + 0.5,
                low=price - 0.5,
                close=price,
                volume=1_000.0 + index,
            )
        )
    return tuple(bars)


def _fold(bars: tuple[MarketBar, ...]) -> Fold:
    return Fold(
        fold_id="fold_1",
        train_start_idx=0,
        train_end_idx=4,
        test_start_idx=5,
        test_end_idx=len(bars) - 1,
        embargo_bars=0,
        train_start_utc=bars[0].timestamp_utc,
        train_end_utc=bars[4].timestamp_utc,
        test_start_utc=bars[5].timestamp_utc,
        test_end_utc=bars[-1].timestamp_utc,
    )


def test_feature_pipeline_caches_indicator_slices_by_scope_and_params(monkeypatch) -> None:
    bars = _bars(8)
    cache = {}
    calls = Counter()

    def counted(name: str):
        def primitive(values, param):
            calls[(name, param)] += 1
            return [float(param)] * len(values)

        return primitive

    monkeypatch.setattr("trader.features.pipeline.ema", counted("ema"))
    monkeypatch.setattr("trader.features.pipeline.rsi", counted("rsi"))
    monkeypatch.setattr("trader.features.pipeline.rolling_max_exclusive", counted("rolling_high"))
    monkeypatch.setattr("trader.features.pipeline.rolling_min_exclusive", counted("rolling_low"))

    with feature_cache_context(cache, ("snapshot_a", "fold_1")):
        first = FeaturePipeline.from_segments(bars[:4], bars[4:])
        first.ema_for_test(3)[0] = 999.0
        first.rsi_for_test(3)
        first.rolling_high_for_test(2)
        first.rolling_low_for_test(2)

        second = FeaturePipeline.from_segments(bars[:4], bars[4:])
        assert second.ema_for_test(3)[0] == 3.0
        second.rsi_for_test(3)
        second.rolling_high_for_test(2)
        second.rolling_low_for_test(2)

    assert calls == {
        ("ema", 3): 1,
        ("rsi", 3): 1,
        ("rolling_high", 2): 1,
        ("rolling_low", 2): 1,
    }


def test_feature_pipeline_cache_is_scoped_by_snapshot(monkeypatch) -> None:
    bars = _bars(8)
    cache = {}
    calls = Counter()

    def counted_ema(values, length):
        calls[length] += 1
        return [float(length)] * len(values)

    monkeypatch.setattr("trader.features.pipeline.ema", counted_ema)

    with feature_cache_context(cache, ("snapshot_a", "fold_1")):
        FeaturePipeline.from_segments(bars[:4], bars[4:]).ema_for_test(3)
    with feature_cache_context(cache, ("snapshot_b", "fold_1")):
        FeaturePipeline.from_segments(bars[:4], bars[4:]).ema_for_test(3)

    assert calls[3] == 2


def test_runner_reuses_ema_by_snapshot_fold_and_length(monkeypatch) -> None:
    bars = _bars(10)
    fold = _fold(bars)
    runner = EvaluationRunner.__new__(EvaluationRunner)
    runner.registry = REGISTRY
    runner._fold_result_cache = {}
    runner._baseline_cache = {}
    runner._indicator_cache = {}
    calls = Counter()

    def counted_ema(values, length):
        calls[length] += 1
        return [float(length)] * len(values)

    monkeypatch.setattr("trader.features.pipeline.ema", counted_ema)

    runner._evaluate_fold(
        StrategySpec(
            name="ema_one",
            signal=SignalSpec("ema_cross", {"fast_length": 2, "slow_length": 5, "signal_buffer_bps": 0.0}),
        ),
        fold,
        bars,
        data_snapshot_id="snapshot_a",
    )
    runner._evaluate_fold(
        StrategySpec(
            name="ema_two",
            signal=SignalSpec("ema_cross", {"fast_length": 2, "slow_length": 6, "signal_buffer_bps": 0.0}),
        ),
        fold,
        bars,
        data_snapshot_id="snapshot_a",
    )

    assert calls == {2: 1, 5: 1, 6: 1}
