from __future__ import annotations

from trader.features.pipeline import FeaturePipeline
from trader.features.primitives import rolling_max_exclusive, rolling_min_exclusive
from trader.strategies.signals import breakout


def test_rolling_primitives_are_exclusive() -> None:
    values = [1.0, 2.0, 99.0, 3.0]
    assert rolling_max_exclusive(values, 2) == [None, None, 2.0, 99.0]
    assert rolling_min_exclusive(values, 2) == [None, None, 1.0, 2.0]


def test_breakout_signal_does_not_peek_forward(seeded_store) -> None:
    from trader.data.view import DataView

    data_view = DataView(seeded_store.database_path)
    bars = data_view.bars("SPY", 1, "minute", regular_session_only=False)
    history_bars = bars[:20]
    base_test_bars = bars[20:23]
    altered_test_bars = list(base_test_bars)
    altered_test_bars[-1] = altered_test_bars[-1].__class__(
        timestamp_ms=altered_test_bars[-1].timestamp_ms,
        timestamp_utc=altered_test_bars[-1].timestamp_utc,
        open=altered_test_bars[-1].open,
        high=altered_test_bars[-1].high,
        low=altered_test_bars[-1].low,
        close=altered_test_bars[-1].close + 50.0,
        volume=altered_test_bars[-1].volume,
    )
    params = breakout.normalize_params({"entry_window": 20, "exit_window": 10, "buffer_bps": 0.0})
    base_regime = breakout.generate_regime(
        history_bars,
        base_test_bars,
        params,
    )
    altered_regime = breakout.generate_regime(history_bars, tuple(altered_test_bars), params)
    assert len(base_regime) == len(base_test_bars)
    assert base_regime[0] == altered_regime[0]


def test_feature_pipeline_uses_history_for_test_only(seeded_store) -> None:
    from trader.data.view import DataView

    data_view = DataView(seeded_store.database_path)
    bars = data_view.bars("SPY", 1, "minute", regular_session_only=False)
    pipeline = FeaturePipeline.from_segments(bars[:10], bars[10:12])
    ema_values = pipeline.ema_for_test(3)
    assert len(ema_values) == 2
    assert ema_values[1] > ema_values[0]
