from __future__ import annotations

from typing import Sequence

from trader.data.models import MarketBar
from trader.features.pipeline import FeaturePipeline


DEFAULT_PARAMS = {
    "fast_length": 20,
    "slow_length": 80,
    "signal_buffer_bps": 0.0,
}


def normalize_params(params: dict[str, object]) -> dict[str, int | float]:
    merged = {**DEFAULT_PARAMS, **params}
    fast_length = int(merged["fast_length"])
    slow_length = int(merged["slow_length"])
    signal_buffer_bps = float(merged["signal_buffer_bps"])
    if fast_length < 2:
        raise ValueError("ema_cross.fast_length must be >= 2")
    if slow_length <= fast_length:
        raise ValueError("ema_cross.slow_length must be greater than fast_length")
    if signal_buffer_bps < 0:
        raise ValueError("ema_cross.signal_buffer_bps must be >= 0")
    return {
        "fast_length": fast_length,
        "slow_length": slow_length,
        "signal_buffer_bps": signal_buffer_bps,
    }


def required_history(params: dict[str, int | float]) -> int:
    return int(params["slow_length"])


def generate_regime(
    history_bars: Sequence[MarketBar],
    test_bars: Sequence[MarketBar],
    params: dict[str, int | float],
) -> list[bool]:
    pipeline = FeaturePipeline.from_segments(history_bars, test_bars)
    fast_ema = pipeline.ema_for_test(int(params["fast_length"]))
    slow_ema = pipeline.ema_for_test(int(params["slow_length"]))
    signal_buffer_bps = float(params["signal_buffer_bps"])
    output: list[bool] = []
    for index in range(len(test_bars)):
        combined_index = pipeline.combined_index_for_test(index)
        regime_is_long = fast_ema[index] > slow_ema[index] * (1.0 + signal_buffer_bps / 10_000.0)
        output.append(regime_is_long if combined_index + 1 >= int(params["slow_length"]) else False)
    return output


def parameter_grid() -> tuple[dict[str, int | float], ...]:
    grid: list[dict[str, int | float]] = []
    for fast_length in (8, 12, 20, 34):
        for slow_length in (34, 55, 80, 120):
            if slow_length <= fast_length:
                continue
            for buffer_bps in (0.0, 5.0, 10.0):
                grid.append(
                    {
                        "fast_length": fast_length,
                        "slow_length": slow_length,
                        "signal_buffer_bps": buffer_bps,
                    }
                )
    return tuple(grid)


def neighbors(params: dict[str, int | float]) -> tuple[dict[str, int | float], ...]:
    fast_length = int(params["fast_length"])
    slow_length = int(params["slow_length"])
    buffer_bps = float(params["signal_buffer_bps"])
    candidates = {
        (fast_length - 2, slow_length, buffer_bps),
        (fast_length + 2, slow_length, buffer_bps),
        (fast_length, slow_length - 5, buffer_bps),
        (fast_length, slow_length + 5, buffer_bps),
        (fast_length, slow_length, max(0.0, buffer_bps - 5.0)),
        (fast_length, slow_length, buffer_bps + 5.0),
    }
    normalized: list[dict[str, int | float]] = []
    for candidate_fast, candidate_slow, candidate_buffer in sorted(candidates):
        try:
            normalized.append(
                normalize_params(
                    {
                        "fast_length": candidate_fast,
                        "slow_length": candidate_slow,
                        "signal_buffer_bps": candidate_buffer,
                    }
                )
            )
        except ValueError:
            continue
    return tuple(normalized)
