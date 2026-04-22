from __future__ import annotations

from typing import Sequence

from trader.data.models import MarketBar
from trader.features.pipeline import FeaturePipeline


DEFAULT_PARAMS = {
    "entry_window": 20,
    "exit_window": 10,
    "buffer_bps": 0.0,
}


def normalize_params(params: dict[str, object]) -> dict[str, int | float]:
    merged = {**DEFAULT_PARAMS, **params}
    entry_window = int(merged["entry_window"])
    exit_window = int(merged["exit_window"])
    buffer_bps = float(merged["buffer_bps"])
    if entry_window < 2:
        raise ValueError("breakout.entry_window must be >= 2")
    if exit_window < 2:
        raise ValueError("breakout.exit_window must be >= 2")
    if exit_window > entry_window:
        raise ValueError("breakout.exit_window must be <= entry_window")
    if buffer_bps < 0:
        raise ValueError("breakout.buffer_bps must be >= 0")
    return {
        "entry_window": entry_window,
        "exit_window": exit_window,
        "buffer_bps": buffer_bps,
    }


def required_history(params: dict[str, int | float]) -> int:
    return max(int(params["entry_window"]), int(params["exit_window"]))


def generate_regime(
    history_bars: Sequence[MarketBar],
    test_bars: Sequence[MarketBar],
    params: dict[str, int | float],
) -> list[bool]:
    pipeline = FeaturePipeline.from_segments(history_bars, test_bars)
    highs = pipeline.rolling_high_for_test(int(params["entry_window"]))
    lows = pipeline.rolling_low_for_test(int(params["exit_window"]))
    buffer_bps = float(params["buffer_bps"])
    regime = False
    output: list[bool] = []
    for index, bar in enumerate(test_bars):
        prev_high = highs[index]
        prev_low = lows[index]
        if prev_high is None or prev_low is None:
            regime = False
        elif not regime and bar.close > prev_high * (1.0 + buffer_bps / 10_000.0):
            regime = True
        elif regime and bar.close < prev_low * (1.0 - buffer_bps / 10_000.0):
            regime = False
        output.append(regime)
    return output


def parameter_grid() -> tuple[dict[str, int | float], ...]:
    grid: list[dict[str, int | float]] = []
    for entry_window in (10, 20, 40, 60):
        for exit_window in (5, 10, 20):
            if exit_window > entry_window:
                continue
            for buffer_bps in (0.0, 5.0):
                grid.append(
                    {
                        "entry_window": entry_window,
                        "exit_window": exit_window,
                        "buffer_bps": buffer_bps,
                    }
                )
    return tuple(grid)


def neighbors(params: dict[str, int | float]) -> tuple[dict[str, int | float], ...]:
    entry_window = int(params["entry_window"])
    exit_window = int(params["exit_window"])
    buffer_bps = float(params["buffer_bps"])
    candidates = {
        (entry_window - 5, exit_window, buffer_bps),
        (entry_window + 5, exit_window, buffer_bps),
        (entry_window, exit_window - 2, buffer_bps),
        (entry_window, exit_window + 2, buffer_bps),
        (entry_window, exit_window, max(0.0, buffer_bps - 5.0)),
        (entry_window, exit_window, buffer_bps + 5.0),
    }
    normalized: list[dict[str, int | float]] = []
    for candidate_entry, candidate_exit, candidate_buffer in sorted(candidates):
        try:
            normalized.append(
                normalize_params(
                    {
                        "entry_window": candidate_entry,
                        "exit_window": candidate_exit,
                        "buffer_bps": candidate_buffer,
                    }
                )
            )
        except ValueError:
            continue
    return tuple(normalized)
