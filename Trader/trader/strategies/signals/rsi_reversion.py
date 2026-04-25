from __future__ import annotations

from typing import Sequence

from trader.data.models import MarketBar
from trader.features.pipeline import FeaturePipeline


DEFAULT_PARAMS = {
    "rsi_length": 14,
    "oversold_threshold": 30.0,
    "overbought_threshold": 70.0,
}


def normalize_params(params: dict[str, object]) -> dict[str, int | float]:
    merged = {**DEFAULT_PARAMS, **params}
    rsi_length = int(merged["rsi_length"])
    oversold_threshold = float(merged["oversold_threshold"])
    overbought_threshold = float(merged["overbought_threshold"])
    if rsi_length < 2:
        raise ValueError("rsi_reversion.rsi_length must be >= 2")
    if not 0.0 < oversold_threshold < 50.0:
        raise ValueError("rsi_reversion.oversold_threshold must be in (0, 50)")
    if not 50.0 < overbought_threshold < 100.0:
        raise ValueError("rsi_reversion.overbought_threshold must be in (50, 100)")
    if overbought_threshold <= oversold_threshold:
        raise ValueError("rsi_reversion.overbought_threshold must be greater than oversold_threshold")
    return {
        "rsi_length": rsi_length,
        "oversold_threshold": oversold_threshold,
        "overbought_threshold": overbought_threshold,
    }


def required_history(params: dict[str, int | float]) -> int:
    # Need rsi_length + 1 prices to compute the first RSI value
    return int(params["rsi_length"]) + 1


def generate_regime(
    history_bars: Sequence[MarketBar],
    test_bars: Sequence[MarketBar],
    params: dict[str, int | float],
) -> list[bool]:
    pipeline = FeaturePipeline.from_segments(history_bars, test_bars)
    rsi_values = pipeline.rsi_for_test(int(params["rsi_length"]))
    oversold = float(params["oversold_threshold"])
    overbought = float(params["overbought_threshold"])

    # Stateful: enter long when RSI dips below oversold, exit when RSI rises above overbought.
    # This models mean-reversion: buy the dip, sell the rip.
    regime = False
    output: list[bool] = []
    for rsi_val in rsi_values:
        if rsi_val is None:
            regime = False
        elif not regime and rsi_val < oversold:
            regime = True
        elif regime and rsi_val > overbought:
            regime = False
        output.append(regime)
    return output


def parameter_grid() -> tuple[dict[str, int | float], ...]:
    grid: list[dict[str, int | float]] = []
    for rsi_length in (7, 14, 21):
        for oversold in (20.0, 30.0):
            for overbought in (70.0, 80.0):
                grid.append(
                    {
                        "rsi_length": rsi_length,
                        "oversold_threshold": oversold,
                        "overbought_threshold": overbought,
                    }
                )
    return tuple(grid)


def neighbors(params: dict[str, int | float]) -> tuple[dict[str, int | float], ...]:
    rsi_length = int(params["rsi_length"])
    oversold = float(params["oversold_threshold"])
    overbought = float(params["overbought_threshold"])
    candidates = {
        (rsi_length - 2, oversold, overbought),
        (rsi_length + 2, oversold, overbought),
        (rsi_length, oversold - 5.0, overbought),
        (rsi_length, oversold + 5.0, overbought),
        (rsi_length, oversold, overbought - 5.0),
        (rsi_length, oversold, overbought + 5.0),
    }
    normalized: list[dict[str, int | float]] = []
    for candidate_length, candidate_oversold, candidate_overbought in sorted(candidates):
        try:
            normalized.append(
                normalize_params(
                    {
                        "rsi_length": candidate_length,
                        "oversold_threshold": candidate_oversold,
                        "overbought_threshold": candidate_overbought,
                    }
                )
            )
        except ValueError:
            continue
    return tuple(normalized)
