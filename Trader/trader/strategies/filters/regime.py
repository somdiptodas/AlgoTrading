from __future__ import annotations

import math
from collections import OrderedDict
from typing import Sequence

import numpy as np

from trader.data.models import MarketBar


def normalize_intraday_volatility_params(params: dict[str, object]) -> dict[str, int | float]:
    merged = {"lookback_bars": 20, "percentile_window": 120, "min_percentile": 50.0, "max_percentile": 100.0, **params}
    lookback_bars = int(merged["lookback_bars"])
    percentile_window = int(merged["percentile_window"])
    min_percentile = float(merged["min_percentile"])
    max_percentile = float(merged["max_percentile"])
    if lookback_bars < 2:
        raise ValueError("intraday_volatility.lookback_bars must be >= 2")
    if percentile_window < 1:
        raise ValueError("intraday_volatility.percentile_window must be >= 1")
    _validate_percentile_bounds("intraday_volatility", min_percentile, max_percentile)
    return {
        "lookback_bars": lookback_bars,
        "percentile_window": percentile_window,
        "min_percentile": min_percentile,
        "max_percentile": max_percentile,
    }


def generate_intraday_volatility_mask(
    history_bars: Sequence[MarketBar],
    test_bars: Sequence[MarketBar],
    params: dict[str, int | float],
) -> list[bool]:
    bars = tuple(history_bars) + tuple(test_bars)
    history_count = len(history_bars)
    lookback_bars = int(params["lookback_bars"])
    percentile_window = int(params["percentile_window"])
    realized = _intraday_realized_volatility_bps(bars, lookback_bars)
    output: list[bool] = []
    for index in range(history_count, len(bars)):
        value = realized[index]
        sample = [
            item for item in realized[max(0, index - percentile_window) : index]
            if item is not None
        ]
        percentile = _percentile_rank(value, sample)
        output.append(
            percentile is not None
            and float(params["min_percentile"]) <= percentile <= float(params["max_percentile"])
        )
    return output


def normalize_prior_day_range_params(params: dict[str, object]) -> dict[str, float]:
    merged = {"min_range_bps": 50.0, "max_range_bps": 100_000.0, **params}
    min_range_bps = float(merged["min_range_bps"])
    max_range_bps = float(merged["max_range_bps"])
    if min_range_bps < 0:
        raise ValueError("prior_day_range.min_range_bps must be >= 0")
    if max_range_bps < min_range_bps:
        raise ValueError("prior_day_range.max_range_bps must be >= min_range_bps")
    return {"min_range_bps": min_range_bps, "max_range_bps": max_range_bps}


def generate_prior_day_range_mask(
    history_bars: Sequence[MarketBar],
    test_bars: Sequence[MarketBar],
    params: dict[str, float],
) -> list[bool]:
    bars = tuple(history_bars) + tuple(test_bars)
    ranges = _prior_session_range_bps(bars)
    return [
        (range_bps := ranges.get(bar.session_date)) is not None
        and params["min_range_bps"] <= range_bps <= params["max_range_bps"]
        for bar in test_bars
    ]


def normalize_relative_volume_params(params: dict[str, object]) -> dict[str, int | float]:
    merged = {"lookback_bars": 20, "min_ratio": 1.25, "max_ratio": 100.0, **params}
    lookback_bars = int(merged["lookback_bars"])
    min_ratio = float(merged["min_ratio"])
    max_ratio = float(merged["max_ratio"])
    if lookback_bars < 1:
        raise ValueError("relative_volume.lookback_bars must be >= 1")
    if min_ratio < 0:
        raise ValueError("relative_volume.min_ratio must be >= 0")
    if max_ratio < min_ratio:
        raise ValueError("relative_volume.max_ratio must be >= min_ratio")
    return {"lookback_bars": lookback_bars, "min_ratio": min_ratio, "max_ratio": max_ratio}


def generate_relative_volume_mask(
    history_bars: Sequence[MarketBar],
    test_bars: Sequence[MarketBar],
    params: dict[str, int | float],
) -> list[bool]:
    bars = tuple(history_bars) + tuple(test_bars)
    history_count = len(history_bars)
    lookback_bars = int(params["lookback_bars"])
    output: list[bool] = []
    for index in range(history_count, len(bars)):
        previous = [
            bar.volume for bar in bars[max(0, index - lookback_bars) : index]
            if bar.session_date == bars[index].session_date
        ]
        average_volume = sum(previous) / len(previous) if previous else None
        ratio = bars[index].volume / average_volume if average_volume and average_volume > 0 else None
        output.append(ratio is not None and float(params["min_ratio"]) <= ratio <= float(params["max_ratio"]))
    return output


def normalize_day_type_params(params: dict[str, object]) -> dict[str, int | float | str]:
    merged = {
        "mode": "trend",
        "min_bars": 30,
        "trend_bps": 50.0,
        "min_efficiency": 0.60,
        "max_efficiency": 0.35,
        **params,
    }
    mode = str(merged["mode"])
    min_bars = int(merged["min_bars"])
    trend_bps = float(merged["trend_bps"])
    min_efficiency = float(merged["min_efficiency"])
    max_efficiency = float(merged["max_efficiency"])
    if mode not in {"trend", "mean_reversion"}:
        raise ValueError("day_type.mode must be trend or mean_reversion")
    if min_bars < 1:
        raise ValueError("day_type.min_bars must be >= 1")
    if trend_bps < 0:
        raise ValueError("day_type.trend_bps must be >= 0")
    if not 0.0 <= max_efficiency <= min_efficiency <= 1.0:
        raise ValueError("day_type efficiency bounds must satisfy 0 <= max_efficiency <= min_efficiency <= 1")
    return {
        "mode": mode,
        "min_bars": min_bars,
        "trend_bps": trend_bps,
        "min_efficiency": min_efficiency,
        "max_efficiency": max_efficiency,
    }


def generate_day_type_mask(
    history_bars: Sequence[MarketBar],
    test_bars: Sequence[MarketBar],
    params: dict[str, int | float | str],
) -> list[bool]:
    bars = tuple(history_bars) + tuple(test_bars)
    history_count = len(history_bars)
    stats = _session_progress_stats(bars)
    output: list[bool] = []
    for index in range(history_count, len(bars)):
        count, move_bps, range_bps, efficiency = stats[index]
        if count < int(params["min_bars"]):
            output.append(False)
        elif params["mode"] == "trend":
            output.append(move_bps >= float(params["trend_bps"]) and efficiency >= float(params["min_efficiency"]))
        else:
            output.append(range_bps >= float(params["trend_bps"]) and efficiency <= float(params["max_efficiency"]))
    return output


def normalize_vwap_distance_params(params: dict[str, object]) -> dict[str, float | str]:
    merged = {"side": "below", "min_deviation_bps": 0.0, "max_deviation_bps": 100_000.0, **params}
    side = str(merged["side"])
    min_deviation_bps = float(merged["min_deviation_bps"])
    max_deviation_bps = float(merged["max_deviation_bps"])
    if side not in {"below", "above"}:
        raise ValueError("vwap_distance.side must be below or above")
    if min_deviation_bps < 0:
        raise ValueError("vwap_distance.min_deviation_bps must be >= 0")
    if max_deviation_bps < min_deviation_bps:
        raise ValueError("vwap_distance.max_deviation_bps must be >= min_deviation_bps")
    return {
        "side": side,
        "min_deviation_bps": min_deviation_bps,
        "max_deviation_bps": max_deviation_bps,
    }


def generate_vwap_distance_mask(
    history_bars: Sequence[MarketBar],
    test_bars: Sequence[MarketBar],
    params: dict[str, float | str],
) -> list[bool]:
    output: list[bool] = []
    side = str(params["side"])
    min_deviation_bps = float(params["min_deviation_bps"])
    max_deviation_bps = float(params["max_deviation_bps"])
    for bar in test_bars:
        if bar.vwap is None or bar.vwap <= 0:
            output.append(False)
            continue
        if side == "below":
            deviation_bps = ((bar.vwap - bar.close) / bar.vwap) * 10_000.0
        else:
            deviation_bps = ((bar.close - bar.vwap) / bar.vwap) * 10_000.0
        output.append(min_deviation_bps <= deviation_bps <= max_deviation_bps)
    return output


def _intraday_realized_volatility_bps(bars: Sequence[MarketBar], lookback_bars: int) -> list[float | None]:
    n = len(bars)
    if n == 0:
        return []
    closes = np.fromiter((bar.close for bar in bars), dtype=np.float64, count=n)
    sessions = np.fromiter((bar.session_date for bar in bars), dtype=object, count=n)
    valid_returns = np.zeros(n, dtype=bool)
    if n > 1:
        valid_returns[1:] = (sessions[1:] == sessions[:-1]) & (closes[:-1] > 0.0)
    squared_returns = np.zeros(n, dtype=np.float64)
    if valid_returns.any():
        valid_indices = np.flatnonzero(valid_returns)
        returns = ((closes[valid_indices] / closes[valid_indices - 1]) - 1.0) * 10_000.0
        squared_returns[valid_indices] = returns * returns
    kernel = np.ones(lookback_bars, dtype=np.float64)
    rolling_counts = np.convolve(valid_returns.astype(np.float64), kernel, mode="full")[:n]
    rolling_sums = np.convolve(squared_returns, kernel, mode="full")[:n]
    realized = np.sqrt(rolling_sums / lookback_bars)
    return [
        float(realized[index]) if rolling_counts[index] == lookback_bars else None
        for index in range(n)
    ]


def _prior_session_range_bps(bars: Sequence[MarketBar]) -> dict[str, float]:
    sessions: OrderedDict[str, list[MarketBar]] = OrderedDict()
    for bar in bars:
        sessions.setdefault(bar.session_date, []).append(bar)
    prior_by_session: dict[str, float] = {}
    previous_range: float | None = None
    for session_date, session_bars in sessions.items():
        prior_by_session[session_date] = previous_range if previous_range is not None else math.nan
        last_close = session_bars[-1].close
        previous_range = (
            ((max(bar.high for bar in session_bars) - min(bar.low for bar in session_bars)) / last_close) * 10_000.0
            if last_close > 0
            else None
        )
    return {key: value for key, value in prior_by_session.items() if value is not None and math.isfinite(value)}


def _session_progress_stats(bars: Sequence[MarketBar]) -> list[tuple[int, float, float, float]]:
    n = len(bars)
    if n == 0:
        return []
    opens = np.fromiter((bar.open for bar in bars), dtype=np.float64, count=n)
    highs = np.fromiter((bar.high for bar in bars), dtype=np.float64, count=n)
    lows = np.fromiter((bar.low for bar in bars), dtype=np.float64, count=n)
    closes = np.fromiter((bar.close for bar in bars), dtype=np.float64, count=n)
    sessions = np.fromiter((bar.session_date for bar in bars), dtype=object, count=n)
    starts = np.r_[0, np.flatnonzero(sessions[1:] != sessions[:-1]) + 1]
    ends = np.r_[starts[1:], n]

    counts = np.empty(n, dtype=np.int64)
    move_bps = np.empty(n, dtype=np.float64)
    range_bps = np.empty(n, dtype=np.float64)
    efficiency = np.empty(n, dtype=np.float64)
    for start, end in zip(starts, ends):
        session_open = opens[start]
        session_highs = np.maximum.accumulate(highs[start:end])
        session_lows = np.minimum.accumulate(lows[start:end])
        ranges = session_highs - session_lows
        moves = np.abs(closes[start:end] - session_open)
        counts[start:end] = np.arange(1, end - start + 1)
        if session_open > 0.0:
            move_bps[start:end] = (moves / session_open) * 10_000.0
            range_bps[start:end] = (ranges / session_open) * 10_000.0
        else:
            move_bps[start:end] = 0.0
            range_bps[start:end] = 0.0
        efficiency[start:end] = np.divide(
            moves,
            ranges,
            out=np.zeros_like(moves),
            where=ranges > 0.0,
        )
    return [
        (int(counts[index]), float(move_bps[index]), float(range_bps[index]), float(efficiency[index]))
        for index in range(n)
    ]


def _percentile_rank(value: float | None, sample: Sequence[float]) -> float | None:
    if value is None or not sample:
        return None
    return 100.0 * sum(item <= value for item in sample) / len(sample)


def _validate_percentile_bounds(scope: str, minimum: float, maximum: float) -> None:
    if not 0.0 <= minimum <= maximum <= 100.0:
        raise ValueError(f"{scope} percentile bounds must satisfy 0 <= min_percentile <= max_percentile <= 100")
