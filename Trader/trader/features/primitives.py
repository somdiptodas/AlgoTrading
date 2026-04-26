from __future__ import annotations

from typing import Sequence

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def rsi(values: Sequence[float], length: int) -> list[float | None]:
    """Wilder's RSI. Returns None for the first `length` bars (insufficient history)."""
    if length < 2:
        raise ValueError("rsi length must be >= 2")
    n = len(values)
    if n <= length:
        return [None] * n
    array = np.asarray(values, dtype=np.float64)
    deltas = np.diff(array)
    gains = np.maximum(deltas, 0.0)
    losses = np.maximum(-deltas, 0.0)

    def _to_rsi(avg_gain: float, avg_loss: float) -> float:
        if avg_loss == 0.0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    # Seed: simple average of first `length` up/down moves
    avg_gain = float(gains[:length].mean())
    avg_loss = float(losses[:length].mean())
    rsi_values = np.empty(n - length, dtype=np.float64)
    rsi_values[0] = _to_rsi(avg_gain, avg_loss)

    # Wilder smoothing for subsequent bars
    for delta_index in range(length, n - 1):
        gain = float(gains[delta_index])
        loss = float(losses[delta_index])
        avg_gain = (avg_gain * (length - 1) + gain) / length
        avg_loss = (avg_loss * (length - 1) + loss) / length
        rsi_values[delta_index - length + 1] = _to_rsi(avg_gain, avg_loss)

    return [None] * length + rsi_values.tolist()


def ema(values: Sequence[float], length: int) -> list[float]:
    if length < 1:
        raise ValueError("length must be >= 1")
    n = len(values)
    if n == 0:
        return []
    if length == 1:
        return [float(value) for value in values]
    array = np.asarray(values, dtype=np.float64)
    alpha = 2.0 / (length + 1)
    beta = 1.0 - alpha
    result = np.empty(n, dtype=np.float64)
    result[0] = array[0]
    previous = float(array[0])
    chunk_size = 256
    for start in range(1, n, chunk_size):
        chunk = array[start : start + chunk_size]
        powers = beta ** np.arange(1, len(chunk) + 1, dtype=np.float64)
        weighted = np.cumsum(chunk / powers)
        output = powers * (previous + alpha * weighted)
        result[start : start + len(chunk)] = output
        previous = float(output[-1])
    return result.tolist()


def rolling_max_exclusive(values: Sequence[float], window: int) -> list[float | None]:
    if window < 1:
        raise ValueError("window must be >= 1")
    n = len(values)
    if n <= window:
        return [None] * n
    array = np.asarray(values, dtype=np.float64)
    maxima = sliding_window_view(array, window_shape=window)[:-1].max(axis=1)
    return [None] * window + maxima.tolist()


def rolling_min_exclusive(values: Sequence[float], window: int) -> list[float | None]:
    if window < 1:
        raise ValueError("window must be >= 1")
    n = len(values)
    if n <= window:
        return [None] * n
    array = np.asarray(values, dtype=np.float64)
    minima = sliding_window_view(array, window_shape=window)[:-1].min(axis=1)
    return [None] * window + minima.tolist()
