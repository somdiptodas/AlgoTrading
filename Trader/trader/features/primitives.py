from __future__ import annotations

from typing import Sequence


def rsi(values: Sequence[float], length: int) -> list[float | None]:
    """Wilder's RSI. Returns None for the first `length` bars (insufficient history)."""
    if length < 2:
        raise ValueError("rsi length must be >= 2")
    n = len(values)
    if n <= length:
        return [None] * n

    def _to_rsi(avg_gain: float, avg_loss: float) -> float:
        if avg_loss == 0.0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    result: list[float | None] = [None] * length

    # Seed: simple average of first `length` up/down moves
    avg_gain = sum(max(values[i] - values[i - 1], 0.0) for i in range(1, length + 1)) / length
    avg_loss = sum(max(values[i - 1] - values[i], 0.0) for i in range(1, length + 1)) / length
    result.append(_to_rsi(avg_gain, avg_loss))

    # Wilder smoothing for subsequent bars
    for i in range(length + 1, n):
        gain = max(values[i] - values[i - 1], 0.0)
        loss = max(values[i - 1] - values[i], 0.0)
        avg_gain = (avg_gain * (length - 1) + gain) / length
        avg_loss = (avg_loss * (length - 1) + loss) / length
        result.append(_to_rsi(avg_gain, avg_loss))

    return result


def ema(values: Sequence[float], length: int) -> list[float]:
    if length < 1:
        raise ValueError("length must be >= 1")
    if not values:
        return []
    alpha = 2.0 / (length + 1)
    result: list[float] = []
    current: float | None = None
    for value in values:
        current = value if current is None else (value * alpha) + (current * (1.0 - alpha))
        result.append(current)
    return result


def rolling_max_exclusive(values: Sequence[float], window: int) -> list[float | None]:
    if window < 1:
        raise ValueError("window must be >= 1")
    result: list[float | None] = []
    for index in range(len(values)):
        if index < window:
            result.append(None)
        else:
            result.append(max(values[index - window : index]))
    return result


def rolling_min_exclusive(values: Sequence[float], window: int) -> list[float | None]:
    if window < 1:
        raise ValueError("window must be >= 1")
    result: list[float | None] = []
    for index in range(len(values)):
        if index < window:
            result.append(None)
        else:
            result.append(min(values[index - window : index]))
    return result
