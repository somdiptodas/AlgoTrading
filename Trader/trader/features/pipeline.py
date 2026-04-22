from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from trader.data.models import MarketBar
from trader.features.primitives import ema, rolling_max_exclusive, rolling_min_exclusive


@dataclass(frozen=True)
class FeaturePipeline:
    history_bars: tuple[MarketBar, ...]
    test_bars: tuple[MarketBar, ...]

    @property
    def combined_bars(self) -> tuple[MarketBar, ...]:
        return self.history_bars + self.test_bars

    @property
    def history_count(self) -> int:
        return len(self.history_bars)

    def close_series(self) -> list[float]:
        return [bar.close for bar in self.combined_bars]

    def high_series(self) -> list[float]:
        return [bar.high for bar in self.combined_bars]

    def low_series(self) -> list[float]:
        return [bar.low for bar in self.combined_bars]

    def ema_for_test(self, length: int) -> list[float]:
        return ema(self.close_series(), length)[self.history_count :]

    def rolling_high_for_test(self, window: int) -> list[float | None]:
        return rolling_max_exclusive(self.high_series(), window)[self.history_count :]

    def rolling_low_for_test(self, window: int) -> list[float | None]:
        return rolling_min_exclusive(self.low_series(), window)[self.history_count :]

    def combined_index_for_test(self, test_index: int) -> int:
        return self.history_count + test_index

    @classmethod
    def from_segments(cls, history_bars: Sequence[MarketBar], test_bars: Sequence[MarketBar]) -> "FeaturePipeline":
        return cls(tuple(history_bars), tuple(test_bars))
