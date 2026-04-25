from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Callable, Iterator, Sequence

from trader.data.models import MarketBar
from trader.features.primitives import ema, rolling_max_exclusive, rolling_min_exclusive, rsi

FeatureCache = dict[tuple[object, ...], tuple[object, ...]]


@dataclass(frozen=True)
class FeatureCacheContext:
    cache: FeatureCache
    scope: tuple[object, ...]


_ACTIVE_CACHE: ContextVar[FeatureCacheContext | None] = ContextVar("feature_pipeline_cache", default=None)


@contextmanager
def feature_cache_context(cache: FeatureCache, scope: tuple[object, ...]) -> Iterator[None]:
    token = _ACTIVE_CACHE.set(FeatureCacheContext(cache=cache, scope=scope))
    try:
        yield
    finally:
        _ACTIVE_CACHE.reset(token)


@dataclass(frozen=True)
class FeaturePipeline:
    history_bars: tuple[MarketBar, ...]
    test_bars: tuple[MarketBar, ...]
    cache_context: FeatureCacheContext | None = None

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
        return self._cached_for_test("ema", length, lambda: ema(self.close_series(), length))

    def rolling_high_for_test(self, window: int) -> list[float | None]:
        return self._cached_for_test(
            "rolling_high",
            window,
            lambda: rolling_max_exclusive(self.high_series(), window),
        )

    def rolling_low_for_test(self, window: int) -> list[float | None]:
        return self._cached_for_test(
            "rolling_low",
            window,
            lambda: rolling_min_exclusive(self.low_series(), window),
        )

    def rsi_for_test(self, length: int) -> list[float | None]:
        return self._cached_for_test("rsi", length, lambda: rsi(self.close_series(), length))

    def combined_index_for_test(self, test_index: int) -> int:
        return self.history_count + test_index

    def _cached_for_test(
        self,
        name: str,
        param: int,
        compute: Callable[[], list[object]],
    ) -> list[object]:
        cache_context = self.cache_context
        if cache_context is None:
            return compute()[self.history_count :]
        cache_key = (*cache_context.scope, name, param)
        cached = cache_context.cache.get(cache_key)
        if cached is None:
            cached = tuple(compute()[self.history_count :])
            cache_context.cache[cache_key] = cached
        return list(cached)

    @classmethod
    def from_segments(cls, history_bars: Sequence[MarketBar], test_bars: Sequence[MarketBar]) -> "FeaturePipeline":
        return cls(tuple(history_bars), tuple(test_bars), cache_context=_ACTIVE_CACHE.get())
