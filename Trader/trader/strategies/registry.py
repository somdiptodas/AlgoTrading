from __future__ import annotations

import math
from dataclasses import replace
from typing import Callable

from trader.data.models import MarketBar
from trader.strategies.filters import regime, session
from trader.strategies.signals import breakout, ema_cross, rsi_reversion, vwap_deviation
from trader.strategies.sizers import fixed_fraction, full_notional
from trader.strategies.spec import FilterSpec, SignalSpec, StrategySpec


SignalGenerator = Callable[[tuple[MarketBar, ...], tuple[MarketBar, ...], dict[str, int | float]], list[bool]]


class StrategyRegistry:
    def __init__(self) -> None:
        self.signal_handlers = {
            "ema_cross": {
                "normalize_params": ema_cross.normalize_params,
                "required_history": ema_cross.required_history,
                "generate_regime": ema_cross.generate_regime,
                "parameter_grid": ema_cross.parameter_grid,
                "neighbors": ema_cross.neighbors,
            },
            "breakout": {
                "normalize_params": breakout.normalize_params,
                "required_history": breakout.required_history,
                "generate_regime": breakout.generate_regime,
                "parameter_grid": breakout.parameter_grid,
                "neighbors": breakout.neighbors,
            },
            "rsi_reversion": {
                "normalize_params": rsi_reversion.normalize_params,
                "required_history": rsi_reversion.required_history,
                "generate_regime": rsi_reversion.generate_regime,
                "parameter_grid": rsi_reversion.parameter_grid,
                "neighbors": rsi_reversion.neighbors,
            },
            "vwap_deviation": {
                "normalize_params": vwap_deviation.normalize_params,
                "required_history": vwap_deviation.required_history,
                "generate_regime": vwap_deviation.generate_regime,
                "parameter_grid": vwap_deviation.parameter_grid,
                "neighbors": vwap_deviation.neighbors,
            },
        }
        self.sizing_handlers = {
            "full_notional": {
                "normalize_params": full_notional.normalize_params,
                "compute_fraction": full_notional.compute_fraction,
            },
            "fixed_fraction": {
                "normalize_params": fixed_fraction.normalize_params,
                "compute_fraction": fixed_fraction.compute_fraction,
            },
        }
        self.filter_handlers = {
            "session": {
                "normalize_params": session.normalize_params,
                "required_history": lambda params: 0,
                "generate_mask": session.generate_mask,
            },
            "intraday_volatility": {
                "normalize_params": regime.normalize_intraday_volatility_params,
                "required_history": lambda params: int(params["lookback_bars"]) + int(params["percentile_window"]),
                "generate_mask": regime.generate_intraday_volatility_mask,
            },
            "prior_day_range": {
                "normalize_params": regime.normalize_prior_day_range_params,
                "required_history": lambda params: 390,
                "generate_mask": regime.generate_prior_day_range_mask,
            },
            "relative_volume": {
                "normalize_params": regime.normalize_relative_volume_params,
                "required_history": lambda params: int(params["lookback_bars"]),
                "generate_mask": regime.generate_relative_volume_mask,
            },
            "day_type": {
                "normalize_params": regime.normalize_day_type_params,
                "required_history": lambda params: int(params["min_bars"]),
                "generate_mask": regime.generate_day_type_mask,
            },
        }

    def validate_spec(self, spec: StrategySpec) -> StrategySpec:
        if spec.instrument != "SPY":
            raise ValueError("Only SPY is supported in v1")
        if spec.multiplier != 1 or spec.timespan != "minute":
            raise ValueError("Only SPY 1-minute bars are supported in v1")
        self._validate_exec_config(spec)
        if spec.exec_config.initial_cash <= 0:
            raise ValueError("exec_config.initial_cash must be > 0")
        if spec.exec_config.commission_per_order < 0:
            raise ValueError("exec_config.commission_per_order must be >= 0")
        if spec.exec_config.commission_per_share < 0:
            raise ValueError("exec_config.commission_per_share must be >= 0")
        if spec.exec_config.slippage_bps < 0:
            raise ValueError("exec_config.slippage_bps must be >= 0")
        if spec.exec_config.spread_bps < 0:
            raise ValueError("exec_config.spread_bps must be >= 0")
        if spec.exec_config.max_position_notional is not None and spec.exec_config.max_position_notional <= 0:
            raise ValueError("exec_config.max_position_notional must be > 0 when set")
        if spec.exec_config.stop_loss_bps is not None and spec.exec_config.stop_loss_bps <= 0:
            raise ValueError("exec_config.stop_loss_bps must be > 0 when set")
        if spec.exec_config.entry_session_window not in {"all", "first_30m", "last_30m", "avoid_midday"}:
            raise ValueError("exec_config.entry_session_window must be all, first_30m, last_30m, or avoid_midday")
        if (
            spec.exec_config.no_new_entry_minutes_before_close is not None
            and spec.exec_config.no_new_entry_minutes_before_close < 0
        ):
            raise ValueError("exec_config.no_new_entry_minutes_before_close must be >= 0 when set")
        if (
            spec.exec_config.no_new_entry_minutes_before_close is not None
            and spec.exec_config.no_new_entry_minutes_before_close > 390
        ):
            raise ValueError("exec_config.no_new_entry_minutes_before_close must be <= 390 when set")
        if spec.signal.name not in self.signal_handlers:
            raise ValueError(f"Unknown signal handler: {spec.signal.name}")
        if spec.sizing.name not in self.sizing_handlers:
            raise ValueError(f"Unknown sizing handler: {spec.sizing.name}")
        normalized_signal = self.signal_handlers[spec.signal.name]["normalize_params"](spec.signal.params)
        normalized_sizing = self.sizing_handlers[spec.sizing.name]["normalize_params"](spec.sizing.params)
        self._validate_finite_params(spec.signal.name, normalized_signal)
        self._validate_finite_params(spec.sizing.name, normalized_sizing)
        normalized_filters = tuple(
            filter_spec for filter_spec in (self._validate_filter(filter_spec) for filter_spec in spec.filters)
            if not (
                filter_spec.name == "session"
                and spec.exec_config.regular_session_only
                and filter_spec.params.get("session") == "regular"
            )
        )
        normalized = replace(
            spec,
            signal=SignalSpec(spec.signal.name, normalized_signal),
            sizing=replace(spec.sizing, params=normalized_sizing),
            filters=normalized_filters,
        )
        return normalized

    def _validate_filter(self, filter_spec: FilterSpec) -> FilterSpec:
        if filter_spec.name not in self.filter_handlers:
            raise ValueError(f"Unknown filter handler: {filter_spec.name}")
        normalized = self.filter_handlers[filter_spec.name]["normalize_params"](filter_spec.params)
        self._validate_finite_params(filter_spec.name, normalized)
        return FilterSpec(name=filter_spec.name, params=normalized)

    def _validate_exec_config(self, spec: StrategySpec) -> None:
        config = spec.exec_config
        values = {
            "initial_cash": config.initial_cash,
            "commission_per_order": config.commission_per_order,
            "commission_per_share": config.commission_per_share,
            "slippage_bps": config.slippage_bps,
            "spread_bps": config.spread_bps,
            "max_position_notional": config.max_position_notional,
            "stop_loss_bps": config.stop_loss_bps,
            "no_new_entry_minutes_before_close": config.no_new_entry_minutes_before_close,
        }
        self._validate_finite_params("exec_config", values)

    def _validate_finite_params(self, scope: str, params: dict[str, object]) -> None:
        for name, value in params.items():
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"{scope}.{name} must be finite")

    def required_history(self, spec: StrategySpec) -> int:
        validated = self.validate_spec(spec)
        signal_history = int(self.signal_handlers[validated.signal.name]["required_history"](validated.signal.params))
        filter_history = [
            int(self.filter_handlers[filter_spec.name]["required_history"](filter_spec.params))
            for filter_spec in validated.filters
        ]
        return max((signal_history, *filter_history))

    def generate_regime(self, spec: StrategySpec, history_bars: tuple[MarketBar, ...], test_bars: tuple[MarketBar, ...]) -> list[bool]:
        validated = self.validate_spec(spec)
        handler = self.signal_handlers[validated.signal.name]["generate_regime"]
        signal_regime = handler(history_bars, test_bars, validated.signal.params)
        for filter_spec in validated.filters:
            mask = self.filter_handlers[filter_spec.name]["generate_mask"](history_bars, test_bars, filter_spec.params)
            signal_regime = [regime_on and mask_on for regime_on, mask_on in zip(signal_regime, mask)]
        return signal_regime

    def compute_sizing_fraction(self, spec: StrategySpec) -> float:
        validated = self.validate_spec(spec)
        return float(self.sizing_handlers[validated.sizing.name]["compute_fraction"](validated.sizing.params))

    def parameter_grid(self, signal_name: str) -> tuple[dict[str, int | float], ...]:
        if signal_name not in self.signal_handlers:
            raise ValueError(f"Unknown signal handler: {signal_name}")
        return self.signal_handlers[signal_name]["parameter_grid"]()

    def neighbors(self, spec: StrategySpec) -> tuple[StrategySpec, ...]:
        validated = self.validate_spec(spec)
        neighbor_params = self.signal_handlers[validated.signal.name]["neighbors"](validated.signal.params)
        return tuple(
            replace(
                validated,
                name=f"{validated.signal.name}_{index}",
                signal=SignalSpec(validated.signal.name, params),
            )
            for index, params in enumerate(neighbor_params)
        )


REGISTRY = StrategyRegistry()
