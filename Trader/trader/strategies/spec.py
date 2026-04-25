from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from hashlib import sha256
from typing import Literal


Scalar = int | float | str | bool
ParamValue = Scalar | list["ParamValue"] | dict[str, "ParamValue"]


@dataclass(frozen=True)
class SignalSpec:
    name: str
    params: dict[str, ParamValue] = field(default_factory=dict)


@dataclass(frozen=True)
class SizingSpec:
    name: str
    params: dict[str, Scalar] = field(default_factory=dict)


@dataclass(frozen=True)
class FilterSpec:
    name: str
    params: dict[str, Scalar] = field(default_factory=dict)


@dataclass(frozen=True)
class ExecConfig:
    initial_cash: float = 100_000.0
    commission_per_order: float = 0.0
    commission_per_share: float = 0.0
    slippage_bps: float = 1.0
    spread_bps: float = 0.0
    max_position_notional: float | None = None
    stop_loss_bps: float | None = None
    entry_session_window: str = "all"
    no_new_entry_minutes_before_close: int | None = None
    regular_session_only: bool = True
    flat_at_close: bool = True

    def cost_model_id(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class StrategySpec:
    name: str
    instrument: Literal["SPY"] = "SPY"
    multiplier: int = 1
    timespan: Literal["minute"] = "minute"
    signal: SignalSpec = field(default_factory=lambda: SignalSpec("ema_cross", {}))
    sizing: SizingSpec = field(default_factory=lambda: SizingSpec("full_notional", {}))
    filters: tuple[FilterSpec, ...] = ()
    exec_config: ExecConfig = field(default_factory=ExecConfig)
    feature_set: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    seed: int = 0

    def to_payload(self, *, include_name: bool = True) -> dict[str, object]:
        exec_payload = asdict(self.exec_config)
        payload: dict[str, object] = {
            "instrument": self.instrument,
            "multiplier": self.multiplier,
            "timespan": self.timespan,
            "signal": {"name": self.signal.name, "params": dict(sorted(self.signal.params.items()))},
            "sizing": {"name": self.sizing.name, "params": dict(sorted(self.sizing.params.items()))},
            "filters": [
                {"name": filter_spec.name, "params": dict(sorted(filter_spec.params.items()))}
                for filter_spec in sorted(self._semantic_filters(), key=lambda item: item.name)
            ],
            "exec_config": exec_payload,
            "feature_set": list(self.feature_set),
            "tags": list(self.tags),
            "seed": self.seed,
        }
        if include_name:
            payload["name"] = self.name
        return payload

    def _semantic_filters(self) -> tuple[FilterSpec, ...]:
        return tuple(
            filter_spec for filter_spec in self.filters
            if not _is_redundant_regular_session_filter(filter_spec, self.exec_config)
        )

    def canonical_json(self, *, include_name: bool = True) -> str:
        return json.dumps(
            self.to_payload(include_name=include_name),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )

    def spec_hash(self) -> str:
        # Display names do not change execution semantics and must not affect dedupe.
        return sha256(self.canonical_json(include_name=False).encode("utf-8")).hexdigest()

    @classmethod
    def from_payload(cls, payload: dict[str, object]) -> "StrategySpec":
        signal = payload.get("signal") or {}
        sizing = payload.get("sizing") or {}
        filter_specs = payload.get("filters") or []
        exec_config = payload.get("exec_config") or {}
        return cls(
            name=str(payload.get("name", "unnamed_strategy")),
            instrument=str(payload.get("instrument", "SPY")),  # type: ignore[arg-type]
            multiplier=int(payload.get("multiplier", 1)),
            timespan=str(payload.get("timespan", "minute")),  # type: ignore[arg-type]
            signal=SignalSpec(
                name=str((signal or {}).get("name", "ema_cross")),
                params=dict((signal or {}).get("params", {})),
            ),
            sizing=SizingSpec(
                name=str((sizing or {}).get("name", "full_notional")),
                params=dict((sizing or {}).get("params", {})),
            ),
            filters=tuple(
                FilterSpec(name=str(item.get("name", "")), params=dict(item.get("params", {})))
                for item in filter_specs
            ),
            exec_config=ExecConfig(**dict(exec_config)),
            feature_set=tuple(payload.get("feature_set", ())),  # type: ignore[arg-type]
            tags=tuple(payload.get("tags", ())),  # type: ignore[arg-type]
            seed=int(payload.get("seed", 0)),
        )


def _is_redundant_regular_session_filter(filter_spec: FilterSpec, exec_config: ExecConfig) -> bool:
    if filter_spec.name != "session" or not exec_config.regular_session_only:
        return False
    return str(filter_spec.params.get("session", "regular")) == "regular"
