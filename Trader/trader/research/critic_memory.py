from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from trader.ledger.entry import LedgerEntry, json_dumps, json_loads
from trader.research.critic import planning_penalty_from_critique
from trader.research.suppressor import _parameter_scales, spec_distance
from trader.strategies.registry import REGISTRY
from trader.strategies.spec import StrategySpec

_VERSION = 1
_DEFAULT_RADIUS = 0.15
_DEFAULT_WEIGHT_CAP = 25.0


@dataclass(frozen=True)
class CriticRegionRecord:
    experiment_id: str
    spec_hash: str
    signal_family: str
    spec_payload: dict[str, object]
    planning_penalties: dict[str, float]
    penalty_weight: float
    completed_at_utc: str | None

    def to_payload(self) -> dict[str, object]:
        return {
            "completed_at_utc": self.completed_at_utc,
            "experiment_id": self.experiment_id,
            "penalty_weight": self.penalty_weight,
            "planning_penalties": dict(sorted(self.planning_penalties.items())),
            "signal_family": self.signal_family,
            "spec": self.spec_payload,
            "spec_hash": self.spec_hash,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, object]) -> "CriticRegionRecord":
        penalties = {
            str(key): float(value)
            for key, value in dict(payload.get("planning_penalties", {})).items()
            if isinstance(value, (int, float))
        }
        return cls(
            experiment_id=str(payload["experiment_id"]),
            spec_hash=str(payload["spec_hash"]),
            signal_family=str(payload["signal_family"]),
            spec_payload=dict(payload["spec"]),
            planning_penalties=penalties,
            penalty_weight=float(payload["penalty_weight"]),
            completed_at_utc=(
                None if payload.get("completed_at_utc") is None else str(payload.get("completed_at_utc"))
            ),
        )


class CriticRegionMemory:
    def __init__(
        self,
        records: Sequence[CriticRegionRecord] = (),
        *,
        registry: Any = REGISTRY,
        radius: float = _DEFAULT_RADIUS,
        weight_cap: float = _DEFAULT_WEIGHT_CAP,
    ) -> None:
        if radius <= 0:
            raise ValueError("radius must be > 0")
        if weight_cap < 0:
            raise ValueError("weight_cap must be >= 0")
        self.records = tuple(sorted(records, key=lambda record: (
            record.signal_family,
            record.spec_hash,
            record.experiment_id,
        )))
        self.radius = radius
        self.weight_cap = weight_cap
        self._parameter_scales_by_family = _parameter_scales(registry)

    @classmethod
    def from_entries(
        cls,
        entries: Sequence[LedgerEntry],
        *,
        registry: Any = REGISTRY,
        radius: float = _DEFAULT_RADIUS,
        weight_cap: float = _DEFAULT_WEIGHT_CAP,
    ) -> "CriticRegionMemory":
        records = []
        for entry in entries:
            if entry.status != "completed":
                continue
            penalty = planning_penalty_from_critique(entry.critique)
            if penalty <= 0.0:
                continue
            records.append(
                CriticRegionRecord(
                    experiment_id=entry.experiment_id,
                    spec_hash=entry.spec_hash,
                    signal_family=entry.spec.signal.name,
                    spec_payload=entry.spec.to_payload(include_name=False),
                    planning_penalties=_planning_penalties(entry.critique),
                    penalty_weight=penalty,
                    completed_at_utc=entry.completed_at_utc,
                )
            )
        return cls(records, registry=registry, radius=radius, weight_cap=weight_cap)

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        registry: Any = REGISTRY,
    ) -> "CriticRegionMemory":
        memory_path = Path(path)
        if not memory_path.exists():
            return cls(registry=registry)
        try:
            payload = json_loads(memory_path.read_text(encoding="utf-8"), default={}) or {}
            records = [
                CriticRegionRecord.from_payload(dict(item))
                for item in payload.get("records", [])
                if isinstance(item, dict)
            ]
            return cls(
                records,
                registry=registry,
                radius=float(payload.get("radius", _DEFAULT_RADIUS)),
                weight_cap=float(payload.get("weight_cap", _DEFAULT_WEIGHT_CAP)),
            )
        except Exception:
            return cls(registry=registry)

    @property
    def record_count(self) -> int:
        return len(self.records)

    def penalty(self, spec: StrategySpec) -> float:
        if not self.records:
            return 0.0
        candidate_payload = spec.to_payload(include_name=False)
        total = 0.0
        for record in self.records:
            if record.signal_family != spec.signal.name:
                continue
            distance = spec_distance(
                candidate_payload,
                record.spec_payload,
                parameter_scales=self._parameter_scales_by_family.get(spec.signal.name),
            )
            if distance < self.radius:
                total += record.penalty_weight * (1.0 - distance / self.radius)
        return round(min(total, self.weight_cap), 4)

    def to_payload(self) -> dict[str, object]:
        return {
            "radius": self.radius,
            "records": [record.to_payload() for record in self.records],
            "version": _VERSION,
            "weight_cap": self.weight_cap,
        }

    def write(self, path: str | Path) -> None:
        memory_path = Path(path)
        memory_path.parent.mkdir(parents=True, exist_ok=True)
        memory_path.write_text(json_dumps(self.to_payload(), pretty=True), encoding="utf-8")


def _planning_penalties(payload: dict[str, object] | None) -> dict[str, float]:
    if not payload:
        return {}
    penalties = payload.get("planning_penalties")
    if not isinstance(penalties, dict):
        return {}
    return {
        str(key): float(value)
        for key, value in penalties.items()
        if isinstance(value, (int, float))
    }
