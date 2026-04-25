"""
RegionSuppressor — critic-to-planner feedback.

Reads completed ledger entries that failed robustness checks and computes a
suppression penalty for new candidates that are nearby in parameter space.
The penalty is subtracted from the candidate's static score in
DeterministicCandidateQueue, so the loop organically avoids re-exploring
regions that have consistently failed.

Design constraints:
- Deterministic: same history → same suppression weights, no randomness.
- Auditable: every suppression decision is captured in a SuppressedSpec record
  and written to the ledger's suppression_log table by loop_cmd.
- Evaluator-blind: this module never imports from trader.evaluation.*.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from trader.strategies.registry import REGISTRY
from trader.strategies.spec import StrategySpec

if TYPE_CHECKING:
    # LedgerEntry uses datetime.UTC (Python 3.11+); only import at type-check time.
    from trader.ledger.entry import LedgerEntry


# ---------------------------------------------------------------------------
# Shared distance function (also imported by candidates.py to avoid duplication)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _ParamScale:
    minimum: float
    maximum: float
    step: float

    @property
    def denominator(self) -> float:
        return max(self.maximum - self.minimum, self.step, 1.0)


def _parameter_scales(registry: Any) -> dict[str, dict[str, _ParamScale]]:
    scales: dict[str, dict[str, _ParamScale]] = {}
    for signal_name in registry.signal_handlers:
        grid = registry.parameter_grid(signal_name)
        keys = sorted({key for params in grid for key, value in params.items() if isinstance(value, (int, float))})
        family_scales: dict[str, _ParamScale] = {}
        for key in keys:
            values = sorted({float(params[key]) for params in grid if isinstance(params.get(key), (int, float))})
            positive_steps = [
                right - left
                for left, right in zip(values, values[1:])
                if right > left
            ]
            step = min(positive_steps) if positive_steps else 1.0
            family_scales[key] = _ParamScale(minimum=min(values), maximum=max(values), step=step)
        scales[signal_name] = family_scales
    return scales


def spec_distance(
    left: dict[str, object],
    right: dict[str, object],
    *,
    parameter_scales: Mapping[str, _ParamScale] | None = None,
) -> float:
    """
    Normalized parameter distance between two spec payloads in [0, 1].

    Only compares signal params within the same family; cross-family specs
    always return 1.0 (maximum distance, so they never suppress each other).
    """
    left_signal = dict(left.get("signal", {}))
    right_signal = dict(right.get("signal", {}))
    if left_signal.get("name") != right_signal.get("name"):
        return 1.0
    signal_name = str(left_signal.get("name"))
    left_params = dict(left_signal.get("params", {}))
    right_params = dict(right_signal.get("params", {}))
    keys = sorted(set(left_params) | set(right_params))
    if not keys:
        return 0.0
    scales = parameter_scales or {}
    distance = 0.0
    for key in keys:
        lv = left_params.get(key)
        rv = right_params.get(key)
        if isinstance(lv, (int, float)) and isinstance(rv, (int, float)):
            scale = scales.get(key)
            denominator = scale.denominator if scale is not None else max(abs(float(lv)), abs(float(rv)), 1.0)
            distance += min(abs(float(lv) - float(rv)) / denominator, 1.0)
        else:
            distance += 0.0 if lv == rv else 1.0
    normalized = distance / len(keys)
    return normalized if isfinite(normalized) else 0.0


# ---------------------------------------------------------------------------
# Robustness failure detection
# ---------------------------------------------------------------------------

_ROBUSTNESS_GATE_CHECKS = (
    "fold_consistency_pass",
    "neighborhood_pass",
    "drawdown_pass",
    "regime_pass",
)
_FAILURE_CHECK_WEIGHTS = {
    "neighborhood_pass": 1.0,
    "fold_consistency_pass": 0.75,
    "regime_pass": 0.6,
    "drawdown_pass": 0.5,
}
_LARGE_SUPPRESSION_FAILURE_COUNT = 2
_SINGLE_FAILURE_WEIGHT_MULTIPLIER = 0.5


def _failed_gate_checks(entry: Any) -> tuple[str, ...]:
    """Return the names of Boolean robustness gate checks that failed."""
    checks = entry.robustness_checks
    return tuple(
        name for name in _ROBUSTNESS_GATE_CHECKS
        if name in checks and not bool(checks[name])
    )


def _is_robustness_failure(entry: Any) -> bool:
    """True when the entry has real robustness data and at least one gate failed."""
    if not entry.robustness_checks:
        return False  # empty checks = exploratory by default, not a measured failure
    return len(_failed_gate_checks(entry)) > 0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SuppressedSpec:
    """
    Audit record for one candidate that received a suppression penalty.

    Written to the ledger's suppression_log table by loop_cmd so every
    suppression decision is traceable to the specific failure(s) that caused it.
    """
    spec_hash: str
    signal_family: str
    nearest_failure_experiment_id: str
    nearest_failure_distance: float
    failed_check_names: tuple[str, ...]
    failure_count_in_radius: int        # how many failures contributed
    suppression_weight: float           # penalty subtracted from static_score


class RegionSuppressor:
    """
    Assigns suppression penalties to candidates near known-bad parameter regions.

    Parameters
    ----------
    entries:
        All completed ledger entries.  The suppressor internally filters to
        those that failed at least one robustness gate.
    radius:
        Normalized distance threshold.  Failures within this radius
        contribute to the penalty; beyond it they have zero influence.
        Default 0.15 ≈ within ~1-2 parameter steps on the existing grids.
    weight_cap:
        Maximum suppression penalty per candidate.  Set relative to the
        static_score range (~0–100 in DeterministicCandidateQueue).
        Default 40.0 means a heavily suppressed candidate scores ≤ the
        neutral floor but is never outright removed from the queue.
    """

    def __init__(
        self,
        entries: Sequence[Any],  # Sequence[LedgerEntry] at type-check time
        *,
        registry: Any = REGISTRY,
        radius: float = 0.15,
        weight_cap: float = 40.0,
    ) -> None:
        if radius <= 0:
            raise ValueError("radius must be > 0")
        if weight_cap < 0:
            raise ValueError("weight_cap must be >= 0")
        self.radius = radius
        self.weight_cap = weight_cap
        self._parameter_scales_by_family = _parameter_scales(registry)
        self._failures: tuple[Any, ...] = tuple(
            entry for entry in entries if _is_robustness_failure(entry)
        )

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    @property
    def failure_count(self) -> int:
        return len(self._failures)

    def assess(self, spec: StrategySpec) -> SuppressedSpec | None:
        """
        Return a SuppressedSpec if this candidate is near known failures,
        else None (no suppression).

        The returned suppression_weight should be subtracted from the
        candidate's static score.
        """
        candidate_payload = spec.to_payload(include_name=False)
        same_family = [
            f for f in self._failures
            if f.spec.signal.name == spec.signal.name
        ]
        if not same_family:
            return None

        nearest: Any | None = None
        nearest_dist = float("inf")
        weighted_proximity = 0.0
        count_in_radius = 0

        for failure in same_family:
            dist = spec_distance(
                candidate_payload,
                failure.spec.to_payload(include_name=False),
                parameter_scales=self._parameter_scales_by_family.get(spec.signal.name),
            )
            if dist < nearest_dist:
                nearest_dist = dist
                nearest = failure
            if dist < self.radius:
                # Linear decay: full penalty at dist=0, zero at dist=radius
                failure_weight = _failure_weight(_failed_gate_checks(failure))
                weighted_proximity += failure_weight * (1.0 - dist / self.radius)
                count_in_radius += 1

        if nearest is None or weighted_proximity == 0.0:
            return None
        repeat_multiplier = (
            1.0
            if count_in_radius >= _LARGE_SUPPRESSION_FAILURE_COUNT
            else _SINGLE_FAILURE_WEIGHT_MULTIPLIER
        )
        total_weight = self.weight_cap * weighted_proximity * repeat_multiplier

        return SuppressedSpec(
            spec_hash=spec.spec_hash(),
            signal_family=spec.signal.name,
            nearest_failure_experiment_id=nearest.experiment_id,
            nearest_failure_distance=round(nearest_dist, 6),
            failed_check_names=_failed_gate_checks(nearest),
            failure_count_in_radius=count_in_radius,
            suppression_weight=round(min(total_weight, self.weight_cap), 4),
        )

    def suppression_weight(self, spec: StrategySpec) -> float:
        """Convenience: return just the penalty float (0.0 if not suppressed)."""
        record = self.assess(spec)
        return record.suppression_weight if record is not None else 0.0

    def summary(self) -> dict[str, object]:
        """Structured summary for logging."""
        by_family: dict[str, int] = {}
        for entry in self._failures:
            family = entry.spec.signal.name
            by_family[family] = by_family.get(family, 0) + 1
        return {
            "total_failures_loaded": self.failure_count,
            "failures_by_family": dict(sorted(by_family.items())),
            "radius": self.radius,
            "weight_cap": self.weight_cap,
        }


def _failure_weight(failed_check_names: Sequence[str]) -> float:
    if not failed_check_names:
        return 0.0
    return max(_FAILURE_CHECK_WEIGHTS.get(name, 0.4) for name in failed_check_names)
