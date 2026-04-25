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
from typing import TYPE_CHECKING, Any, Sequence

from trader.strategies.spec import StrategySpec

if TYPE_CHECKING:
    # LedgerEntry uses datetime.UTC (Python 3.11+); only import at type-check time.
    from trader.ledger.entry import LedgerEntry


# ---------------------------------------------------------------------------
# Shared distance function (also imported by candidates.py to avoid duplication)
# ---------------------------------------------------------------------------

def spec_distance(left: dict[str, object], right: dict[str, object]) -> float:
    """
    Normalized parameter distance between two spec payloads in [0, 1].

    Only compares signal params within the same family; cross-family specs
    always return 1.0 (maximum distance, so they never suppress each other).
    """
    left_signal = dict(left.get("signal", {}))
    right_signal = dict(right.get("signal", {}))
    if left_signal.get("name") != right_signal.get("name"):
        return 1.0
    left_params = dict(left_signal.get("params", {}))
    right_params = dict(right_signal.get("params", {}))
    keys = sorted(set(left_params) | set(right_params))
    if not keys:
        return 0.0
    distance = 0.0
    for key in keys:
        lv = left_params.get(key)
        rv = right_params.get(key)
        if isinstance(lv, (int, float)) and isinstance(rv, (int, float)):
            scale = max(abs(float(lv)), abs(float(rv)), 1.0)
            distance += abs(float(lv) - float(rv)) / scale
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
        radius: float = 0.15,
        weight_cap: float = 40.0,
    ) -> None:
        if radius <= 0:
            raise ValueError("radius must be > 0")
        if weight_cap < 0:
            raise ValueError("weight_cap must be >= 0")
        self.radius = radius
        self.weight_cap = weight_cap
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
        total_weight = 0.0
        count_in_radius = 0

        for failure in same_family:
            dist = spec_distance(
                candidate_payload,
                failure.spec.to_payload(include_name=False),
            )
            if dist < nearest_dist:
                nearest_dist = dist
                nearest = failure
            if dist < self.radius:
                # Linear decay: full penalty at dist=0, zero at dist=radius
                contribution = self.weight_cap * (1.0 - dist / self.radius)
                total_weight += contribution
                count_in_radius += 1

        if nearest is None or total_weight == 0.0:
            return None

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
