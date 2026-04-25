from __future__ import annotations

from trader.cli.loop_cmd import DEFAULT_OVERPLAN_FACTOR, DEFAULT_PREVIEW_FACTOR, MIN_PLANNED_SPECS, TIMING_PHASES
from trader.cli.loop_cmd import _add_timing, _max_preview_count, _new_timings, _planned_spec_count, _suppression_audit_types, _timing_payload
from trader.cli.loop_cmd import build_parser
from trader.research.suppressor import SuppressedSpec


def test_planned_spec_count_uses_named_overplan_factor_and_floor() -> None:
    assert _planned_spec_count(batch_size=20, overplan_factor=DEFAULT_OVERPLAN_FACTOR) == 80
    assert _planned_spec_count(batch_size=1, overplan_factor=DEFAULT_OVERPLAN_FACTOR) == MIN_PLANNED_SPECS
    assert _planned_spec_count(batch_size=30, overplan_factor=3) == 90


def test_max_preview_count_uses_named_preview_factor() -> None:
    assert _max_preview_count(batch_size=6, preview_factor=DEFAULT_PREVIEW_FACTOR) == 24
    assert _max_preview_count(batch_size=6, preview_factor=0) == 6


def test_suppression_audit_types_separate_evaluated_from_discarded() -> None:
    records = (
        SuppressedSpec("evaluated_hash", "ema_cross", "failure_1", 0.0, ("neighborhood_pass",), 2, 20.0),
        SuppressedSpec("discarded_hash", "ema_cross", "failure_2", 0.1, ("regime_pass",), 1, 5.0),
    )

    assert _suppression_audit_types(records, {"evaluated_hash"}) == {
        "evaluated_hash": "evaluated",
        "discarded_hash": "preview_discarded",
    }


def test_timing_payload_includes_all_loop_phases(monkeypatch) -> None:
    timings = _new_timings()
    monkeypatch.setattr("trader.cli.loop_cmd.perf_counter", lambda: 1.1234567)

    _add_timing(timings, "planning", 1.0)
    payload = _timing_payload(timings)

    assert tuple(payload) == TIMING_PHASES
    assert payload["planning"] == 0.123457
    assert payload["stage_a"] == 0.0


def test_loop_signal_family_accepts_composite() -> None:
    args = build_parser().parse_args(["--signal-family", "composite"])

    assert args.signal_family == ["composite"]
