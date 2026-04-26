from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from trader.cli.loop_cmd import DEFAULT_OVERPLAN_FACTOR, DEFAULT_PREVIEW_FACTOR, MIN_PLANNED_SPECS, TIMING_PHASES
from trader.cli.loop_cmd import _add_timing, _max_preview_count, _new_timings, _planned_spec_count, _suppression_audit_types, _timing_payload
from trader.cli.loop_cmd import _evaluate_candidate_worker, _evaluate_selected_candidates, _stage_b_worker_count, build_parser
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


def test_stage_b_worker_count_caps_at_eight() -> None:
    assert _stage_b_worker_count(0) == 0
    assert _stage_b_worker_count(1) == 4
    assert _stage_b_worker_count(4) == 4
    assert _stage_b_worker_count(6) == 6
    assert _stage_b_worker_count(12) == 8


def test_evaluate_selected_candidates_preserves_order_and_sums_worker_timings(monkeypatch) -> None:
    used_workers = []

    class InlineExecutor:
        def __init__(self, *, max_workers: int) -> None:
            used_workers.append(max_workers)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def map(self, fn, payloads):
            return [fn(payload) for payload in payloads]

    def fake_worker(payload):
        _, preview = payload
        return preview, {"stage_a": 0.5, "stage_b": 2.0, "robustness_neighbors": 1.25}

    monkeypatch.setattr("trader.cli.loop_cmd._evaluate_candidate_worker", fake_worker)
    selected = (
        SimpleNamespace(preview="first"),
        SimpleNamespace(preview="second"),
    )

    results, timings = _evaluate_selected_candidates(
        selected,
        "/tmp/market.db",
        executor_factory=InlineExecutor,
    )

    assert used_workers == [4]
    assert results == ("first", "second")
    assert timings == {
        "stage_a": 1.0,
        "stage_b": 4.0,
        "robustness_neighbors": 2.5,
    }


def test_evaluate_candidate_worker_reopens_database_path(monkeypatch) -> None:
    seen = {}

    class FakeDataView:
        def __init__(self, database_path: Path) -> None:
            seen["database_path"] = database_path

    class FakeRunner:
        phase_timings = {"stage_b": 1.25}

        def __init__(self, data_view, registry) -> None:
            seen["data_view"] = data_view
            seen["registry"] = registry

        def evaluate_preview(self, preview, *, include_robustness: bool):
            seen["preview"] = preview
            seen["include_robustness"] = include_robustness
            return "result"

    monkeypatch.setattr("trader.cli.loop_cmd.DataView", FakeDataView)
    monkeypatch.setattr("trader.cli.loop_cmd.EvaluationRunner", FakeRunner)

    result, timings = _evaluate_candidate_worker(("/tmp/market.db", "preview"))

    assert result == "result"
    assert timings == {"stage_b": 1.25}
    assert seen["database_path"] == Path("/tmp/market.db")
    assert seen["preview"] == "preview"
    assert seen["include_robustness"] is True
