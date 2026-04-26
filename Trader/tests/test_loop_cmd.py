from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from trader.cli.loop_cmd import DEFAULT_OVERPLAN_FACTOR, DEFAULT_PREVIEW_FACTOR, MIN_PLANNED_SPECS, TIMING_PHASES
from trader.cli.loop_cmd import _active_data_snapshot_id, _add_timing, _current_snapshot_entries, _max_preview_count, _new_timings, _planned_spec_count, _should_restart_planner, _suppression_audit_types, _timing_payload
from trader.cli.loop_cmd import _evaluate_candidate_worker, _evaluate_selected_candidates, _load_or_seed_critic_memory, _loop_experiment_summary, _mark_stage_a_passed, _prescreen_stage_a_candidates, _stage_b_worker_count, _write_loop_outputs, build_parser
from trader.config import Settings
from trader.evaluation.runner import ExperimentResult
from trader.research.suppressor import SuppressedSpec
from trader.strategies.spec import SignalSpec, StrategySpec


def test_planned_spec_count_uses_named_overplan_factor_and_floor() -> None:
    assert _planned_spec_count(batch_size=20, overplan_factor=DEFAULT_OVERPLAN_FACTOR) == 80
    assert _planned_spec_count(batch_size=1, overplan_factor=DEFAULT_OVERPLAN_FACTOR) == MIN_PLANNED_SPECS
    assert _planned_spec_count(batch_size=30, overplan_factor=3) == 90


def test_max_preview_count_uses_named_preview_factor() -> None:
    assert _max_preview_count(batch_size=6, preview_factor=DEFAULT_PREVIEW_FACTOR) == 24
    assert _max_preview_count(batch_size=6, preview_factor=0) == 6


def test_should_restart_planner_when_candidate_reuse_is_high() -> None:
    assert _should_restart_planner(SimpleNamespace(previewed_count=0, duplicate_count=1), planned_count=64)
    assert _should_restart_planner(SimpleNamespace(previewed_count=8, duplicate_count=48), planned_count=64)
    assert not _should_restart_planner(SimpleNamespace(previewed_count=8, duplicate_count=12), planned_count=64)
    assert not _should_restart_planner(SimpleNamespace(previewed_count=0, duplicate_count=1), planned_count=0)


def test_suppression_audit_types_separate_evaluated_from_discarded() -> None:
    records = (
        SuppressedSpec("evaluated_hash", "ema_cross", "failure_1", 0.0, ("neighborhood_pass",), 2, 20.0),
        SuppressedSpec("discarded_hash", "ema_cross", "failure_2", 0.1, ("regime_pass",), 1, 5.0),
    )

    assert _suppression_audit_types(records, {"evaluated_hash"}) == {
        "evaluated_hash": "evaluated",
        "discarded_hash": "preview_discarded",
    }
    assert _suppression_audit_types(records, set(), unevaluated_type="stage_a_suppressed") == {
        "evaluated_hash": "stage_a_suppressed",
        "discarded_hash": "stage_a_suppressed",
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


def test_loop_signal_family_accepts_vwap_deviation_as_opt_in() -> None:
    args = build_parser().parse_args(["--signal-family", "vwap_deviation"])

    assert args.signal_family == ["vwap_deviation"]


def test_active_data_snapshot_id_uses_evaluation_preview_research_slice() -> None:
    calls = []

    class FakeRunner:
        def preview_walk_forward(self, spec, *, num_folds, embargo_bars, locked_holdout_months):
            calls.append(
                {
                    "spec_name": spec.name,
                    "num_folds": num_folds,
                    "embargo_bars": embargo_bars,
                    "locked_holdout_months": locked_holdout_months,
                }
            )
            return SimpleNamespace(data_slice=SimpleNamespace(snapshot_id="active_snapshot"))

    assert _active_data_snapshot_id(
        FakeRunner(),
        num_folds=3,
        embargo_bars=1,
        locked_holdout_months=2,
    ) == "active_snapshot"
    assert calls == [
        {
            "spec_name": "snapshot_probe",
            "num_folds": 3,
            "embargo_bars": 1,
            "locked_holdout_months": 2,
        }
    ]


def test_current_snapshot_entries_filters_stale_entries_and_keeps_distinct_costs() -> None:
    entries = (
        SimpleNamespace(experiment_id="old", data_snapshot_id="old_snapshot", spec="old_spec", cost_model_id="cost_a"),
        SimpleNamespace(experiment_id="active_a", data_snapshot_id="active_snapshot", spec="active_spec_a", cost_model_id="cost_a"),
        SimpleNamespace(experiment_id="active_b", data_snapshot_id="active_snapshot", spec="active_spec_b", cost_model_id="cost_b"),
    )

    class FakeRunner:
        def preview_walk_forward(self, spec, *, num_folds, embargo_bars, locked_holdout_months):
            assert num_folds == 3
            assert embargo_bars == 1
            assert locked_holdout_months == 2
            snapshot_id = "old_current_snapshot" if spec == "old_spec" else "active_snapshot"
            return SimpleNamespace(data_slice=SimpleNamespace(snapshot_id=snapshot_id))

    current = _current_snapshot_entries(
        entries,
        FakeRunner(),
        num_folds=3,
        embargo_bars=1,
        locked_holdout_months=2,
    )

    assert [entry.experiment_id for entry in current] == ["active_a", "active_b"]


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

        def evaluate_preview(self, preview, *, include_robustness: bool, run_stage_a: bool):
            seen["preview"] = preview
            seen["include_robustness"] = include_robustness
            seen["run_stage_a"] = run_stage_a
            return "result"

    monkeypatch.setattr("trader.cli.loop_cmd.DataView", FakeDataView)
    monkeypatch.setattr("trader.cli.loop_cmd.EvaluationRunner", FakeRunner)

    result, timings = _evaluate_candidate_worker(("/tmp/market.db", "preview"))

    assert result == "result"
    assert timings == {"stage_b": 1.25}
    assert seen["database_path"] == Path("/tmp/market.db")
    assert seen["preview"] == "preview"
    assert seen["include_robustness"] is True
    assert seen["run_stage_a"] is False


def test_prescreen_stage_a_suppresses_third_nearby_failure(monkeypatch) -> None:
    specs = (
        StrategySpec(
            name="near_1",
            signal=SignalSpec("ema_cross", {"fast_length": 20, "slow_length": 80, "signal_buffer_bps": 0.0}),
        ),
        StrategySpec(
            name="near_2",
            signal=SignalSpec("ema_cross", {"fast_length": 20, "slow_length": 80, "signal_buffer_bps": 0.0}),
        ),
        StrategySpec(
            name="near_3",
            signal=SignalSpec("ema_cross", {"fast_length": 20, "slow_length": 80, "signal_buffer_bps": 0.0}),
        ),
        StrategySpec(
            name="far",
            signal=SignalSpec("breakout", {"entry_window": 20, "exit_window": 10, "buffer_bps": 0.0}),
        ),
    )
    candidates = tuple(
        SimpleNamespace(preview=SimpleNamespace(spec=spec, evaluation_key=f"key_{index}"))
        for index, spec in enumerate(specs)
    )

    def stage_a_result(preview):
        if preview.spec.signal.name == "breakout":
            return None
        return SimpleNamespace(
            experiment_id=preview.evaluation_key,
            spec=preview.spec,
            robustness_checks={
                "stage_a_pass": False,
                "stage_a_reject_non_positive_return": True,
            },
        )

    runner = SimpleNamespace(evaluate_stage_a_preview=stage_a_result)
    monkeypatch.setattr("trader.cli.loop_cmd.perf_counter", lambda: 1.0)

    result = _prescreen_stage_a_candidates(candidates, runner, suppressor_radius=0.2)

    assert [item[0].preview.spec.name for item in result.completed_results] == ["near_1", "near_2"]
    assert [candidate.preview.spec.name for candidate in result.stage_b_candidates] == ["far"]
    assert len(result.suppression_records) == 1
    assert result.suppression_records[0].spec_hash == specs[2].spec_hash()


def test_mark_stage_a_passed_carries_prescreen_evidence() -> None:
    spec = StrategySpec(
        name="ema",
        signal=SignalSpec("ema_cross", {"fast_length": 20, "slow_length": 80, "signal_buffer_bps": 0.0}),
    )
    result = ExperimentResult(
        experiment_id="exp_1",
        status="completed",
        spec=spec,
        spec_hash=spec.spec_hash(),
        data_snapshot_id="snapshot",
        split_plan_id="split",
        cost_model_id=spec.exec_config.cost_model_id(),
        aggregate_metrics={"return_pct": 1.0},
        fold_results=tuple(),
        robustness_checks={},
        promotion_stage="exploratory",
    )

    marked = _mark_stage_a_passed(result)

    assert marked is not result
    assert marked.aggregate_metrics == {"return_pct": 1.0, "stage_a_pass": 1.0}


def test_load_or_seed_critic_memory_consumes_existing_disk_file(tmp_path: Path, monkeypatch) -> None:
    calls = []
    path = tmp_path / "critic_memory.json"
    path.write_text("{}", encoding="utf-8")

    class FakeMemory:
        @classmethod
        def load(cls, memory_path, *, registry):
            calls.append(("load", memory_path))
            return "loaded"

        @classmethod
        def from_entries(cls, history_entries, *, registry):
            calls.append(("from_entries", tuple(history_entries)))
            return cls()

        def write(self, memory_path) -> None:
            calls.append(("write", memory_path))

    monkeypatch.setattr("trader.cli.loop_cmd.CriticRegionMemory", FakeMemory)

    assert _load_or_seed_critic_memory(path, ("history",)) == "loaded"
    assert calls == [("load", path)]


def test_load_or_seed_critic_memory_seeds_missing_disk_file(tmp_path: Path, monkeypatch) -> None:
    calls = []
    path = tmp_path / "critic_memory.json"

    class FakeMemory:
        @classmethod
        def load(cls, memory_path, *, registry):
            calls.append(("load", memory_path))
            return "loaded"

        @classmethod
        def from_entries(cls, history_entries, *, registry):
            calls.append(("from_entries", tuple(history_entries)))
            return cls()

        def write(self, memory_path) -> None:
            calls.append(("write", memory_path))

    monkeypatch.setattr("trader.cli.loop_cmd.CriticRegionMemory", FakeMemory)

    memory = _load_or_seed_critic_memory(path, ("history",))

    assert isinstance(memory, FakeMemory)
    assert calls == [("from_entries", ("history",)), ("write", path)]


def test_loop_experiment_summary_carries_reporting_payload() -> None:
    spec = StrategySpec(
        name="multi_signal_test",
        signal=SignalSpec("multi_signal", {"entry_rule": {}, "exit_rule": {}}),
    )
    result = ExperimentResult(
        experiment_id="exp_1",
        status="completed",
        spec=spec,
        spec_hash=spec.spec_hash(),
        data_snapshot_id="snapshot",
        split_plan_id="split",
        cost_model_id=spec.exec_config.cost_model_id(),
        aggregate_metrics={"return_pct": 2.0, "trade_count": 4.0},
        fold_results=tuple(),
        robustness_checks={},
        promotion_stage="candidate",
    )

    summary = _loop_experiment_summary(
        result,
        generator_kind="frontier_neighborhood",
        artifact_paths={"result": "/tmp/result.json"},
    )

    assert summary == {
        "experiment_id": "exp_1",
        "family": "multi_signal",
        "promotion_stage": "candidate",
        "generator_kind": "frontier_neighborhood",
        "shape_key": "multi_signal",
        "aggregate_metrics": {"return_pct": 2.0, "trade_count": 4.0},
        "artifact_paths": {"result": "/tmp/result.json"},
    }


def test_write_loop_outputs_uses_settings_report_conventions(tmp_path: Path, monkeypatch) -> None:
    calls = {}
    sentinel = SimpleNamespace(loop_json_path="loop.json")

    def fake_write_loop_run_outputs(loop_payload, paths):
        calls["loop_payload"] = loop_payload
        calls["paths"] = paths
        return sentinel

    monkeypatch.setattr("trader.cli.loop_cmd.write_loop_run_outputs", fake_write_loop_run_outputs)
    settings = Settings(
        database_path=tmp_path / "market.db",
        research_dir=tmp_path / "research",
        ledger_path=tmp_path / "research" / "ledger.db",
        artifacts_dir=tmp_path / "research" / "artifacts",
        reports_dir=tmp_path / "research" / "reports",
    )

    output = _write_loop_outputs({"loop_run_id": "run_1"}, settings)

    assert output is sentinel
    assert calls["loop_payload"] == {"loop_run_id": "run_1"}
    assert calls["paths"].reports_dir == settings.reports_dir
    assert calls["paths"].artifacts_dir == settings.artifacts_dir
