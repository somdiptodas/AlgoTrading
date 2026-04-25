from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Sequence

from trader.data.models import MarketBar
from trader.data.view import DataSlice, DataView
from trader.evaluation.baselines import baseline_deltas, evaluate_baselines
from trader.evaluation.metrics import aggregate_metric_dicts, annualized_sharpe_for_backtests, calculate_metrics
from trader.evaluation.promotion import promotion_stage
from trader.evaluation.robustness import RobustnessResult, assess_robustness
from trader.evaluation.splits import Fold, build_walk_forward_folds
from trader.execution.engine import BacktestResult, run_long_only_engine
from trader.strategies.registry import StrategyRegistry
from trader.strategies.spec import StrategySpec


@dataclass(frozen=True)
class FoldResult:
    fold_id: str
    train_start_utc: str
    train_end_utc: str
    test_start_utc: str
    test_end_utc: str
    metrics: dict[str, float]
    baseline_metrics: dict[str, dict[str, float]]
    baseline_deltas: dict[str, float]
    warnings: tuple[str, ...]
    backtest: BacktestResult


@dataclass(frozen=True)
class ExperimentResult:
    experiment_id: str
    status: str
    spec: StrategySpec
    spec_hash: str
    data_snapshot_id: str
    split_plan_id: str
    cost_model_id: str
    aggregate_metrics: dict[str, float]
    fold_results: tuple[FoldResult, ...]
    robustness_checks: dict[str, float | bool]
    promotion_stage: str


@dataclass(frozen=True)
class EvaluationPreview:
    spec: StrategySpec
    data_slice: DataSlice
    split_plan_id: str
    folds: tuple[Fold, ...]
    required_history: int
    cost_model_id: str
    evaluation_key: str


class EvaluationRunner:
    def __init__(self, data_view: DataView, registry: StrategyRegistry) -> None:
        self.data_view = data_view
        self.registry = registry
        self._data_slice_cache: dict[tuple[object, ...], DataSlice] = {}
        self._split_cache: dict[tuple[str, int, int, int], tuple[str, tuple[Fold, ...]]] = {}
        self._fold_result_cache: dict[tuple[str, str, str, str, int], FoldResult] = {}

    def evaluate_single_window(
        self,
        spec: StrategySpec,
        *,
        start_ms: int | None = None,
        end_ms: int | None = None,
    ) -> BacktestResult:
        validated = self.registry.validate_spec(spec)
        data_slice = self.data_view.slice(
            ticker=validated.instrument,
            multiplier=validated.multiplier,
            timespan=validated.timespan,
            start_ms=start_ms,
            end_ms=end_ms,
            regular_session_only=validated.exec_config.regular_session_only,
        )
        if len(data_slice.bars) <= self.registry.required_history(validated) + 1:
            raise RuntimeError("Not enough bars for the selected strategy lookback")
        regime = self.registry.generate_regime(validated, tuple(), data_slice.bars)
        sizing_fraction = self.registry.compute_sizing_fraction(validated)
        return run_long_only_engine(data_slice.bars, regime, validated.exec_config, sizing_fraction)

    def evaluate_walk_forward(
        self,
        spec: StrategySpec,
        *,
        num_folds: int = 3,
        embargo_bars: int = 1,
        start_ms: int | None = None,
        end_ms: int | None = None,
        include_robustness: bool = True,
        experiment_id: str | None = None,
    ) -> ExperimentResult:
        preview = self.preview_walk_forward(
            spec,
            num_folds=num_folds,
            embargo_bars=embargo_bars,
            start_ms=start_ms,
            end_ms=end_ms,
        )
        return self.evaluate_preview(
            preview,
            include_robustness=include_robustness,
            experiment_id=experiment_id,
        )

    def evaluate_preview(
        self,
        preview: EvaluationPreview,
        *,
        include_robustness: bool = True,
        experiment_id: str | None = None,
    ) -> ExperimentResult:
        fold_results, aggregate_metrics = self._evaluate_preview_folds(preview.spec, preview)
        experiment_id = experiment_id or sha256(preview.evaluation_key.encode("utf-8")).hexdigest()[:16]
        robustness = (
            assess_robustness(
                spec=preview.spec,
                aggregate_metrics=aggregate_metrics,
                fold_metrics=[fold.metrics for fold in fold_results],
                fold_backtests=[fold.backtest for fold in fold_results],
                registry=self.registry,
                neighbor_metric_fn=lambda neighbor_spec: self._evaluate_preview_folds(
                    neighbor_spec,
                    preview,
                )[1],
            )
            if include_robustness
            else RobustnessResult(checks={}, passed=False)
        )
        return ExperimentResult(
            experiment_id=experiment_id,
            status="completed",
            spec=preview.spec,
            spec_hash=preview.spec.spec_hash(),
            data_snapshot_id=preview.data_slice.snapshot_id,
            split_plan_id=preview.split_plan_id,
            cost_model_id=preview.cost_model_id,
            aggregate_metrics=aggregate_metrics,
            fold_results=fold_results,
            robustness_checks=robustness.checks,
            promotion_stage=promotion_stage(aggregate_metrics, robustness.checks),
        )

    def preview_walk_forward(
        self,
        spec: StrategySpec,
        *,
        num_folds: int = 3,
        embargo_bars: int = 1,
        start_ms: int | None = None,
        end_ms: int | None = None,
    ) -> EvaluationPreview:
        validated = self.registry.validate_spec(spec)
        data_slice = self._load_data_slice(
            validated,
            start_ms=start_ms,
            end_ms=end_ms,
        )
        required_history = self.registry.required_history(validated)
        split_plan_id, folds = self._load_split_plan(
            data_slice,
            required_history=required_history,
            num_folds=num_folds,
            embargo_bars=embargo_bars,
        )
        cost_model_id = validated.exec_config.cost_model_id()
        return EvaluationPreview(
            spec=validated,
            data_slice=data_slice,
            split_plan_id=split_plan_id,
            folds=folds,
            required_history=required_history,
            cost_model_id=cost_model_id,
            evaluation_key=self.evaluation_key(
                validated.spec_hash(),
                data_slice.snapshot_id,
                split_plan_id,
                cost_model_id,
            ),
        )

    def _evaluate_fold(self, spec: StrategySpec, fold: Fold, bars: tuple[MarketBar, ...]) -> FoldResult:
        cache_key = (
            spec.spec_hash(),
            fold.fold_id,
            fold.train_start_utc,
            fold.test_end_utc,
            len(bars),
        )
        cached = self._fold_result_cache.get(cache_key)
        if cached is not None:
            return cached
        train_bars = tuple(bars[fold.train_start_idx : fold.train_end_idx + 1])
        test_bars = tuple(bars[fold.test_start_idx : fold.test_end_idx + 1])
        regime = self.registry.generate_regime(spec, train_bars, test_bars)
        sizing_fraction = self.registry.compute_sizing_fraction(spec)
        result = run_long_only_engine(test_bars, regime, spec.exec_config, sizing_fraction)
        metrics = calculate_metrics(result)
        baselines = evaluate_baselines(test_bars, spec.exec_config.initial_cash)
        fold_result = FoldResult(
            fold_id=fold.fold_id,
            train_start_utc=fold.train_start_utc,
            train_end_utc=fold.train_end_utc,
            test_start_utc=fold.test_start_utc,
            test_end_utc=fold.test_end_utc,
            metrics=metrics,
            baseline_metrics=baselines,
            baseline_deltas=baseline_deltas(metrics, baselines),
            warnings=tuple(),
            backtest=result,
        )
        self._fold_result_cache[cache_key] = fold_result
        return fold_result

    def _evaluate_preview_folds(
        self,
        spec: StrategySpec,
        preview: EvaluationPreview,
    ) -> tuple[tuple[FoldResult, ...], dict[str, float]]:
        fold_results = tuple(self._evaluate_fold(spec, fold, preview.data_slice.bars) for fold in preview.folds)
        aggregate_metrics = aggregate_metric_dicts(
            [fold.metrics | fold.baseline_deltas for fold in fold_results],
            weights=[len(fold.backtest.bars) for fold in fold_results],
        )
        aggregate_metrics["annualized_sharpe"] = annualized_sharpe_for_backtests(
            [fold.backtest for fold in fold_results]
        )
        return fold_results, aggregate_metrics

    def evaluation_key(
        self,
        spec_hash: str,
        data_snapshot_id: str,
        split_plan_id: str,
        cost_model_id: str,
    ) -> str:
        payload = "|".join([spec_hash, data_snapshot_id, split_plan_id, cost_model_id])
        return sha256(payload.encode("utf-8")).hexdigest()

    def _load_data_slice(
        self,
        spec: StrategySpec,
        *,
        start_ms: int | None = None,
        end_ms: int | None = None,
    ) -> DataSlice:
        cache_key = (
            spec.instrument,
            spec.multiplier,
            spec.timespan,
            start_ms,
            end_ms,
            spec.exec_config.regular_session_only,
        )
        cached = self._data_slice_cache.get(cache_key)
        if cached is not None:
            return cached
        data_slice = self.data_view.slice(
            ticker=spec.instrument,
            multiplier=spec.multiplier,
            timespan=spec.timespan,
            start_ms=start_ms,
            end_ms=end_ms,
            regular_session_only=spec.exec_config.regular_session_only,
        )
        self._data_slice_cache[cache_key] = data_slice
        return data_slice

    def _load_split_plan(
        self,
        data_slice: DataSlice,
        *,
        required_history: int,
        num_folds: int,
        embargo_bars: int,
    ) -> tuple[str, tuple[Fold, ...]]:
        cache_key = (
            data_slice.snapshot_id,
            required_history,
            num_folds,
            embargo_bars,
        )
        cached = self._split_cache.get(cache_key)
        if cached is not None:
            return cached
        min_train_bars = max(required_history * 5, required_history + 50)
        split_plan = build_walk_forward_folds(
            data_slice.bars,
            num_folds=num_folds,
            min_train_bars=min_train_bars,
            embargo_bars=embargo_bars,
        )
        self._split_cache[cache_key] = split_plan
        return split_plan
