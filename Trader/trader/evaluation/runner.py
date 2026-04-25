from __future__ import annotations

from calendar import monthrange
from dataclasses import dataclass, replace
from datetime import datetime, time
from hashlib import sha256
from typing import Sequence

from trader.data.models import MarketBar
from trader.data.view import DataSlice, DataView
from trader.evaluation.baselines import (
    always_flat,
    baseline_deltas,
    buy_and_hold,
    evaluate_baselines,
    randomized_entry_same_exposure,
    regular_session_open_to_close_long,
    session_long_flat_at_close,
)
from trader.evaluation.data_quality import validate_bars
from trader.evaluation.metrics import aggregate_metric_dicts, annualized_sharpe_for_backtests, calculate_metrics
from trader.evaluation.promotion import promotion_stage
from trader.evaluation.robustness import RobustnessResult, assess_robustness
from trader.evaluation.splits import Fold, build_walk_forward_folds
from trader.execution.engine import BacktestResult, run_long_only_engine
from trader.features.pipeline import FeatureCache, feature_cache_context
from trader.strategies.registry import StrategyRegistry
from trader.strategies.spec import StrategySpec

DEFAULT_LOCKED_HOLDOUT_MONTHS = 3
_HOLDOUT_STAGES = {"candidate"}
_STAGE_A_MIN_TRADES = 10.0
_STAGE_A_MAX_TRADES = 400.0
_STAGE_A_MAX_DRAWDOWN_PCT = 25.0
_STAGE_A_MIN_EXPOSURE_PCT = 1.0
_STAGE_A_MAX_EXPOSURE_PCT = 90.0


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
    holdout_result: FoldResult | None = None


@dataclass(frozen=True)
class EvaluationPreview:
    spec: StrategySpec
    data_slice: DataSlice
    split_plan_id: str
    folds: tuple[Fold, ...]
    required_history: int
    cost_model_id: str
    evaluation_key: str
    holdout_bars: tuple[MarketBar, ...] = tuple()
    holdout_snapshot_id: str | None = None


class EvaluationRunner:
    def __init__(self, data_view: DataView, registry: StrategyRegistry) -> None:
        self.data_view = data_view
        self.registry = registry
        self._data_slice_cache: dict[tuple[object, ...], DataSlice] = {}
        self._split_cache: dict[tuple[str, int, int, int], tuple[str, tuple[Fold, ...]]] = {}
        self._fold_result_cache: dict[tuple[object, ...], FoldResult] = {}
        self._baseline_cache: dict[tuple[object, ...], dict[str, dict[str, float]]] = {}
        self._indicator_cache: FeatureCache = {}

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
        locked_holdout_months: int | None = None,
    ) -> ExperimentResult:
        preview = self.preview_walk_forward(
            spec,
            num_folds=num_folds,
            embargo_bars=embargo_bars,
            start_ms=start_ms,
            end_ms=end_ms,
            locked_holdout_months=locked_holdout_months,
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
        experiment_id = experiment_id or sha256(preview.evaluation_key.encode("utf-8")).hexdigest()[:16]
        stage_a = self._evaluate_stage_a(preview) if include_robustness else None
        if stage_a is not None:
            stage_a_fold, stage_a_metrics = stage_a
            reject_reasons = self._stage_a_reject_reasons(stage_a_metrics)
            if reject_reasons:
                stage_a_metrics = dict(stage_a_metrics)
                stage_a_metrics["stage_a_pass"] = 0.0
                stage_a_metrics["stage_a_reject_count"] = float(len(reject_reasons))
                return ExperimentResult(
                    experiment_id=experiment_id,
                    status="completed",
                    spec=preview.spec,
                    spec_hash=preview.spec.spec_hash(),
                    data_snapshot_id=preview.data_slice.snapshot_id,
                    split_plan_id=preview.split_plan_id,
                    cost_model_id=preview.cost_model_id,
                    aggregate_metrics=stage_a_metrics,
                    fold_results=(stage_a_fold,),
                    robustness_checks={
                        "stage_a_pass": False,
                        **{f"stage_a_reject_{reason}": True for reason in reject_reasons},
                    },
                    promotion_stage="exploratory",
                    holdout_result=None,
                )

        fold_results, aggregate_metrics = self._evaluate_preview_folds(preview.spec, preview)
        aggregate_metrics = dict(aggregate_metrics)
        if stage_a is not None:
            aggregate_metrics["stage_a_pass"] = 1.0
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
        stage = promotion_stage(aggregate_metrics, robustness.checks)
        if stage != "exploratory":
            fold_results = tuple(
                self._add_cost_scenario_metrics(
                    preview.spec,
                    fold,
                    preview.data_slice.bars,
                    fold_result,
                    data_snapshot_id=preview.data_slice.snapshot_id,
                )
                for fold, fold_result in zip(preview.folds, fold_results)
            )
            aggregate_metrics = self._aggregate_fold_results(fold_results)
            if stage_a is not None:
                aggregate_metrics["stage_a_pass"] = 1.0
        holdout_result = (
            self._evaluate_holdout(preview.spec, preview)
            if stage in _HOLDOUT_STAGES and preview.holdout_bars
            else None
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
            promotion_stage=stage,
            holdout_result=holdout_result,
        )

    def preview_walk_forward(
        self,
        spec: StrategySpec,
        *,
        num_folds: int = 3,
        embargo_bars: int = 1,
        start_ms: int | None = None,
        end_ms: int | None = None,
        locked_holdout_months: int | None = None,
    ) -> EvaluationPreview:
        validated = self.registry.validate_spec(spec)
        full_slice = self._load_data_slice(
            validated,
            start_ms=start_ms,
            end_ms=end_ms,
        )
        data_slice, holdout_bars, holdout_snapshot_id = self._split_research_and_holdout(
            full_slice,
            embargo_bars=embargo_bars,
            locked_holdout_months=locked_holdout_months,
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
            holdout_bars=holdout_bars,
            holdout_snapshot_id=holdout_snapshot_id,
        )

    def evaluation_key_for_spec(
        self,
        spec: StrategySpec,
        *,
        num_folds: int = 3,
        embargo_bars: int = 1,
        start_ms: int | None = None,
        end_ms: int | None = None,
        locked_holdout_months: int | None = None,
    ) -> str:
        validated = self.registry.validate_spec(spec)
        full_slice = self._load_data_slice(
            validated,
            start_ms=start_ms,
            end_ms=end_ms,
        )
        data_slice, _, _ = self._split_research_and_holdout(
            full_slice,
            embargo_bars=embargo_bars,
            locked_holdout_months=locked_holdout_months,
        )
        split_plan_id, _ = self._load_split_plan(
            data_slice,
            required_history=self.registry.required_history(validated),
            num_folds=num_folds,
            embargo_bars=embargo_bars,
        )
        return self.evaluation_key(
            validated.spec_hash(),
            data_slice.snapshot_id,
            split_plan_id,
            validated.exec_config.cost_model_id(),
        )

    def _evaluate_stage_a(self, preview: EvaluationPreview) -> tuple[FoldResult, dict[str, float]] | None:
        if not preview.folds:
            return None
        fold = preview.folds[-1]
        fold_result = self._evaluate_fast_fold(
            preview.spec,
            fold,
            preview.data_slice.bars,
            data_snapshot_id=preview.data_slice.snapshot_id,
        )
        metrics = dict(fold_result.metrics)
        metrics["annualized_sharpe"] = annualized_sharpe_for_backtests((fold_result.backtest,))
        return fold_result, metrics

    def _evaluate_fast_fold(
        self,
        spec: StrategySpec,
        fold: Fold,
        bars: tuple[MarketBar, ...],
        *,
        data_snapshot_id: str | None = None,
    ) -> FoldResult:
        train_bars = tuple(bars[fold.train_start_idx : fold.train_end_idx + 1])
        test_bars = tuple(bars[fold.test_start_idx : fold.test_end_idx + 1])
        warnings = self._quality_warnings(spec, test_bars)
        regime = self._generate_fold_regime(spec, fold, train_bars, test_bars, data_snapshot_id=data_snapshot_id)
        sizing_fraction = self.registry.compute_sizing_fraction(spec)
        result = run_long_only_engine(test_bars, regime, spec.exec_config, sizing_fraction)
        metrics = calculate_metrics(result)
        return FoldResult(
            fold_id=fold.fold_id,
            train_start_utc=fold.train_start_utc,
            train_end_utc=fold.train_end_utc,
            test_start_utc=fold.test_start_utc,
            test_end_utc=fold.test_end_utc,
            metrics=metrics,
            baseline_metrics={},
            baseline_deltas={},
            warnings=warnings,
            backtest=result,
        )

    def _stage_a_reject_reasons(self, metrics: dict[str, float]) -> tuple[str, ...]:
        reasons: list[str] = []
        if metrics.get("return_pct", 0.0) <= 0.0:
            reasons.append("non_positive_return")
        trade_count = metrics.get("trade_count", 0.0)
        if trade_count < _STAGE_A_MIN_TRADES or trade_count > _STAGE_A_MAX_TRADES:
            reasons.append("trade_count")
        if metrics.get("max_drawdown_pct", 0.0) > _STAGE_A_MAX_DRAWDOWN_PCT:
            reasons.append("drawdown")
        exposure_pct = metrics.get("exposure_pct", 0.0)
        if exposure_pct < _STAGE_A_MIN_EXPOSURE_PCT or exposure_pct > _STAGE_A_MAX_EXPOSURE_PCT:
            reasons.append("exposure")
        return tuple(reasons)

    def _evaluate_fold(
        self,
        spec: StrategySpec,
        fold: Fold,
        bars: tuple[MarketBar, ...],
        *,
        data_snapshot_id: str | None = None,
    ) -> FoldResult:
        cache_key = (
            spec.spec_hash(),
            data_snapshot_id,
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
        warnings = self._quality_warnings(spec, test_bars)
        regime = self._generate_fold_regime(spec, fold, train_bars, test_bars, data_snapshot_id=data_snapshot_id)
        sizing_fraction = self.registry.compute_sizing_fraction(spec)
        result = run_long_only_engine(test_bars, regime, spec.exec_config, sizing_fraction)
        metrics = calculate_metrics(result)
        baselines = self._evaluate_baselines(
            spec,
            fold,
            test_bars,
            result.trades,
            data_snapshot_id=data_snapshot_id,
        )
        fold_result = FoldResult(
            fold_id=fold.fold_id,
            train_start_utc=fold.train_start_utc,
            train_end_utc=fold.train_end_utc,
            test_start_utc=fold.test_start_utc,
            test_end_utc=fold.test_end_utc,
            metrics=metrics,
            baseline_metrics=baselines,
            baseline_deltas=baseline_deltas(metrics, baselines),
            warnings=warnings,
            backtest=result,
        )
        self._fold_result_cache[cache_key] = fold_result
        return fold_result

    def _generate_fold_regime(
        self,
        spec: StrategySpec,
        fold: Fold,
        train_bars: tuple[MarketBar, ...],
        test_bars: tuple[MarketBar, ...],
        *,
        data_snapshot_id: str | None = None,
    ) -> list[bool]:
        if not hasattr(self, "_indicator_cache"):
            self._indicator_cache = {}
        scope = (
            data_snapshot_id,
            fold.fold_id,
            fold.train_start_utc,
            fold.train_end_utc,
            fold.test_start_utc,
            fold.test_end_utc,
            len(train_bars),
            len(test_bars),
        )
        with feature_cache_context(self._indicator_cache, scope):
            return self.registry.generate_regime(spec, train_bars, test_bars)

    def _evaluate_baselines(
        self,
        spec: StrategySpec,
        fold: Fold,
        test_bars: tuple[MarketBar, ...],
        strategy_trades: tuple,
        data_snapshot_id: str | None = None,
    ) -> dict[str, dict[str, float]]:
        fixed_baselines = {
            name: dict(metrics)
            for name, metrics in self._fixed_baselines(
                spec,
                fold,
                test_bars,
                data_snapshot_id=data_snapshot_id,
            ).items()
        }
        fixed_baselines["randomized_entry_same_exposure"] = randomized_entry_same_exposure(
            test_bars,
            spec.exec_config,
            strategy_trades,
            seed_material=f"{spec.spec_hash()}|{fold.fold_id}|{fold.test_start_utc}|{fold.test_end_utc}",
        )
        return fixed_baselines

    def _fixed_baselines(
        self,
        spec: StrategySpec,
        fold: Fold,
        test_bars: tuple[MarketBar, ...],
        data_snapshot_id: str | None = None,
    ) -> dict[str, dict[str, float]]:
        if not hasattr(self, "_baseline_cache"):
            self._baseline_cache = {}
        cache_key = (
            data_snapshot_id,
            fold.fold_id,
            fold.test_start_utc,
            fold.test_end_utc,
            len(test_bars),
            spec.exec_config.cost_model_id(),
        )
        cached = self._baseline_cache.get(cache_key)
        if cached is not None:
            return {name: dict(metrics) for name, metrics in cached.items()}
        baselines = {
            "always_flat": always_flat(spec.exec_config.initial_cash),
            "buy_and_hold": buy_and_hold(test_bars, spec.exec_config.initial_cash),
            "regular_session_open_to_close_long": regular_session_open_to_close_long(test_bars, spec.exec_config),
            "session_long_flat_at_close": session_long_flat_at_close(test_bars, spec.exec_config),
        }
        self._baseline_cache[cache_key] = baselines
        return {name: dict(metrics) for name, metrics in baselines.items()}

    def _add_cost_scenario_metrics(
        self,
        spec: StrategySpec,
        fold: Fold,
        bars: tuple[MarketBar, ...],
        fold_result: FoldResult,
        *,
        data_snapshot_id: str | None = None,
    ) -> FoldResult:
        train_bars = tuple(bars[fold.train_start_idx : fold.train_end_idx + 1])
        test_bars = tuple(bars[fold.test_start_idx : fold.test_end_idx + 1])
        regime = self._generate_fold_regime(spec, fold, train_bars, test_bars, data_snapshot_id=data_snapshot_id)
        sizing_fraction = self.registry.compute_sizing_fraction(spec)
        metrics = dict(fold_result.metrics)
        metrics.update(self._cost_scenario_metrics(spec, test_bars, regime, sizing_fraction, metrics))
        return replace(fold_result, metrics=metrics)

    def _evaluate_holdout(self, spec: StrategySpec, preview: EvaluationPreview) -> FoldResult:
        train_bars = preview.data_slice.bars
        test_bars = preview.holdout_bars
        warnings = self._quality_warnings(spec, test_bars)
        if not hasattr(self, "_indicator_cache"):
            self._indicator_cache = {}
        scope = (
            preview.data_slice.snapshot_id,
            preview.holdout_snapshot_id,
            "holdout",
            test_bars[0].timestamp_utc,
            test_bars[-1].timestamp_utc,
            len(train_bars),
            len(test_bars),
        )
        with feature_cache_context(self._indicator_cache, scope):
            regime = self.registry.generate_regime(spec, train_bars, test_bars)
        sizing_fraction = self.registry.compute_sizing_fraction(spec)
        result = run_long_only_engine(test_bars, regime, spec.exec_config, sizing_fraction)
        metrics = calculate_metrics(result)
        metrics.update(self._cost_scenario_metrics(spec, test_bars, regime, sizing_fraction, metrics))
        baselines = evaluate_baselines(
            test_bars,
            spec.exec_config,
            strategy_trades=result.trades,
            seed_material=f"{spec.spec_hash()}|holdout|{test_bars[0].timestamp_utc}|{test_bars[-1].timestamp_utc}",
        )
        return FoldResult(
            fold_id="holdout",
            train_start_utc=train_bars[0].timestamp_utc if train_bars else "",
            train_end_utc=train_bars[-1].timestamp_utc if train_bars else "",
            test_start_utc=test_bars[0].timestamp_utc,
            test_end_utc=test_bars[-1].timestamp_utc,
            metrics=metrics,
            baseline_metrics=baselines,
            baseline_deltas=baseline_deltas(metrics, baselines),
            warnings=warnings,
            backtest=result,
        )

    def _cost_scenario_metrics(
        self,
        spec: StrategySpec,
        bars: tuple[MarketBar, ...],
        regime: Sequence[bool],
        sizing_fraction: float,
        metrics: dict[str, float],
    ) -> dict[str, float]:
        base_config = spec.exec_config
        zero_cost = replace(
            base_config,
            commission_per_order=0.0,
            commission_per_share=0.0,
            slippage_bps=0.0,
            spread_bps=0.0,
        )
        slippage_stress = replace(base_config, slippage_bps=base_config.slippage_bps + 2.0)
        spread_stress = replace(base_config, spread_bps=base_config.spread_bps + 2.0)
        zero_return = calculate_metrics(
            run_long_only_engine(bars, regime, zero_cost, sizing_fraction)
        )["return_pct"]
        slippage_return = calculate_metrics(
            run_long_only_engine(bars, regime, slippage_stress, sizing_fraction)
        )["return_pct"]
        spread_return = calculate_metrics(
            run_long_only_engine(bars, regime, spread_stress, sizing_fraction)
        )["return_pct"]
        base_return = metrics["return_pct"]
        return {
            "cost_drag_return_pct": zero_return - base_return,
            "cost_scenario_slippage_plus_2bps_return_pct": slippage_return,
            "cost_scenario_slippage_plus_2bps_delta_pct": slippage_return - base_return,
            "cost_scenario_spread_plus_2bps_return_pct": spread_return,
            "cost_scenario_spread_plus_2bps_delta_pct": spread_return - base_return,
        }

    def _evaluate_preview_folds(
        self,
        spec: StrategySpec,
        preview: EvaluationPreview,
    ) -> tuple[tuple[FoldResult, ...], dict[str, float]]:
        fold_results = tuple(
            self._evaluate_fold(
                spec,
                fold,
                preview.data_slice.bars,
                data_snapshot_id=preview.data_slice.snapshot_id,
            )
            for fold in preview.folds
        )
        return fold_results, self._aggregate_fold_results(fold_results)

    def _aggregate_fold_results(self, fold_results: Sequence[FoldResult]) -> dict[str, float]:
        aggregate_metrics = aggregate_metric_dicts(
            [fold.metrics | fold.baseline_deltas for fold in fold_results],
            weights=[len(fold.backtest.bars) for fold in fold_results],
        )
        aggregate_metrics["annualized_sharpe"] = annualized_sharpe_for_backtests(
            [fold.backtest for fold in fold_results]
        )
        return aggregate_metrics

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

    def _quality_warnings(self, spec: StrategySpec, bars: tuple[MarketBar, ...]) -> tuple[str, ...]:
        raw_warnings: tuple[str, ...] = tuple()
        if bars and hasattr(self, "data_view"):
            start_ms, end_ms = _raw_warning_bounds(bars)
            raw_warnings = self.data_view.quality_warnings(
                ticker=spec.instrument,
                multiplier=spec.multiplier,
                timespan=spec.timespan,
                start_ms=start_ms,
                end_ms=end_ms,
                regular_session_only=spec.exec_config.regular_session_only,
            )
        bar_warnings = validate_bars(bars)
        return tuple(dict.fromkeys(raw_warnings + bar_warnings))

    def _split_research_and_holdout(
        self,
        data_slice: DataSlice,
        *,
        embargo_bars: int,
        locked_holdout_months: int | None = DEFAULT_LOCKED_HOLDOUT_MONTHS,
    ) -> tuple[DataSlice, tuple[MarketBar, ...], str | None]:
        if not data_slice.bars or locked_holdout_months is None or locked_holdout_months <= 0:
            return data_slice, tuple(), None
        cutoff = _subtract_months(data_slice.bars[-1].dt_local, locked_holdout_months)
        holdout_start_idx = len(data_slice.bars)
        for index, bar in enumerate(data_slice.bars):
            if bar.dt_local >= cutoff:
                holdout_start_idx = index
                break
        research_end_idx = holdout_start_idx - max(embargo_bars, 0)
        if research_end_idx <= 0 or holdout_start_idx == len(data_slice.bars):
            raise RuntimeError("Not enough pre-holdout bars for the locked holdout policy")
        research_bars = data_slice.bars[:research_end_idx]
        holdout_bars = data_slice.bars[holdout_start_idx:]
        research_slice = DataSlice(
            bars=research_bars,
            snapshot_id=DataView.snapshot_hash(research_bars),
            first_timestamp_utc=research_bars[0].timestamp_utc,
            last_timestamp_utc=research_bars[-1].timestamp_utc,
        )
        return research_slice, holdout_bars, DataView.snapshot_hash(holdout_bars)

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


def _subtract_months(value: datetime, months: int) -> datetime:
    month = value.month - months
    year = value.year
    while month <= 0:
        month += 12
        year -= 1
    day = min(value.day, monthrange(year, month)[1])
    return value.replace(year=year, month=month, day=day)


def _raw_warning_bounds(bars: tuple[MarketBar, ...]) -> tuple[int, int]:
    start_ms = bars[0].timestamp_ms
    end_ms = bars[-1].timestamp_ms
    first_local = bars[0].dt_local
    last_local = bars[-1].dt_local
    regular_start = time(9, 30)
    regular_second = time(9, 31)
    regular_penultimate = time(15, 58)
    regular_last = time(15, 59)
    if (
        last_local.timetz().replace(tzinfo=None) == regular_last
        and first_local.timetz().replace(tzinfo=None) == regular_second
    ):
        session_start = first_local.replace(hour=9, minute=30, second=0, microsecond=0)
        start_ms = int(session_start.timestamp() * 1000)
    if (
        first_local.timetz().replace(tzinfo=None) == regular_start
        and last_local.timetz().replace(tzinfo=None) == regular_penultimate
    ):
        session_end = last_local.replace(hour=15, minute=59, second=0, microsecond=0)
        end_ms = int(session_end.timestamp() * 1000)
    return start_ms, end_ms
