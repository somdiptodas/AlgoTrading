from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

from trader.evaluation.runner import ExperimentResult, FoldResult
from trader.strategies.spec import FilterSpec, StrategySpec


@dataclass(frozen=True)
class ReportEntry:
    label: str
    detail: str | None = None
    metrics: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class ReportSectionContext:
    title: str
    summary: str | None = None
    bullets: tuple[str, ...] = ()
    metrics: tuple[tuple[str, str], ...] = ()
    entries: tuple[ReportEntry, ...] = ()


ContextInput = ReportSectionContext | str | Sequence[str] | Mapping[str, object] | None

_PRIMARY_METRIC_ORDER = (
    "return_pct",
    "annualized_sharpe",
    "sharpe_like",
    "max_drawdown_pct",
    "exposure_pct",
    "trade_count",
    "win_rate_pct",
    "profit_factor",
    "avg_trade_pct",
    "delta_always_flat_return_pct",
    "delta_regular_session_open_to_close_long_return_pct",
    "delta_session_long_flat_at_close_return_pct",
    "delta_randomized_entry_same_exposure_return_pct",
    "delta_buy_and_hold_return_pct",
    "delta_always_flat_annualized_sharpe",
    "delta_regular_session_open_to_close_long_annualized_sharpe",
    "delta_session_long_flat_at_close_annualized_sharpe",
    "delta_randomized_entry_same_exposure_annualized_sharpe",
    "delta_buy_and_hold_annualized_sharpe",
    "delta_always_flat_sharpe_like",
    "delta_regular_session_open_to_close_long_sharpe_like",
    "delta_session_long_flat_at_close_sharpe_like",
    "delta_randomized_entry_same_exposure_sharpe_like",
    "delta_buy_and_hold_sharpe_like",
)

_ROBUSTNESS_ORDER = (
    "fold_consistency_pass",
    "positive_fold_ratio",
    "regime_pass",
    "monthly_concentration_pct",
    "positive_monthly_pnl_concentration_pct",
    "loss_monthly_pnl_concentration_pct",
    "positive_monthly_pnl_present",
    "monthly_pnl_month_count",
    "neighborhood_pass",
    "neighborhood_gap_pct",
    "neighborhood_median_return_pct",
    "drawdown_pass",
)

_METRIC_LABELS = {
    "return_pct": "Return",
    "annualized_sharpe": "Annualized Sharpe",
    "sharpe_like": "Sharpe-like",
    "max_drawdown_pct": "Max drawdown",
    "exposure_pct": "Exposure",
    "trade_count": "Trade count",
    "win_rate_pct": "Win rate",
    "profit_factor": "Profit factor",
    "avg_trade_pct": "Average trade",
    "delta_always_flat_return_pct": "Return vs always flat",
    "delta_regular_session_open_to_close_long_return_pct": "Return vs open-to-close long",
    "delta_session_long_flat_at_close_return_pct": "Return vs session-long flat-at-close",
    "delta_randomized_entry_same_exposure_return_pct": "Return vs randomized matched exposure",
    "delta_buy_and_hold_return_pct": "Return vs buy and hold",
    "delta_always_flat_annualized_sharpe": "Annualized Sharpe vs always flat",
    "delta_regular_session_open_to_close_long_annualized_sharpe": "Annualized Sharpe vs open-to-close long",
    "delta_session_long_flat_at_close_annualized_sharpe": "Annualized Sharpe vs session-long flat-at-close",
    "delta_randomized_entry_same_exposure_annualized_sharpe": "Annualized Sharpe vs randomized matched exposure",
    "delta_buy_and_hold_annualized_sharpe": "Annualized Sharpe vs buy and hold",
    "delta_always_flat_sharpe_like": "Sharpe-like vs always flat",
    "delta_regular_session_open_to_close_long_sharpe_like": "Sharpe-like vs open-to-close long",
    "delta_session_long_flat_at_close_sharpe_like": "Sharpe-like vs session-long flat-at-close",
    "delta_randomized_entry_same_exposure_sharpe_like": "Sharpe-like vs randomized matched exposure",
    "delta_buy_and_hold_sharpe_like": "Sharpe-like vs buy and hold",
    "fold_consistency_pass": "Fold consistency pass",
    "positive_fold_ratio": "Positive fold ratio",
    "regime_pass": "Regime pass",
    "monthly_concentration_pct": "Monthly PnL concentration",
    "positive_monthly_pnl_concentration_pct": "Positive monthly PnL concentration",
    "loss_monthly_pnl_concentration_pct": "Loss monthly PnL concentration",
    "positive_monthly_pnl_present": "Positive monthly PnL present",
    "monthly_pnl_month_count": "Monthly PnL months",
    "neighborhood_pass": "Neighborhood pass",
    "neighborhood_gap_pct": "Neighborhood gap",
    "neighborhood_median_return_pct": "Neighborhood median return",
    "drawdown_pass": "Drawdown pass",
}

_SIGNED_PERCENT_METRICS = {
    "return_pct",
    "avg_trade_pct",
    "delta_always_flat_return_pct",
    "delta_regular_session_open_to_close_long_return_pct",
    "delta_session_long_flat_at_close_return_pct",
    "delta_randomized_entry_same_exposure_return_pct",
    "delta_buy_and_hold_return_pct",
    "neighborhood_gap_pct",
    "neighborhood_median_return_pct",
}


def render_experiment_report(
    result: ExperimentResult,
    *,
    critique: ContextInput = None,
    frontier: ContextInput = None,
) -> str:
    if result.status.lower() != "completed":
        raise ValueError("render_experiment_report requires a completed ExperimentResult")

    sections = [
        _render_header(result),
        _render_identity(result),
        _render_strategy(result.spec),
        _render_metric_block("Aggregate Metrics", result.aggregate_metrics, _PRIMARY_METRIC_ORDER),
        _render_metric_block("Robustness Checks", result.robustness_checks, _ROBUSTNESS_ORDER),
        _render_holdout(result.holdout_result),
        _render_folds(result.fold_results),
        _render_context(_normalize_context("Frontier Context", frontier)),
        _render_context(_normalize_context("Critique", critique)),
    ]
    return "\n\n".join(section for section in sections if section).strip() + "\n"


def _render_header(result: ExperimentResult) -> str:
    instrument = f"{result.spec.instrument} {result.spec.multiplier}-{result.spec.timespan}"
    return "\n".join(
        [
            f"# Experiment `{result.experiment_id}`",
            "",
            (
                f"Completed walk-forward evaluation for `{result.spec.name}` on `{instrument}`. "
                f"Promotion stage: `{result.promotion_stage}`."
            ),
        ]
    )


def _render_identity(result: ExperimentResult) -> str:
    lines = [
        "## Identity",
        f"- Status: `{result.status}`",
        f"- Experiment ID: `{result.experiment_id}`",
        f"- Spec hash: `{result.spec_hash}`",
        f"- Data snapshot: `{result.data_snapshot_id}`",
        f"- Split plan: `{result.split_plan_id}`",
        f"- Cost model: `{result.cost_model_id}`",
        f"- Promotion stage: `{result.promotion_stage}`",
    ]
    return "\n".join(lines)


def _render_strategy(spec: StrategySpec) -> str:
    filters = ", ".join(_format_filter(filter_spec) for filter_spec in spec.filters) if spec.filters else "none"
    features = ", ".join(f"`{name}`" for name in spec.feature_set) if spec.feature_set else "none"
    tags = ", ".join(f"`{tag}`" for tag in spec.tags) if spec.tags else "none"
    lines = [
        "## Strategy",
        f"- Name: `{spec.name}`",
        f"- Signal: `{_format_component(spec.signal.name, spec.signal.params)}`",
        f"- Sizing: `{_format_component(spec.sizing.name, spec.sizing.params)}`",
        f"- Filters: {filters}",
        (
            "- Execution: "
            f"initial_cash={spec.exec_config.initial_cash:,.2f}, "
            f"commission_per_order={spec.exec_config.commission_per_order:,.2f}, "
            f"slippage_bps={spec.exec_config.slippage_bps:.2f}, "
            f"regular_session_only={spec.exec_config.regular_session_only}, "
            f"flat_at_close={spec.exec_config.flat_at_close}"
        ),
        f"- Feature set: {features}",
        f"- Tags: {tags}",
        f"- Seed: `{spec.seed}`",
    ]
    return "\n".join(lines)


def _render_metric_block(title: str, metrics: Mapping[str, float | bool], preferred_order: Sequence[str]) -> str:
    if not metrics:
        return ""
    ordered_keys = [key for key in preferred_order if key in metrics]
    ordered_keys.extend(sorted(key for key in metrics if key not in ordered_keys))
    lines = [f"## {title}"]
    for key in ordered_keys:
        lines.append(f"- {_label_for_metric(key)}: {_format_metric_value(key, metrics[key])}")
    return "\n".join(lines)


def _render_folds(folds: Sequence[FoldResult]) -> str:
    if not folds:
        return ""
    sections = ["## Fold Results"]
    for fold in folds:
        sections.extend(
            [
                f"### Fold `{fold.fold_id}`",
                f"- Train window: `{fold.train_start_utc}` to `{fold.train_end_utc}`",
                f"- Test window: `{fold.test_start_utc}` to `{fold.test_end_utc}`",
                "- Metrics: "
                + ", ".join(
                    [
                        f"return={_format_metric_value('return_pct', fold.metrics.get('return_pct', 0.0))}",
                        f"sharpe={_format_metric_value('sharpe_like', fold.metrics.get('sharpe_like', 0.0))}",
                        f"drawdown={_format_metric_value('max_drawdown_pct', fold.metrics.get('max_drawdown_pct', 0.0))}",
                        f"trades={_format_metric_value('trade_count', fold.metrics.get('trade_count', 0.0))}",
                    ]
                ),
                "- Baselines: "
                + ", ".join(
                    f"{baseline} return={_format_metric_value('return_pct', values.get('return_pct', 0.0))}"
                    for baseline, values in sorted(fold.baseline_metrics.items())
                ),
                "- Baseline deltas: "
                + ", ".join(
                    f"{_label_for_metric(name)}={_format_metric_value(name, value)}"
                    for name, value in sorted(fold.baseline_deltas.items())
                ),
                (
                    f"- Backtest: final_cash={fold.backtest.final_cash:,.2f}, "
                    f"trades={len(fold.backtest.trades)}, "
                    f"bars={len(fold.backtest.bars)}"
                ),
                f"- Warnings: {', '.join(fold.warnings) if fold.warnings else 'none'}",
            ]
        )
    return "\n".join(sections)


def _render_context(context: ReportSectionContext | None) -> str:
    if context is None:
        return ""
    lines = [f"## {context.title}"]
    if context.summary:
        lines.extend(["", context.summary])
    if context.metrics:
        lines.extend([f"- {label}: {value}" for label, value in context.metrics])
    if context.bullets:
        lines.extend([f"- {bullet}" for bullet in context.bullets])
    for entry in context.entries:
        lines.append(f"### {entry.label}")
        if entry.detail:
            lines.append(entry.detail)
        for label, value in entry.metrics:
            lines.append(f"- {label}: {value}")
    return "\n".join(lines)


def _render_holdout(holdout: FoldResult | None) -> str:
    if holdout is None:
        return ""
    lines = ["## Holdout"]
    lines.append(
        "- "
        + ", ".join(
            (
                f"test={holdout.test_start_utc} to {holdout.test_end_utc}",
                f"return={_format_metric_value('return_pct', holdout.metrics.get('return_pct', 0.0))}",
                f"annualized_sharpe={_format_metric_value('annualized_sharpe', holdout.metrics.get('annualized_sharpe', 0.0))}",
                f"drawdown={_format_metric_value('max_drawdown_pct', holdout.metrics.get('max_drawdown_pct', 0.0))}",
                f"trades={_format_metric_value('trade_count', holdout.metrics.get('trade_count', 0.0))}",
                f"bars={len(holdout.backtest.bars)}",
            )
        )
    )
    return "\n".join(lines)


def _normalize_context(default_title: str, value: ContextInput) -> ReportSectionContext | None:
    if value is None:
        return None
    if isinstance(value, ReportSectionContext):
        return value
    if isinstance(value, str):
        return ReportSectionContext(title=default_title, summary=value)
    if isinstance(value, Mapping):
        title = str(value.get("title") or default_title)
        summary = _optional_text(value.get("summary") or value.get("body") or value.get("note"))
        bullets = _coerce_text_list(
            value.get("bullets")
            or value.get("items")
            or value.get("points")
            or value.get("risks")
            or value.get("next_steps")
        )
        metrics = _coerce_metric_pairs(value.get("metrics"))
        entries = _coerce_entries(value.get("entries"))
        return ReportSectionContext(
            title=title,
            summary=summary,
            bullets=bullets,
            metrics=metrics,
            entries=entries,
        )
    return ReportSectionContext(title=default_title, bullets=_coerce_text_list(value))


def _coerce_entries(raw_entries: object) -> tuple[ReportEntry, ...]:
    if raw_entries is None:
        return ()
    if isinstance(raw_entries, Sequence) and not isinstance(raw_entries, (str, bytes)):
        entries: list[ReportEntry] = []
        for item in raw_entries:
            if isinstance(item, ReportEntry):
                entries.append(item)
                continue
            if isinstance(item, Mapping):
                entries.append(
                    ReportEntry(
                        label=str(item.get("label") or item.get("title") or "Entry"),
                        detail=_optional_text(item.get("detail") or item.get("summary") or item.get("note")),
                        metrics=_coerce_metric_pairs(item.get("metrics")),
                    )
                )
                continue
            if item is not None:
                entries.append(ReportEntry(label=str(item)))
        return tuple(entries)
    return (ReportEntry(label=str(raw_entries)),)


def _coerce_metric_pairs(raw_metrics: object) -> tuple[tuple[str, str], ...]:
    if raw_metrics is None:
        return ()
    if isinstance(raw_metrics, Mapping):
        return tuple((str(key), _format_generic_value(value)) for key, value in raw_metrics.items())
    if isinstance(raw_metrics, Sequence) and not isinstance(raw_metrics, (str, bytes)):
        pairs: list[tuple[str, str]] = []
        for item in raw_metrics:
            if (
                isinstance(item, Sequence)
                and not isinstance(item, (str, bytes))
                and len(item) == 2
            ):
                label, value = item
                pairs.append((str(label), _format_generic_value(value)))
        return tuple(pairs)
    return ()


def _coerce_text_list(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(str(item) for item in value if item is not None)
    return (str(value),)


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _format_component(name: str, params: Mapping[str, object]) -> str:
    if not params:
        return name
    args = ", ".join(f"{key}={value}" for key, value in sorted(params.items()))
    return f"{name}({args})"


def _format_filter(filter_spec: FilterSpec) -> str:
    return f"`{_format_component(filter_spec.name, filter_spec.params)}`"


def _label_for_metric(name: str) -> str:
    return _METRIC_LABELS.get(name, name.replace("_", " ").title())


def _format_metric_value(name: str, value: float | bool) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if name == "trade_count":
        return f"{int(round(value)):,}"
    if name.endswith("_pct"):
        signed = name in _SIGNED_PERCENT_METRICS
        return f"{value:+.2f}%" if signed else f"{value:.2f}%"
    if name in {"annualized_sharpe", "sharpe_like", "profit_factor", "positive_fold_ratio"}:
        return _format_number(value)
    return _format_number(value)


def _format_generic_value(value: object) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return _format_number(value)
    return str(value)


def _format_number(value: float, *, signed: bool = False) -> str:
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    if math.isnan(value):
        return "nan"
    if signed:
        return f"{value:+.2f}"
    return f"{value:.2f}"
