from __future__ import annotations

from pathlib import Path

from trader.evaluation.runner import ExperimentResult
from trader.evaluation.runner import FoldResult
from trader.ledger.entry import equity_sample_indices, experiment_result_to_payload, json_dumps, trade_to_payload

_EQUITY_SAMPLE_INTERVAL_MINUTES = 30


class ArtifactStore:
    def __init__(self, artifacts_dir: str | Path, reports_dir: str | Path) -> None:
        self.artifacts_dir = Path(artifacts_dir)
        self.reports_dir = Path(reports_dir)

    def write_experiment(
        self,
        result: ExperimentResult,
        *,
        report_markdown: str | None = None,
        critique: dict[str, object] | None = None,
        generator_kind: str | None = None,
    ) -> dict[str, str]:
        experiment_dir = self.artifacts_dir / result.experiment_id
        experiment_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        spec_path = experiment_dir / "spec.json"
        result_path = experiment_dir / "result.json"
        trades_path = experiment_dir / "trades.json"
        equity_path = experiment_dir / "equity.json"
        report_path = self.reports_dir / f"{result.experiment_id}.md"

        spec_path.write_text(result.spec.canonical_json(), encoding="utf-8")
        result_path.write_text(
            json_dumps(
                experiment_result_to_payload(result, equity_interval_minutes=_EQUITY_SAMPLE_INTERVAL_MINUTES),
                pretty=True,
            ),
            encoding="utf-8",
        )
        trades_path.write_text(json_dumps(self._trades_payload(result), pretty=True), encoding="utf-8")
        equity_path.write_text(json_dumps(self._equity_payload(result), pretty=True), encoding="utf-8")
        report_path.write_text(
            report_markdown if report_markdown is not None else self._default_report(result, critique),
            encoding="utf-8",
        )

        paths = {
            "spec": str(spec_path.resolve()),
            "result": str(result_path.resolve()),
            "trades": str(trades_path.resolve()),
            "equity": str(equity_path.resolve()),
            "report": str(report_path.resolve()),
        }
        manifest = {
            "experiment_id": result.experiment_id,
            "status": result.status,
            "spec_hash": result.spec_hash,
            "data_snapshot_id": result.data_snapshot_id,
            "split_plan_id": result.split_plan_id,
            "cost_model_id": result.cost_model_id,
            "promotion_stage": result.promotion_stage,
            "aggregate_metrics": dict(result.aggregate_metrics),
            "generator_kind": generator_kind,
            "artifact_paths": paths,
            "critique": critique,
        }
        manifest_path = experiment_dir / "manifest.json"
        manifest_path.write_text(json_dumps(manifest, pretty=True), encoding="utf-8")
        paths["manifest"] = str(manifest_path.resolve())
        return paths

    def write_experiment_bundle(
        self,
        result: ExperimentResult,
        *,
        report_markdown: str | None = None,
        critique: dict[str, object] | None = None,
        generator_kind: str | None = None,
    ) -> dict[str, str]:
        return self.write_experiment(
            result,
            report_markdown=report_markdown,
            critique=critique,
            generator_kind=generator_kind,
        )

    def _trades_payload(self, result: ExperimentResult) -> dict[str, object]:
        return {
            "experiment_id": result.experiment_id,
            "folds": [
                {
                    "fold_id": fold.fold_id,
                    "trades": [trade_to_payload(trade) for trade in fold.backtest.trades],
                }
                for fold in result.fold_results
            ],
        }

    def _equity_payload(self, result: ExperimentResult) -> dict[str, object]:
        return {
            "experiment_id": result.experiment_id,
            "sampling_interval_minutes": _EQUITY_SAMPLE_INTERVAL_MINUTES,
            "folds": [self._sampled_equity_payload(fold) for fold in result.fold_results],
        }

    def _sampled_equity_payload(self, fold: FoldResult) -> dict[str, object]:
        bars = fold.backtest.bars
        equity_curve = fold.backtest.equity_curve
        source_points = min(len(bars), len(equity_curve))
        sampled_indices = equity_sample_indices(bars, equity_curve, _EQUITY_SAMPLE_INTERVAL_MINUTES)
        return {
            "fold_id": fold.fold_id,
            "sampling_interval_minutes": _EQUITY_SAMPLE_INTERVAL_MINUTES,
            "source_points": source_points,
            "timestamps_utc": [bars[index].timestamp_utc for index in sampled_indices],
            "equity_curve": [equity_curve[index] for index in sampled_indices],
        }

    def _default_report(
        self,
        result: ExperimentResult,
        critique: dict[str, object] | None,
    ) -> str:
        lines = [
            f"# Experiment {result.experiment_id}",
            "",
            f"- Strategy: `{result.spec.name}`",
            f"- Family: `{result.spec.signal.name}`",
            f"- Promotion stage: `{result.promotion_stage}`",
            f"- Return %: `{result.aggregate_metrics.get('return_pct', 0.0):.3f}`",
            f"- Annualized Sharpe: `{result.aggregate_metrics.get('annualized_sharpe', 0.0):.3f}`",
            f"- Sharpe-like: `{result.aggregate_metrics.get('sharpe_like', 0.0):.3f}`",
            f"- Max drawdown %: `{result.aggregate_metrics.get('max_drawdown_pct', 0.0):.3f}`",
            "",
        ]
        if result.holdout_result is not None:
            lines.extend(
                [
                    f"- Holdout return %: `{result.holdout_result.metrics.get('return_pct', 0.0):.3f}`",
                    f"- Holdout annualized Sharpe: `{result.holdout_result.metrics.get('annualized_sharpe', 0.0):.3f}`",
                    "",
                ]
            )
        if critique:
            lines.append("## Critique")
            lines.append("")
            for key, value in critique.items():
                lines.append(f"- {key}: `{value}`")
            lines.append("")
        return "\n".join(lines)
