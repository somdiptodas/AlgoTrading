# SPY 1-Minute Research System Audit Tracker

Created: 2026-04-25

This file consolidates the Codex audit and the Claude Code audit into a prioritized tracker for improving the research loop before treating it as a foundation for automated trading.

## Current Evidence Snapshot

- Completed experiments: 72.
- Families tested: 24 `ema_cross`, 24 `breakout`, 24 `rsi_reversion`.
- Candidates promoted: 0.
- Frontier experiments: 10.
- Experiments beating buy-and-hold: 0.
- Best observed absolute return: about +2.11%, still about -10.63 percentage points behind buy-and-hold.
- Research storage size after the 2026-04-25 ledger compaction: about 4.8 GB total, with 4.8 GB artifacts and a 716 KB `ledger.db`.
- Ledger JSON payload size after the 2026-04-25 compaction: about 0.48 MB total across 72 rows.
- Test status: `.venv/bin/pytest -q` passes 30 tests as of 2026-04-25 after compact ledger storage.

## Verification Of Claude Audit

### Confirmed

- No strategy has reached `candidate`; all 72 completed experiments lag buy-and-hold.
- Before the 2026-04-25 fix, `regime_pass` used raw SPY monthly close-to-close returns, not strategy PnL or equity returns.
- Suppressor distance is parameter-scale biased because it normalizes by `max(abs(left), abs(right), 1)`, not known parameter ranges or grid step sizes. The current suppression log shows RSI has much higher average suppression weight than EMA or breakout.
- Aggregate metrics are simple unweighted means across folds.
- `sharpe_like` is not a conventional annualized Sharpe ratio.
- The legacy backtest regression is broken because it runs over all available data while expecting an older shorter data window.
- Before the 2026-04-25 fix, neighborhood robustness evaluated neighbors on only the first fold.
- The production generator dedupe hook is effectively dead because the loop passes `seen_evaluation_key=lambda spec: False`.
- `fixed_fraction` exists and is registered but the planner never emits it.
- Critic output is mostly decorative; it does not directly steer planning or scoring.
- Before the 2026-04-25 fix, `profit_factor` could serialize as non-standard JSON `Infinity`, and result artifacts contained `Infinity` occurrences.
- Artifact manifests currently write `"generator_kind": null`.
- Session filters are validated and hashed, but execution behavior is controlled by `ExecConfig.regular_session_only`, so filters can create different hashes for identical execution behavior.

### Adjusted

- The failing regression currently sees `194,316` regular-session bars, not `194,423`.
- Before the 2026-04-25 fix, `regime_pass` could vary across specs when different required history changed the fold/test slices, but it was still a market-data property rather than a strategy-performance property.
- The proposed annualized Sharpe fix should be computed from per-bar returns directly. Do not implement it as `sharpe_like * sqrt(252 * 390)`, because current `sharpe_like` already multiplies by `sqrt(N)` for the observed fold/window length.
- "Promotion bar too hard" is plausible, but the immediate issue is not just threshold strictness. The current search space has weak hypotheses, the robustness gates are flawed, and benchmark selection needs refinement.

## P0 - Blockers And Correctness Fixes

- [x] Fix `tests/test_backtest_regression.py` so the suite is green.
  - Pin the regression to a fixed start/end date range or use a fixture DB.
  - Do not assert against "all data" in a growing market-data database.
  - Completed 2026-04-25: pinned the legacy EMA regression to the `2025-10-21` through `2026-04-20` New York session window, preserving the characterized `48,002` bars, `391` trades, and `$94,959.18` final cash.
  - Verification: `.venv/bin/pytest -q` passes 13 tests.

- [x] Replace `regime_pass` with a strategy-specific concentration check.
  - Use strategy returns, equity deltas, or trade PnL by calendar month.
  - Example gate: fail if one month contributes more than 80% of total positive PnL or if one month dominates drawdown.
  - Completed 2026-04-25: `regime_pass` now uses realized strategy trade PnL by New York calendar month instead of SPY close-to-close market returns. The gate fails when positive PnL is unavailable, when one month contributes more than 80% of total positive PnL, or when one month contributes more than 80% of realized loss PnL.
  - Compatibility: kept `regime_pass` and `monthly_concentration_pct`; added explicit positive/loss monthly PnL concentration fields for reports and ledger payloads.
  - Verification: `.venv/bin/pytest -q` passes 18 tests.

- [x] Evaluate robustness neighbors across the same full walk-forward plan.
  - Current behavior compares aggregate candidate metrics against neighbors evaluated only on fold 1.
  - Use the same folds and aggregation path for candidate and neighbors.
  - Completed 2026-04-25: `EvaluationRunner.evaluate_preview` now evaluates candidate and neighbor specs through the same `_evaluate_preview_folds` helper, using the preview's full fold set and shared aggregate metric path.
  - Regression coverage: added a runner-level test proving neighbor metrics are collected from every fold and aggregated instead of using only fold 1.
  - Verification: `.venv/bin/pytest tests/test_robustness.py -q` passes 6 tests; `.venv/bin/pytest -q` passes 19 tests.

- [x] Replace unweighted fold aggregation.
  - Prefer combined out-of-sample equity metrics.
  - If metrics must be averaged, weight by test bar count or elapsed session count.
  - Completed 2026-04-25: `aggregate_metric_dicts` now accepts explicit non-negative weights and `EvaluationRunner._evaluate_preview_folds` weights aggregate fold metrics and baseline deltas by each fold's out-of-sample bar count. This same path is used for robustness neighbor aggregation.
  - Regression coverage: added direct weighted aggregation coverage and updated the robustness neighbor test so unequal fold bar counts produce weighted aggregate neighbor metrics.
  - Verification: `.venv/bin/pytest tests/test_splits_metrics_registry.py tests/test_robustness.py -q` passes 11 tests.

- [x] Fix non-standard JSON values.
  - Replace `Infinity`, `-Infinity`, and `NaN` serialization with `null` or bounded sentinel values.
  - Set `allow_nan=False` once metrics are sanitized.
  - Completed 2026-04-25: added recursive JSON sanitization for ledger/artifact/CLI payloads, writing non-finite floats as `null` with `allow_nan=False`; legacy ledger JSON constants now read back as null and null metric fields are skipped on reconstruction.
  - Compatibility: strategy spec JSON output now uses `allow_nan=False`; `eval` strategy JSON input rejects `NaN`, `Infinity`, and `-Infinity`; strategy validation rejects non-finite numeric exec/signal/sizing/filter values.
  - Data cleanup: sanitized the existing ignored `data/research/ledger.db` rows and artifact JSON files; follow-up scans found 0 ledger rows and 0 artifact JSON files containing `Infinity`, `-Infinity`, or `NaN`.
  - Verification: `.venv/bin/pytest tests/test_ledger.py tests/test_splits_metrics_registry.py -q` passes 12 tests; `.venv/bin/pytest -q` passes 26 tests.

- [x] Compact ledger storage.
  - Keep only compact summary fields in `ledger_entries.entry_json`.
  - Store full bars, trades, and equity curves only in artifact files.
  - Query frontier and history from indexed columns instead of deserializing full results.
  - Completed 2026-04-25: ledger entries now store compact fold summaries with metrics, baselines, warnings, and `backtest_summary` counts/cash only; full bars, trades, and equity curves remain in artifact `result.json`, `trades.json`, and `equity.json`.
  - Query path: added a scalar `trade_count` ledger column and changed `top_experiments()` to rank from scalar DB columns before deserializing only the selected compact entries.
  - Migration/data cleanup: legacy full `entry_json` rows remain readable; missing `trade_count` columns are backfilled during initialization; the ignored live `data/research/ledger.db` was compacted from about 2.35 GB of `entry_json` to about 0.48 MB, with 216 fold summaries, 0 embedded full backtests, and 0 zero-count fold summaries.
  - Verification: `.venv/bin/pytest tests/test_ledger.py tests/test_research_queue.py -q` passes 9 tests; `.venv/bin/pytest -q` passes 30 tests.

## P1 - Evaluation Quality And Promotion Semantics

- [x] Add a proper annualized risk metric.
  - Compute annualized Sharpe from per-bar or per-session returns directly.
  - Keep `sharpe_like` only as a backward-compatible internal field if needed.
  - Completed 2026-04-25: added `annualized_sharpe` from direct per-bar equity returns with SPY 1-minute annualization, preserved `sharpe_like`, added annualized baseline deltas, displayed the new metric in reports/artifact summaries, and recomputed aggregate annualized Sharpe from combined fold backtest returns instead of averaging fold Sharpe values.
  - Verification: `.venv/bin/pytest tests/test_splits_metrics_registry.py tests/test_robustness.py -q` passes 18 tests; `.venv/bin/pytest -q` passes 33 tests. Verification subagent reported no blockers.

- [x] Tighten and rename promotion stages.
  - `frontier` should not imply tradable edge when alpha is negative.
  - Require positive return, positive risk-adjusted metric, minimum trade count, and relevant benchmark pass for frontier/candidate status.
  - Completed 2026-04-25: new evaluations now use `research_frontier` instead of `frontier`, and promotion requires all robustness gates, positive return, positive annualized risk-adjusted return, at least 10 trades, and positive buy-and-hold edge. `candidate` additionally requires more than 0.5 percentage points of buy-and-hold edge. Active frontier-neighborhood planning now seeds only from `research_frontier` and `candidate`, while legacy `frontier` rows remain readable but are downgraded for ranking and critic semantics.
  - Verification: `.venv/bin/pytest tests/test_promotion.py tests/test_ledger.py tests/test_research_queue.py -q` passes 24 tests; `.venv/bin/pytest -q` passes 48 tests. Verification subagent re-review reported no blockers.

- [x] Add fairer intraday baselines.
  - Keep buy-and-hold, but add baselines that match the strategy's overnight-flat constraint:
    - always flat
    - regular-session open-to-close long
    - session-long flat-at-close
    - randomized entry with same exposure/trade count
  - Completed 2026-04-25: kept `always_flat` and `buy_and_hold`, added `regular_session_open_to_close_long`, engine-faithful `session_long_flat_at_close`, and deterministic `randomized_entry_same_exposure` baselines. The randomized baseline is seeded from spec/fold/window material, uses cost-aware fills, preserves realized trade count and exposure when a valid schedule exists, and otherwise falls back explicitly to flat.
  - Verification: `.venv/bin/pytest tests/test_baselines.py tests/test_research_queue.py -q` passes 8 tests; `.venv/bin/pytest -q` passes 54 tests. Verification subagent re-review reported no blockers.

- [x] Add a locked holdout policy.
  - Reserve the most recent 3-6 months from research-loop selection.
  - Only run promoted candidates against this holdout.
  - Track holdout results separately from research folds.
  - Completed 2026-04-25: added a loop-level locked holdout policy with `--holdout-months` defaulting to 3 calendar months. Candidate previews build research folds only from pre-holdout bars with an embargo gap, short/constrained slices fail closed when no pre-holdout research data remains, and holdout evaluation runs only after a result reaches `candidate`.
  - Tracking: `ExperimentResult` and `LedgerEntry` now preserve an optional `holdout_result` separately from research `fold_results` and aggregate metrics; reports/artifact summaries display holdout metrics when present.
  - Verification: `.venv/bin/pytest tests/test_holdout.py tests/test_ledger.py tests/test_research_queue.py -q` passes 16 tests; `.venv/bin/pytest -q` passes 59 tests. Verification subagent re-review reported no blockers.

- [x] Add data-quality validation before evaluation.
  - Detect missing bars, duplicate timestamps, null OHLC, OHLC sanity violations, volume anomalies, and unexpected session lengths.
  - Propagate warnings into `FoldResult.warnings`.
  - Completed 2026-04-25: added evaluation data-quality warnings for raw null OHLC, duplicate timestamps, missing one-minute bars, OHLC sanity violations, null/invalid volume, and unexpected full-session lengths. Fold and holdout paths collect warnings before signal generation/backtesting, preserve them in `FoldResult.warnings`, serialize them, and render holdout warnings in reports.
  - False-positive control: session-length checks only fire for sessions that appear to span the full regular session, and raw boundary expansion is limited to immediate missing first/last minute cases so partial folds do not inherit outside-session warnings.
  - Verification: `.venv/bin/pytest tests/test_data_quality.py tests/test_ledger.py tests/test_holdout.py -q` passes 19 tests; `.venv/bin/pytest -q` passes 65 tests. Verification subagent re-review reported no blockers.

- [x] Improve cost and fill assumptions for research.
  - Add commission per share.
  - Add spread/slippage scenarios.
  - Add volume participation or max notional caps.
  - Report cost drag per strategy.
  - Completed 2026-04-25: extended `ExecConfig` with `commission_per_share`, `spread_bps`, and `max_position_notional`; fills now apply per-share commissions, half-spread plus slippage on both sides, and max-notional share caps. Trades record realized execution cost, metrics report cash/percent cost drag, and fold/holdout metrics include zero-cost drag plus +2 bps slippage/spread stress scenarios.
  - Reporting/compatibility: reports show the new execution fields and cost metrics; legacy trade payloads read with `cost_cash=0.0`; backtest CLI accepts the new cost flags; default EMA regression remains unchanged.
  - Verification: `.venv/bin/pytest tests/test_costs.py tests/test_backtest_regression.py tests/test_splits_metrics_registry.py -q` passes 16 tests; `.venv/bin/pytest -q` passes 69 tests. Verification subagent reported no blockers.

## P2 - Loop Efficiency And Search Control

- [x] Make planner allocation explicit.
  - Reserve slots for grid exploration, frontier neighborhoods, and each enabled family before truncation.
  - Avoid frontier candidates being accidentally crowded out by grid ordering.
  - Completed 2026-04-25: `DeterministicPlanner` now builds explicit grid buckets per enabled family and frontier-neighborhood buckets per enabled family, then round-robins across non-empty buckets before truncation/dedupe. Frontier neighborhoods and later grid families can no longer be crowded out by early grid ordering when the batch has enough capacity.
  - Verification: `.venv/bin/pytest tests/test_planner.py tests/test_research_queue.py -q` passes 7 tests; `.venv/bin/pytest -q` passes 73 tests. Verification subagent reported no blockers.

- [x] Replace the hidden `batch_size * 12` over-plan factor.
  - Use a named configuration value.
  - Track planned, previewed, selected, evaluated, duplicate, and suppressed counts separately.
  - Completed 2026-04-25: replaced the loop's hidden over-plan expression with named `DEFAULT_OVERPLAN_FACTOR`, `MIN_PLANNED_SPECS`, and a `--overplan-factor` CLI flag. Candidate queue results now track successful previews separately from duplicates and suppressions, and loop output includes explicit planned, previewed, selected, evaluated, duplicate, and suppressed counts.
  - Verification: `.venv/bin/pytest tests/test_loop_cmd.py tests/test_research_queue.py -q` passes 6 tests; `.venv/bin/pytest -q` passes 76 tests. Verification subagent reported no blockers.

- [x] Move dedupe earlier.
  - Compute evaluation keys before preview where possible.
  - Use current data snapshot, split plan, cost model, and spec hash to reject known evaluations before building previews.
  - Completed 2026-04-25: added `EvaluationRunner.evaluation_key_for_spec()` to compute the same research evaluation key as preview using validated spec hash, current data snapshot, holdout-adjusted split plan, and cost model id. `DeterministicCandidateQueue` now rejects historical and same-batch duplicate evaluation keys before building previews, while preserving preview, duplicate, selection, and suppression counts.
  - Verification: `.venv/bin/pytest tests/test_research_queue.py -q` passes 6 tests; `.venv/bin/pytest -q` passes 77 tests. Verification subagent reported no blockers.

- [x] Limit preview work.
  - Cheap-rank candidates first using params/history.
  - Preview only a bounded multiple of final batch size.
  - Completed 2026-04-25: added a named `--preview-factor` loop cap and a cheap pre-preview ranking pass in `DeterministicCandidateQueue` using existing family history, parent score, generator kind, novelty, and required-history signals. The queue now caps preview creation while continuing to scan evaluation keys so duplicate accounting remains uncapped.
  - Verification: `.venv/bin/pytest tests/test_loop_cmd.py tests/test_research_queue.py -q` passes 10 tests; `.venv/bin/pytest -q` passes 80 tests. Verification subagent re-review reported no blockers.

- [x] Fix suppressor geometry.
  - Normalize parameter distances by registry-known min/max ranges or grid steps.
  - Weight failure types differently.
  - Require repeated nearby failures before applying large suppression.
  - Completed 2026-04-25: `RegionSuppressor` now uses registry-derived per-family parameter ranges/steps for suppression distance while leaving candidate novelty/diversity distance semantics unchanged. Suppression weights now vary by failed gate type, and single nearby failures receive only a small penalty while repeated nearby failures can reach the cap.
  - Verification: `.venv/bin/pytest tests/test_suppressor.py tests/test_research_queue.py tests/test_loop_cmd.py -q` passes 15 tests; `.venv/bin/pytest -q` passes 85 tests. Verification subagent reported no blockers.

- [ ] Separate suppression audit types.
  - Distinguish selected/evaluated suppression decisions from discarded preview noise.

- [ ] Wire critic output into planning.
  - Convert notes such as poor fold consistency, excessive trading, or benchmark failure into scoring penalties or planner constraints.

- [ ] Resolve session filter redundancy.
  - Either remove `FilterSpec("session", {"session": "regular"})` from generated specs or make it control `ExecConfig.regular_session_only`.
  - Ensure semantically identical execution specs have identical hashes.

- [ ] Pass `generator_kind` into artifact manifests.
  - Current ledger rows know generator kind, but artifact manifests write `null`.

## P3 - Search Space Expansion

- [ ] Add `fixed_fraction` to planner output.
  - Test fractions such as 0.25, 0.50, and 1.00 across existing signal families.
  - Evaluate whether lower exposure improves drawdown-adjusted returns.

- [ ] Add uniform risk-management exits.
  - Start with `stop_loss_bps` in `ExecConfig`.
  - Consider `take_profit_bps`, `max_hold_bars`, and trailing exits after stop-loss behavior is characterized.

- [ ] Add real session-time controls.
  - First 30 minutes only.
  - Last 30 minutes only.
  - Avoid midday.
  - Configurable no-new-entry cutoff before close.

- [ ] Add VWAP deviation signal family.
  - Long below VWAP by an entry threshold.
  - Exit at VWAP reversion, timeout, stop, or close.

- [ ] Add volatility and volume regime filters.
  - Intraday realized volatility percentile.
  - Prior-day range.
  - Volume percentile or relative volume.
  - Trend day vs mean-reversion day classification.

- [ ] Add ensemble or confirmation specs only after metrics and validation are fixed.
  - Example: RSI reversion gated by volatility regime or VWAP distance.

## P4 - Future Trading Readiness

- [ ] Add paper-trading adapter boundaries.
  - Broker abstraction.
  - Order IDs and idempotency.
  - Position reconciliation.
  - Trade audit log.

- [ ] Add hard risk controls.
  - Max position notional.
  - Max daily loss.
  - Max orders per day.
  - Kill switch.
  - No-trade calendar/event windows.

- [ ] Add live data integrity checks.
  - Stale quote/bar detection.
  - Missing minute handling.
  - Market-hours and early-close calendar.

- [ ] Add strategy decay monitoring.
  - Re-evaluate promoted specs as new data arrives.
  - Link evaluations across changing data snapshots.
  - Report rolling degradation and promotion/demotion status.

## Suggested Execution Order

1. Fix the broken regression test.
2. Sanitize metric serialization and compact ledger/artifact payloads.
3. Fix fold aggregation.
4. Tighten promotion semantics and add fair intraday baselines.
5. Add explicit planner allocation, early dedupe, and bounded previews.
6. Fix suppressor distance scaling.
7. Add `fixed_fraction` and run a fresh controlled batch.
8. Add holdout policy before trusting any newly promoted result.
9. Add stop-loss, time filters, and VWAP deviation experiments.
10. Begin paper-trading readiness work only after a strategy survives corrected validation.

## Definition Of Done For A Credible Candidate

A strategy should not be considered a credible candidate until it:

- Passes the full test suite.
- Uses sanitized, reproducible artifacts and compact ledger records.
- Has positive combined out-of-sample return.
- Beats an appropriate intraday baseline, not only always-flat.
- Has acceptable drawdown and tail-loss metrics.
- Passes strategy-specific concentration checks.
- Shows neighborhood robustness across the full walk-forward plan.
- Survives realistic cost/slippage sensitivity.
- Has enough trades to avoid a tiny-sample illusion.
- Passes the locked holdout after all research selection decisions are complete.
