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
- Research storage size: about 7.0 GB total, including 4.8 GB artifacts and 2.2 GB `ledger.db`.
- Ledger JSON payload size: about 2.24 GB total, averaging about 31 MB per ledger row.
- Test status: `.venv/bin/pytest -q` passes 13 tests as of 2026-04-25 after pinning the legacy EMA regression to a fixed characterization window.

## Verification Of Claude Audit

### Confirmed

- No strategy has reached `candidate`; all 72 completed experiments lag buy-and-hold.
- `regime_pass` currently uses raw SPY monthly close-to-close returns, not strategy PnL or equity returns.
- Suppressor distance is parameter-scale biased because it normalizes by `max(abs(left), abs(right), 1)`, not known parameter ranges or grid step sizes. The current suppression log shows RSI has much higher average suppression weight than EMA or breakout.
- Aggregate metrics are simple unweighted means across folds.
- `sharpe_like` is not a conventional annualized Sharpe ratio.
- The legacy backtest regression is broken because it runs over all available data while expecting an older shorter data window.
- Neighborhood robustness evaluates neighbors on only the first fold.
- The production generator dedupe hook is effectively dead because the loop passes `seen_evaluation_key=lambda spec: False`.
- `fixed_fraction` exists and is registered but the planner never emits it.
- Critic output is mostly decorative; it does not directly steer planning or scoring.
- `profit_factor` can serialize as non-standard JSON `Infinity`; all current result artifacts contain at least one `Infinity` occurrence.
- Artifact manifests currently write `"generator_kind": null`.
- Session filters are validated and hashed, but execution behavior is controlled by `ExecConfig.regular_session_only`, so filters can create different hashes for identical execution behavior.

### Adjusted

- The failing regression currently sees `194,316` regular-session bars, not `194,423`.
- `regime_pass` may vary across specs when different required history changes the fold/test slices, but it is still a market-data property rather than a strategy-performance property.
- The proposed annualized Sharpe fix should be computed from per-bar returns directly. Do not implement it as `sharpe_like * sqrt(252 * 390)`, because current `sharpe_like` already multiplies by `sqrt(N)` for the observed fold/window length.
- "Promotion bar too hard" is plausible, but the immediate issue is not just threshold strictness. The current search space has weak hypotheses, the robustness gates are flawed, and benchmark selection needs refinement.

## P0 - Blockers And Correctness Fixes

- [x] Fix `tests/test_backtest_regression.py` so the suite is green.
  - Pin the regression to a fixed start/end date range or use a fixture DB.
  - Do not assert against "all data" in a growing market-data database.
  - Completed 2026-04-25: pinned the legacy EMA regression to the `2025-10-21` through `2026-04-20` New York session window, preserving the characterized `48,002` bars, `391` trades, and `$94,959.18` final cash.
  - Verification: `.venv/bin/pytest -q` passes 13 tests.

- [ ] Replace `regime_pass` with a strategy-specific concentration check.
  - Use strategy returns, equity deltas, or trade PnL by calendar month.
  - Example gate: fail if one month contributes more than 80% of total positive PnL or if one month dominates drawdown.

- [ ] Evaluate robustness neighbors across the same full walk-forward plan.
  - Current behavior compares aggregate candidate metrics against neighbors evaluated only on fold 1.
  - Use the same folds and aggregation path for candidate and neighbors.

- [ ] Replace unweighted fold aggregation.
  - Prefer combined out-of-sample equity metrics.
  - If metrics must be averaged, weight by test bar count or elapsed session count.

- [ ] Fix non-standard JSON values.
  - Replace `Infinity`, `-Infinity`, and `NaN` serialization with `null` or bounded sentinel values.
  - Set `allow_nan=False` once metrics are sanitized.

- [ ] Compact ledger storage.
  - Keep only compact summary fields in `ledger_entries.entry_json`.
  - Store full bars, trades, and equity curves only in artifact files.
  - Query frontier and history from indexed columns instead of deserializing full results.

## P1 - Evaluation Quality And Promotion Semantics

- [ ] Add a proper annualized risk metric.
  - Compute annualized Sharpe from per-bar or per-session returns directly.
  - Keep `sharpe_like` only as a backward-compatible internal field if needed.

- [ ] Tighten and rename promotion stages.
  - `frontier` should not imply tradable edge when alpha is negative.
  - Require positive return, positive risk-adjusted metric, minimum trade count, and relevant benchmark pass for frontier/candidate status.

- [ ] Add fairer intraday baselines.
  - Keep buy-and-hold, but add baselines that match the strategy's overnight-flat constraint:
    - always flat
    - regular-session open-to-close long
    - session-long flat-at-close
    - randomized entry with same exposure/trade count

- [ ] Add a locked holdout policy.
  - Reserve the most recent 3-6 months from research-loop selection.
  - Only run promoted candidates against this holdout.
  - Track holdout results separately from research folds.

- [ ] Add data-quality validation before evaluation.
  - Detect missing bars, duplicate timestamps, null OHLC, OHLC sanity violations, volume anomalies, and unexpected session lengths.
  - Propagate warnings into `FoldResult.warnings`.

- [ ] Improve cost and fill assumptions for research.
  - Add commission per share.
  - Add spread/slippage scenarios.
  - Add volume participation or max notional caps.
  - Report cost drag per strategy.

## P2 - Loop Efficiency And Search Control

- [ ] Make planner allocation explicit.
  - Reserve slots for grid exploration, frontier neighborhoods, and each enabled family before truncation.
  - Avoid frontier candidates being accidentally crowded out by grid ordering.

- [ ] Replace the hidden `batch_size * 12` over-plan factor.
  - Use a named configuration value.
  - Track planned, previewed, selected, evaluated, duplicate, and suppressed counts separately.

- [ ] Move dedupe earlier.
  - Compute evaluation keys before preview where possible.
  - Use current data snapshot, split plan, cost model, and spec hash to reject known evaluations before building previews.

- [ ] Limit preview work.
  - Cheap-rank candidates first using params/history.
  - Preview only a bounded multiple of final batch size.

- [ ] Fix suppressor geometry.
  - Normalize parameter distances by registry-known min/max ranges or grid steps.
  - Weight failure types differently.
  - Require repeated nearby failures before applying large suppression.

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
3. Fix `regime_pass`, neighbor robustness, and fold aggregation.
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
