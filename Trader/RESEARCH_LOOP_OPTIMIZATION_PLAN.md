# Research Loop Optimization Plan

Date: 2026-04-25

## What I Ran

Baseline before the additional runs:

- Ledger rows: 78 completed experiments.
- Latest completed experiment before these runs: `2026-04-25T17:29:25.206810+00:00`.
- Current decay monitor output: empty, because there are no current `candidate` or `research_frontier` entries.

Commands:

```bash
/usr/bin/time -p .venv/bin/python -m trader loop --folds 3
/usr/bin/time -p .venv/bin/python -m trader loop --folds 3
```

Results:

| Run | Planned | Previewed | Selected | Evaluated | Duplicate | Suppressed | Runtime |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Previous reference run | 72 | 24 | 24 | 6 | 0 | 0 | about 10 min |
| Iteration 1 | 72 | 24 | 24 | 6 | 6 | 22 | 541s |
| Iteration 2 | 72 | 24 | 24 | 6 | 12 | 24 | 739s |

The 12 new evaluations from the two additional loops:

| Family | Count | Best Return | Worst Return | Promotion |
| --- | ---: | ---: | ---: | --- |
| `rsi_reversion` | 5 | `-0.743%` | `-4.741%` | all `exploratory` |
| `vwap_deviation` | 5 | `-1.581%` | `-6.844%` | all `exploratory` |
| `breakout` | 2 | `-1.429%` | `-6.917%` | all `exploratory` |
| `ema_cross` | 0 | n/a | n/a | n/a |

Final ledger state after the two runs:

- Ledger rows: 90 completed experiments.
- No new promoted strategies.
- `ledger decay` still returns an empty report because there are still no current `candidate` or `research_frontier` rows.

## Why The Loop Took So Long

The main cost is full robustness evaluation, not planning.

Timing probe on one representative `vwap_deviation` strategy:

| Step | Time |
| --- | ---: |
| Preview setup | `1.539s` |
| Evaluate without robustness | `15.687s` |
| Evaluate with robustness | `94.809s` |

The loop evaluates 6 selected candidates with robustness. Robustness evaluates up to 6 neighboring specs across the same walk-forward folds. So one selected strategy is closer to 7 strategy evaluations than 1.

The current default loop also does more work than the final `completed=6` number suggests:

- Plans 72 specs.
- Computes current evaluation keys for many plans.
- Previews 24 specs.
- Selects 24 specs.
- Fully evaluates only the first 6 selected specs.

That means the output field `selected=24` is misleading: only 6 selected candidates are actually evaluated.

## Why The First Run Was All `vwap_deviation`

This was caused by the selection scoring, not by the planner only producing VWAP specs.

Before the first run:

- `ema_cross`, `breakout`, and `rsi_reversion` each had about two dozen historical evaluations.
- `vwap_deviation` had zero history.
- The candidate queue gives a large boost to underexplored families:
  - `family_quota_boost = (max_family_count - family_count) * 3.0`
- The suppressor had failure regions for `ema_cross`, `breakout`, and `rsi_reversion`, but not yet for `vwap_deviation`.

So the first run heavily favored `vwap_deviation`. After six bad VWAP results landed, later loops became mixed:

- Iteration 1 evaluated `rsi_reversion` and `vwap_deviation`.
- Iteration 2 evaluated `breakout`, `rsi_reversion`, and `vwap_deviation`.
- Suppression increased from `0` to `22` to `24`, which shows the feedback loop is starting to work.

The problem is not that exploration exists. The problem is that the exploration boost can dominate quality and can spend many expensive robustness evaluations on a clearly weak family before enough failure history accumulates.

## Current Strategy Shape Limitation

The system currently supports:

- One primary signal per `StrategySpec`.
- Optional filters that mask the primary signal.
- Some confirmation-style variants, for example RSI confirmed by VWAP distance.

The system does not yet support true multi-signal strategies such as:

- EMA trend plus RSI pullback.
- Breakout only when relative volume and trend regime agree.
- Majority vote across EMA, breakout, RSI, and VWAP.
- Weighted ensembles trained only on the train folds.

The user's expectation is right: the best strategy is likely to combine multiple signals and indicators. The current design can approximate this with filters, but it cannot express a real strategy that combines several signal regimes as peers.

## No Future Data / Lookahead Assessment

Current signal generation mostly follows the right contract:

- Signals may use the completed current bar.
- The execution engine enters on the next bar.
- Breakout rolling highs/lows are exclusive of the current bar.
- Several leakage tests already exist for breakout, VWAP, VWAP filters, relative volume, day type, intraday volatility, and prior-day range.

Important detail:

- `vwap_deviation` uses the current bar's close and VWAP to decide the regime for that bar, but the engine enters on the next bar. That is acceptable for minute-close decisions if the current minute bar is complete.
- It would not be acceptable for intrabar/live decisions before the bar is complete. Live trading should use the new live-data integrity boundary and only pass completed-bar snapshots into signal generation.

No obvious direct future-peeking bug showed up in the inspected signal path. The next work should preserve this invariant with explicit tests for any new composite or ensemble signal.

## Prioritized Task List

### P0 - Stop Spending Robustness Time On Obvious Losers

Implement staged evaluation:

1. Evaluate the selected candidate on the normal folds.
2. Compute aggregate metrics and cheap promotion prerequisites.
3. Skip neighbor robustness if the candidate already fails hard prerequisites such as:
   - non-positive return,
   - weak annualized Sharpe,
   - too few or too many trades,
   - benchmark underperformance,
   - unacceptable drawdown.
4. Only run robustness for candidates that could plausibly become `research_frontier` or `candidate`.

Expected impact:

- For the recent losing candidates, runtime could drop from about `95s` each to about `16s` each.
- A 6-candidate loop could drop from roughly 9-12 minutes to closer to 2-4 minutes when most candidates are poor.

Tests:

- Candidate that fails base metrics does not call neighbor robustness.
- Candidate that passes base metrics still runs full robustness.
- Promotion stage remains `exploratory` when robustness is skipped.

### P0 - Fix Loop Reporting And Selection Semantics

The loop reports `selected=24` but evaluates only 6. Make this explicit:

- `previewed`: candidates that had walk-forward preview objects built.
- `ranked`: candidates ranked by the queue.
- `evaluated`: candidates actually fully evaluated.
- `selected_for_evaluation`: the first `batch_size` candidates.

Also include evaluated family counts and generator-kind counts in the JSON output.

Why this matters:

- It makes performance diagnosis much easier.
- It prevents us from thinking 24 strategies were evaluated when only 6 were.

### P0 - Add Per-Phase Timing Instrumentation

Emit timing for:

- planning,
- validation/dedupe,
- evaluation-key computation,
- preview generation,
- queue scoring,
- each full candidate evaluation,
- base fold evaluation,
- robustness neighbor evaluation,
- artifact/report writing,
- ledger writes.

Use this to track speed improvements with real evidence.

### P0 - Add A Trade-Count And Churn Gate Before Robustness

Recent bad candidates had very high trade counts:

- Some VWAP candidates: 400-700 trades.
- One breakout candidate: about 1537 trades.

Add a cheap gate before robustness:

- reject or down-rank candidates with excessive trades,
- require cost-stress survival before robustness,
- penalize high turnover in queue scoring.

This is both a speed improvement and a research-quality improvement.

### P1 - Build True Composite Signals

Add a first-class composite signal capability.

Start deterministic and simple:

- `all`: long only when every child signal is long.
- `any`: long when any child signal is long.
- `majority`: long when at least N of M child signals are long.
- `primary_with_confirmations`: primary signal must be long and confirmation signals must agree.

Example target specs:

- RSI pullback only when EMA trend is up.
- Breakout only when relative volume is high and day type is trend.
- VWAP mean reversion only when RSI is oversold and volatility regime is favorable.

No-lookahead requirement:

- Every child signal must generate a same-length regime using only `history_bars` and completed `test_bars` up to each index.
- Composite logic can only combine child outputs at the same index.
- Engine still enters on the next bar.
- Add leakage tests that mutate future bars and verify earlier composite outputs do not change.

### P1 - Add Composite Strategy Planning

Once composite signals exist, add planner grids for a small number of sensible combinations instead of exploding the search space.

Recommended first grid:

- `ema_cross AND rsi_reversion`
- `ema_cross AND breakout`
- `rsi_reversion AND vwap_deviation`
- `breakout AND relative_volume`
- `vwap_deviation AND intraday_volatility`

Keep the grid small and evidence-driven. Do not cross every signal with every parameter set.

### P1 - Improve Family Balancing

The current underexplored-family boost is useful but too coarse.

Replace it with a bounded exploration policy:

- guarantee at least one slot for underexplored families,
- cap a family at a max share per batch,
- decay the exploration boost after a small number of poor results,
- account for suppressor failures before spending full evaluations.

This would have prevented the first all-VWAP batch while still allowing VWAP exploration.

### P1 - Add A Fast Preview Score

Preview currently builds folds but does not run a cheap partial backtest score. Add an optional fast preview phase:

- one fold only,
- shorter recent window,
- no robustness,
- no artifacts,
- cheap trade-count and return checks.

Use this to choose the 6 full evaluations from the 24 previews.

### P1 - Parallelize Full Candidate Evaluation

Full selected candidates are independent. Evaluate them in parallel worker processes after confirming SQLite/artifact writes remain serialized or conflict-free.

Safe shape:

1. Build selected candidate list in the main process.
2. Evaluate candidates in workers.
3. Return result payloads.
4. Main process writes artifacts and ledger entries.

This avoids concurrent SQLite writes while using multiple CPU cores for backtests.

### P2 - Cache Feature And Regime Computation

Robustness evaluates neighboring specs that often reuse the same base features:

- EMA series,
- RSI series,
- rolling highs/lows,
- volatility windows,
- volume windows.

Add per-fold feature caches keyed by:

- data snapshot,
- fold range,
- feature name,
- lookback/window params.

This is lower priority than staged robustness because it is more invasive, but it should help once we run more serious ensemble searches.

### P2 - Add Candidate-Level Failure Memory

The suppressor works after failures are in the ledger, but the first bad batch can still spend many evaluations in a weak region.

Add within-run suppression:

- if first 2 evaluated candidates from the same family/region are bad,
- down-rank or skip remaining selected candidates from that same region in the same loop run.

This would reduce repeated VWAP losses inside one loop.

### P2 - Add Explicit Current-Snapshot Rechecks Once Promotions Exist

The decay monitor is ready, but currently no current `candidate` or `research_frontier` rows exist. Once promotions exist:

- run `eval --decay-promoted` after each data refresh,
- include decay status in the loop summary,
- demote or quarantine strategies that fail current-snapshot rechecks.

## Recommended Next Implementation Order

1. Implement staged robustness skip for obvious losers.
2. Add timing instrumentation and clearer loop counters.
3. Add trade-count/churn gates before robustness.
4. Add bounded family balancing so exploration cannot dominate an entire expensive batch.
5. Add deterministic composite signals with leakage tests.
6. Add a small composite planner grid.
7. Add fast preview scoring.
8. Parallelize full candidate evaluation.
9. Add feature/regime caching.

The first four items should improve speed and search quality immediately without changing strategy semantics. Composite signals should come after that so the system can explore richer strategies without making each loop even slower.
