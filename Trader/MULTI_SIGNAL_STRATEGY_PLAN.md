# Multi-Signal Strategy Plan

## Intent

Build a robust multi-signal trade decision mechanism for the SPY research loop.
The current system can combine signals through the `composite` signal family, but
that model still produces one boolean long/not-long regime. That is too limited
for serious strategy search because it couples entry and exit logic, hides which
signals actually drove a trade, and does not scale cleanly to a large search
space.

This plan replaces that with explicit entry and exit decision rules. A strategy
should be able to require one multi-signal policy for entering a trade and a
different multi-signal policy for exiting it, while recording the exact votes
and reasons behind each trade.

## Goals

- Support multi-signal strategies with separate entry and exit rules.
- Require all active multi-signal strategies to use at least 3 signals for entry
  and at least 3 signals for exit.
- Preserve the existing hard execution constraints:
  - SPY only.
  - 1-minute bars.
  - no forward-looking data.
  - at most one open position at a time.
  - stop loss, session close, and final-bar exits remain hard execution exits.
- Make every trade explainable through structured entry and exit vote records.
- Support a large search space without eagerly enumerating every combination.
- Generate HTML reports for each research-loop run and maintain a main
  dashboard page that links all run pages, experiment reports, trade
  visualizations, and raw artifacts.
- Keep legacy single-signal and `composite` behavior available for compatibility
  tests and old artifacts, but move the active research loop toward
  `multi_signal`.

## High-Level Approach

Introduce a new decision layer between signal generation and execution.

Today:

```text
bars -> signal regime -> execution engine -> trades
```

Target:

```text
bars -> atomic signal votes -> entry rule + exit rule -> execution engine -> trades with reasons
                                                           -> run/report HTML
```

The core change is to stop treating a strategy as only a list of booleans. A
strategy should produce a per-bar decision trace:

```python
SignalVote(name="rsi_below", passed=True, detail="RSI 28.4 < 30.0")
RuleDecision(passed=True, reason="k_of_n passed: 3/5 signals", votes=(...))
TradeDecision(entry=RuleDecision(...), exit=RuleDecision(...))
```

The execution engine then uses:

- `entry.passed` while flat.
- `exit.passed` while long.

Hard exits still override signal exits.

## Proposed Strategy Shape

Use a new signal family named `multi_signal`:

```json
{
  "name": "multi_signal",
  "params": {
    "entry_rule": {
      "combiner": "k_of_n",
      "k": 3,
      "signals": [
        {"name": "rsi_below", "params": {"length": 7, "threshold": 30.0}},
        {"name": "vwap_distance", "params": {"side": "below", "min_bps": 25.0}},
        {"name": "relative_volume", "params": {"lookback": 20, "min_ratio": 1.25}},
        {"name": "ema_trend_up", "params": {"fast": 8, "slow": 34}}
      ]
    },
    "exit_rule": {
      "combiner": "any",
      "signals": [
        {"name": "rsi_above", "params": {"length": 7, "threshold": 70.0}},
        {"name": "vwap_reclaimed", "params": {"min_bps": 0.0}},
        {"name": "ema_trend_down", "params": {"fast": 8, "slow": 34}}
      ]
    }
  }
}
```

Validation rules:

- `entry_rule.signals` must contain at least 3 signals.
- `exit_rule.signals` must contain at least 3 signals.
- Child signal names must be known atomic predicates.
- Child signal params must be normalized before hashing or evaluation.
- Rules must be canonicalized so equivalent specs dedupe correctly.

## Supported Rule Combiners

- `all`: all child signals must pass.
- `any`: at least one child signal must pass.
- `k_of_n`: at least `k` child signals must pass.
- `primary_plus_k`: one primary signal must pass, plus at least `k`
  confirmation signals must pass.
- `weighted_vote`: child signals have weights and total passing weight must meet
  a threshold.

Start with `all`, `any`, and `k_of_n`. Add `primary_plus_k` and
`weighted_vote` after the basic decision contract is stable.

## Search Space Strategy

Do not eagerly materialize all possible multi-signal specs. The number of
combinations will grow too quickly.

Use a versioned search grammar that can generate candidates lazily:

```text
search_space_version: multi_signal_v1
entry signal count: 3..6
exit signal count: 3..6
entry combiners: all, k_of_n, primary_plus_k
exit combiners: any, k_of_n, primary_plus_k
atomic signal families: rsi, ema, breakout, vwap, volume, volatility, day_type
sizing: existing sizing grid
execution config: existing execution grid
```

Planner requirements:

- Generate only valid `multi_signal` candidates for the active multi-signal
  search path.
- Enforce 3+ entry signals and 3+ exit signals at generation time and validation
  time.
- Group UCB/search statistics by strategy shape, not only by the broad
  `multi_signal` family.
- Use randomized restarts when duplicate/reused candidate rates get high.
- Let Optuna/TPE tune parameters inside selected strategy shapes instead of
  trying to optimize the entire grammar as one flat parameter space.

Example shape keys:

```text
entry:k_of_n:rsi+vwap+volume+ema|exit:any:rsi+vwap+ema
entry:primary_plus_k:rsi+vwap+volume|exit:k_of_n:rsi+ema+breakout
entry:k_of_n:breakout+volume+trend|exit:any:ema+rsi+vwap
```

## Artifact And Visualization Requirements

Each persisted trade should include:

- `entry_reason`
- `exit_reason`
- `entry_rule`
- `exit_rule`
- `entry_votes`
- `exit_votes`

The trade visualizer should show:

- equity curve per fold.
- trade entry and exit markers.
- entry rule result.
- exit rule result.
- child signal vote table for each trade.

Each loop run should generate:

- one run-level HTML report summarizing planned, accepted, completed, reused,
  suppressed, and promoted experiments.
- links to each experiment's result, trade/equity visualization, markdown
  report, and raw artifact files.
- a compact table of top candidates, failed candidates, duplicate/reused counts,
  generator mix, family/shape mix, and timing breakdown.

The reporting system should also generate a main dashboard page that lists all
known loop runs in reverse chronological order and links to:

- run-level HTML reports.
- generated experiment trade/equity HTML pages.
- ledger summaries.
- loop JSON files.
- relevant report directories.

Legacy artifacts that lack vote data should still render with the existing
reason strings. Missing generated HTML should not break the dashboard; it should
show the missing link as unavailable.

## Parallel Workflow Map

Use these workflows to split implementation across agents. Each workflow should
own its files and avoid editing another workflow's files unless the dependency
has already landed.

| Workflow | Scope | Primary Ownership | Blocked By | Can Run In Parallel With |
| --- | --- | --- | --- | --- |
| A. Decision Contract | Decision dataclasses, JSON helpers, legacy regime-to-decision wrapper. | `trader/strategies/decisions.py`, decision serialization tests. | None. This should land first. | None until the interface is merged. |
| B. Atomic Predicates | Atomic predicate registry and RSI/EMA/breakout/VWAP/volume/regime predicates. | `trader/strategies/predicates/`, predicate tests. | A interface merged. | C planning only, not C implementation. |
| C. Multi-Signal Evaluator | `multi_signal` validation, rule combiners, required-history logic, no-lookahead tests. | `trader/strategies/signals/multi_signal.py`, `trader/strategies/registry.py`, multi-signal tests. | A and B. | F planner scaffolding after validation shape is agreed. |
| D. Engine And Trade Payload | Decision-based execution path, position/trade decision detail persistence, ledger/artifact serialization. | `trader/execution/`, `trader/ledger/entry.py`, artifact store tests. | C. | E2 visualizer scaffolding after payload shape is agreed. |
| E1. Run Dashboard | Per-run HTML reports, main dashboard page, report rebuild command, navigation links across generated files. | `trader/reporting/run_dashboard.py`, report CLI files, dashboard tests. | Existing loop JSON/artifact conventions only. Can start immediately. | A, B, C, F. |
| E2. Trade Vote Visualization | HTML trade/equity vote display and legacy fallback rendering. | `trader/reporting/trade_visualization.py`, visualizer tests. | D payload shape. | F. |
| F. Planner/Search | `multi_signal_v1` search grammar, lazy candidate generation, shape keys, planner tests. | `trader/research/`, loop CLI family defaults/tests. | C validation shape. | E1 and E2. |
| G. Rollout Validation | Full tests, stale artifact reset, smoke batch, 10-loop run, generated HTML review, report results. | Local generated artifacts and final plan checklist updates. | D, E1, E2, and F. | None. Run last. |

Merge order:

1. Land A first.
2. Land B.
3. Land C.
4. E1 can proceed independently once current artifact conventions are confirmed.
5. Land D, E2, and F after their blockers are merged. E2 and F can proceed in
   parallel once their input contracts are stable.
6. Run G last.

Commit rule: keep one commit per completed leaf checklist item. If a workflow
finishes multiple leaf items, it must still make one commit per item and update
the checklist in each commit.

## Checklist

### Phase 1: Decision Contract

- [x] Add immutable decision dataclasses:
  - [x] `SignalVote`
  - [x] `RuleDecision`
  - [x] `TradeDecision`
- [x] Add JSON serialization helpers for decision traces.
- [x] Add unit tests for serialization and backward compatibility.
- [x] Add a compatibility wrapper that converts existing boolean regimes into
      default decision traces.

### Phase 2: Atomic Signal Predicates

- [x] Create an atomic predicate registry separate from full strategy regimes.
- [x] Add RSI predicates:
  - [x] `rsi_below`
  - [x] `rsi_above`
- [x] Add EMA predicates:
  - [x] `ema_trend_up`
  - [x] `ema_trend_down`
- [x] Add breakout predicates:
  - [x] `breakout_up`
  - [x] `breakout_failed`
- [x] Add VWAP predicates:
  - [x] `vwap_distance`
  - [x] `vwap_reclaimed`
- [x] Add volume and regime predicates:
  - [x] `relative_volume`
  - [x] `intraday_volatility`
  - [x] `day_type`
- [x] Add no-lookahead tests for every atomic predicate.

### Phase 3: Multi-Signal Rule Evaluation

- [x] Add `multi_signal` normalization and validation.
- [x] Reject entry rules with fewer than 3 signals.
- [x] Reject exit rules with fewer than 3 signals.
- [x] Implement `all`, `any`, and `k_of_n` combiners.
- [x] Canonicalize child signal params for stable hashing and dedupe.
- [x] Compute required history as the max required history across entry and exit
      children.
- [x] Add tests for asymmetric entry and exit rules.
- [x] Add tests that prove changing future bars does not alter current
      decisions.

### Phase 4: Engine Integration

- [x] Add a decision-based execution path to the engine.
- [x] While flat, schedule entry from `entry_rule.passed`.
- [x] While long, schedule exit from `exit_rule.passed`.
- [ ] Preserve hard exits:
  - [ ] stop loss
  - [ ] session close
  - [ ] final bar
- [ ] Preserve one-position-at-a-time behavior.
- [ ] Store entry decision details on the open position.
- [ ] Copy entry and exit decision details into each completed trade.
- [ ] Keep the old regime-based engine path available for legacy strategies.
- [ ] Add focused tests for entry/exit asymmetry and reason persistence.

### Phase 5: Ledger And Artifacts

- [ ] Extend `Trade` payloads with optional decision trace fields.
- [ ] Keep legacy trade payload reads backward compatible.
- [ ] Persist entry and exit votes to `trades.json`.
- [ ] Add compact ledger summaries so decision traces do not bloat the ledger DB.
- [ ] Add tests for old and new artifact payloads.

### Phase 6: Reporting And Visualizer

- [ ] Add stable report path conventions for generated run, dashboard, and
      experiment HTML files.
- [ ] Generate one HTML report per loop run from loop JSON and ledger/artifact
      data.
- [ ] Add a main dashboard page that lists all loop runs and links to generated
      run reports, experiment trade reports, markdown reports, loop JSON, and
      raw artifacts.
- [ ] Add a CLI command to rebuild all generated reporting HTML from existing
      local artifacts.
- [ ] Update the loop command to write or refresh the run HTML and main
      dashboard after each loop run.
- [ ] Update the trade HTML report to show rule-level entry and exit reasons.
- [ ] Add expandable vote details per trade.
- [ ] Show passed and failed child signals distinctly.
- [ ] Keep legacy reports rendering when vote data is missing.
- [ ] Add HTML rendering tests for run reports, dashboard links, and vote
      details.

### Phase 7: Planner And Search Space

- [ ] Add a versioned `multi_signal_v1` search grammar.
- [ ] Generate candidates lazily instead of precomputing the full Cartesian
      product.
- [ ] Generate only 3+ signal entry rules and 3+ signal exit rules.
- [ ] Add shape keys for UCB/search grouping.
- [ ] Add randomized restarts when candidate reuse gets too high.
- [ ] Add Optuna/TPE support for tuning parameters inside a fixed rule shape.
- [ ] Disable 2-signal composite strategies in the active research loop.
- [ ] Keep old `composite` tests as compatibility coverage.
- [ ] Add planner tests proving active multi-signal candidates always satisfy
      the 3+ entry and 3+ exit signal rule.

### Phase 8: Evaluation And Rollout

- [ ] Run focused tests for strategy validation, decisions, engine behavior,
      artifacts, and planner generation.
- [ ] Run the full test suite.
- [ ] Reset stale local research artifacts before measuring new loop behavior.
- [ ] Run a small smoke batch of multi-signal-only experiments.
- [ ] Inspect generated run dashboard, one run report, and trade visualizations
      for at least one winning and one losing strategy.
- [ ] Run 10 loop iterations with `multi_signal` only.
- [ ] Report:
  - [ ] best Sharpe-like score.
  - [ ] best return.
  - [ ] trade count.
  - [ ] duplicate/reused candidate rate.
  - [ ] top-performing rule shapes.
  - [ ] whether improvement is happening run over run.

## Initial Success Criteria

- Multi-signal specs with fewer than 3 entry or exit signals are rejected.
- A strategy can enter from one rule and exit from a different rule.
- Trade artifacts explain both why the trade was entered and why it was exited.
- Each loop run produces a navigable HTML report, and the main dashboard links
  all generated run and experiment HTML files.
- The active loop can generate a large set of valid multi-signal candidates
  without exhausting memory or repeatedly proposing duplicates.
- The full test suite passes after the evaluator, artifact, and planner changes.
