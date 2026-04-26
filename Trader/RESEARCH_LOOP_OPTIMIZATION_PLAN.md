# Research Loop Optimization Plan

Date: 2026-04-25 (rev 2 — Claude audit + multi-signal direction)

## Audit Snapshot (what the system currently shows)

Ledger state (from `data/research/ledger.db`):

| Family | Completed | Avg return | Best return | Avg trades | Avg DD | Avg Δ buy-hold |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `breakout` | 26 | -3.76% | +2.11% | 370 | 8.4 | -10.6 pp |
| `ema_cross` | 24 | -4.89% | +0.18% | 350 | 9.3 | -12.2 pp |
| `rsi_reversion` | 29 | -2.08% | +2.10% | 164 | 5.1 | -10.6 pp |
| `vwap_deviation` | 11 | -5.07% | -1.58% | 518 | 6.9 | -13.1 pp |

By generator kind:

| Kind | Count | Avg return |
| --- | ---: | ---: |
| `frontier_neighborhood` | 47 | **-1.50%** |
| `confirmation_grid` | 13 | -5.03% |
| `grid` | 30 | -6.51% |

Headline observations:

- 90 completed experiments, **0 promoted** (no `candidate`, only legacy `frontier`).
- **0/90** beat buy-and-hold; the closest was -10.6 pp.
- `frontier_neighborhood` is by far the strongest generator — the loop's local search around best-so-far is doing real work; raw `grid` is wasting most of its budget.
- Best raw returns are RSI reversion specs with ~20–90 trades. The high-turnover regime (>500 trades) is uniformly catastrophic.
- ~5.9 GB of artifacts for 90 experiments (~65 MB each) — `equity.json` and `result.json` per experiment are the bulk.
- Market data covers 2024-04-25 → 2026-04-24 (≈417 K minute bars; ≈194 K regular-session bars).

## What The System Does Well

- Clean, evaluator-frozen package boundaries; agent only emits validated `StrategySpec`s.
- Leakage hygiene is solid: rolling highs/lows are exclusive of the current bar, EMA/RSI use only history+test-up-to-i, VWAP uses the current bar's close (acceptable for next-bar entry on minute-close decisions), explicit leakage tests exist for breakout/VWAP/regime filters.
- Walk-forward folds are deterministic, embargo-validated, holdout-locked; aggregate metrics are bar-count-weighted (P0 fix from rev 1 landed).
- Ledger compaction worked: 716 KB DB for 90 rows, with indexed scalar columns for ranking.
- Suppressor uses registry-derived parameter scales now (P2 fix landed) and is auditable per loop run with `audit_type` separation.
- Decay monitor exists, holdout policy enforced, paper-trading boundary scaffolding ready.
- Strong regression coverage for engine, costs, regimes, splits, leakage, robustness, holdout, decay, and paper-trading risk/data-integrity.

## The Biggest Gaps And Risks

### G1. The promotion bar may be structurally unwinnable
SPY 2024–2026 has been a near-monotonic bull. Any long-only minute strategy that is not always-on caps its upside relative to buy-and-hold. The current `candidate` gate requires `delta_buy_and_hold_return_pct > 0.5` — that is asking a discretionary signal to beat a 100%-exposed long during a bull. **No long-only timing strategy will pass this in a bull regime**, regardless of how good the signal is.

Implication: the loop will never promote anything until either (a) the data window includes a meaningful drawdown period, (b) the benchmark is exposure-adjusted, or (c) shorting / hedging is allowed. This is the #1 reason 0/90 promote.

### G2. Compute is wasted on candidates that already failed base metrics
Each evaluated candidate runs:

- 3 folds × (1 strategy + 5 baselines + 3 cost-stress) = **27 backtests**, plus
- 6 robustness neighbors × 3 folds × 4 backtests (strategy + 3 cost-stress) = **72 backtests**.

That's ≈ 99 backtests per candidate, ~95 s/candidate. Most candidates have negative return after the first fold — the remaining work is wasted. Confirmed by the rev 1 timing probe (`94.8 s` with robustness vs `15.7 s` without).

### G3. No first-class composite/multi-signal support
The current "composite" path is `confirmation_grid` — a single primary signal masked by a regime filter. There is no way to express:

- Majority vote across `{ema_cross, rsi_reversion, vwap_deviation}`.
- `ema_cross AND breakout` as peers.
- Weighted ensemble where each child contributes a confidence score.
- Switching ensembles (use mean-reversion in chop, trend-follow in trending days).

The user's expectation is right: the best SPY 1-minute edge is almost certainly a *combination* of weak signals that together filter to high-quality trades. The current grid keeps grinding on isolated single-signal variants.

### G4. Search is uniform-grid + local neighborhood — no learning across runs
- Planner just enumerates a Cartesian product of `signal_grid × sizing × exec × filter`.
- No Bayesian / surrogate model uses prior outcomes to bias new sampling.
- Family balancing is binary ("more for under-explored families"), not a UCB/Thompson bandit.
- Critic feedback feeds *family-level* penalties only; it does not propagate per-parameter-region preferences (e.g. "RSI length ∈ [12,18] is consistently better than [21,28]").

### G5. Benchmarks and metrics inflate the perceived gap to alpha
- `delta_buy_and_hold_return_pct` is computed against a 100%-long benchmark even when the strategy is in market only 5–30% of the time. A strategy with `+1% return @ 8% exposure` is destroyed by `+12% B&H @ 100% exposure` on raw-delta but has a far better return-per-bar-in-market.
- No information ratio, no Calmar, no per-day-Sharpe (annualized Sharpe is computed over per-minute returns including bars where the strategy is flat — that's a noise floor).
- `randomized_entry_same_exposure` is the right idea but is computed once per evaluation (not bootstrapped) so it has no statistical power.

### G6. Feature/regime/baseline computation is not cached across specs
For a given fold (snapshot, range), `buy_and_hold`, `regular_session_open_to_close_long`, `session_long_flat_at_close`, EMA series for length=20, RSI for length=14, etc. are recomputed for every spec/sizing/exec combo. With 6 specs × 3 folds × multiple lookback values per loop run, this is a 5–10× wasted-CPU multiplier.

### G7. Signal/filter implementations are pure Python loops
`ema`, `rsi`, `rolling_max_exclusive`, `_intraday_realized_volatility_bps`, day-type stats are all O(N) Python over lists of ~60–80 K bars per fold. NumPy vectorization is a routine 10–50× speedup here.

### G8. Robustness check is parameter-axis only
`assess_robustness` checks 6 fixed parameter-axis neighbors. It does not:
- Block-bootstrap test windows to measure return distribution (no p-value).
- Stress data (random gaps, missing minutes) to test execution robustness.
- Stress costs beyond +2bps slippage / +2bps spread.
- Test holdout-window neighborhoods (only research-fold neighborhoods).

### G9. Artifact bloat (~5.9 GB / 90 runs)
`equity.json` is per-minute equity for every fold of every neighbor backtest. `result.json` includes all that plus trades. 1-minute equity is rarely useful at audit time; 1-hour or per-day equity is plenty. Cost stress backtests don't need full equity stored.

### G10. vwap_deviation is structurally bad-shape
Generates 400–700 trades over a few months with no per-day cap, no cooldown after exit, no minimum bar gap. Every minute close < VWAP triggers a re-entry, then exit, repeat. The signal *family* is not the problem; the *implementation* needs entry cooldown and a per-day trade cap.

### G11. Frontier signal is weak when nothing is promoted
`frontier_specs` for the planner comes from `promoted_experiments`, but no rows are promoted — so frontier-neighborhood seeding falls back to `research_frontier`/`frontier` legacy entries (good). But this means the local-search engine has only ~5 reasonable seeds and they get crowded out by grid plans whenever the planner overplans.

### G12. Loop is single-process
3 folds × 4 backtests × 7 specs (1 + 6 neighbors) per candidate × 6 candidates = ~500 backtests per loop, all serial. An 8-core M-series Mac is using one core.

## Where The Loop Wastes Time (concrete hotspots)

| Hotspot | Wasted-work cause | Cheap fix |
| --- | --- | --- |
| Robustness neighbors run for losers | No early-exit on bad candidate | Skip neighbors when stage-1 metrics fail |
| Cost stress runs every fold for every spec | Always-on in `_evaluate_fold` | Only run cost stress for promotion candidates |
| Baselines recomputed per spec | `evaluate_baselines` called inside fold loop | Cache `(snapshot_id, fold_id) → baselines` (only `randomized_entry` is spec-dependent) |
| EMA/RSI/rolling recomputed per spec | `FeaturePipeline` is constructed per call | Cache `(snapshot_id, fold_range, feature, params)` for indicator series |
| `snapshot_hash` recomputed in `_split_research_and_holdout` | Hashes 194K bars twice per spec | Cache by `(start_idx, end_idx, base_snapshot_id)` |
| Pure-Python loops for indicators | List ops, no NumPy | Vectorize ema/rsi/rolling/realized-vol |
| Single process | No parallelism | ProcessPool over selected candidates |
| Grid plans dominate frontier seeds | Round-robin keeps adding grid even when frontier neighborhoods are hotter | Allocate batch slots by per-family expected reward (UCB), not equal share |
| Confirmation grid + grid produce 1500+ planned specs only to drop most | `overplan_factor=12` is too generous | Drop to 4–6 once cheap-rank uses live history |

## Multi-Signal Strategy: Concrete Design

Goal: combine multiple signals as **peers** (not just primary + filter), with no lookahead and minimal new evaluator surface.

### M1. CompositeSignal as a first-class signal handler
Add a new signal handler `composite` whose `params` are:

```python
{
  "combiner": "all" | "any" | "vote_k_of_n" | "primary_plus_confirmations",
  "min_agreeing": int,            # for vote_k_of_n
  "primary_index": int,           # for primary_plus_confirmations
  "children": [                   # 2-4 entries; each is a SignalSpec payload
    {"name": "ema_cross", "params": {...}},
    {"name": "rsi_reversion", "params": {...}},
    ...
  ],
}
```

`composite.required_history` = `max(child.required_history for child)`.
`composite.generate_regime` calls each child's `generate_regime(history, test, child_params)` and folds the per-bar booleans through the combiner. This reuses every existing signal's leakage discipline without changes.

### M2. Combiners (deterministic, no fitting)
- `all` — long only when every child is long. Good for AND-style high-conviction entries.
- `any` — long when any child is long. Good for opportunistic systems if cost is low.
- `vote_k_of_n` — long when ≥ k of n children are long. Robust ensemble vote (k=⌈n/2⌉).
- `primary_plus_confirmations` — child[primary_index] must be long *and* every other child must agree. Generalizes the existing `confirmation_grid`.

### M3. Composite spec hashing
Reuse `StrategySpec.canonical_json` on the children inside `params` (children are already plain dicts). Hash stays deterministic.

### M4. Composite parameter neighborhood
For `composite.neighbors`, walk a single child at a time (one child per neighbor variant) so neighbor count stays bounded (≤ 6). This keeps robustness affordable.

### M5. Composite planner grid (small, evidence-driven)
Start with these 6 canonical composites, NOT a Cartesian product:

1. `ema_cross AND rsi_reversion` (trend + pullback)
2. `breakout AND relative_volume>=1.5 AND day_type=trend` (high-volume breakout in trend day)
3. `vwap_deviation AND rsi_oversold` (mean-reversion confirmation)
4. `majority(ema_cross, breakout, rsi_reversion)` (vote ensemble)
5. `ema_cross AND NOT day_type=trend` (counter-trend in chop)
6. `breakout BUT first_30m_window AND relative_volume>=1.5` (open-range breakout)

Each composite gets ~3 parameter variants from the underlying child grids — total ≤ 24 composite specs per planning batch.

### M6. Leakage tests for composites
Add a leakage test that perturbs *future* test bars and verifies `composite.generate_regime[k]` is unchanged for all k ≤ perturbation index, by reusing the child-signal leakage tests transitively.

### M7. (Stretch) Fitted ensemble weights — explicitly out of v1
A weighted/learned ensemble (logistic regression on child outputs trained on training fold, applied to test fold) is a natural V2 once the deterministic composite path is solid. Defer until composites are evaluated.

## Faster Strategy Search: Concrete Levers

### S1. Two-stage evaluation (biggest single win)
1. **Stage A — fast pre-screen** (target ≤ 3 s/spec):
   - 1 fold (most recent) only.
   - No baselines, no cost stress, no neighbor robustness.
   - Compute: return, trade count, max DD, exposure, sharpe-like, monthly concentration.
   - Reject early on: non-positive return, trade count out of `[10, 400]`, drawdown > 25%, exposure > 90% (no edge — just B&H), exposure < 1% (no signal).
2. **Stage B — full evaluation** only for stage-A survivors:
   - Full 3-fold walk-forward with baselines, cost stress, robustness neighbors, holdout if promoted.

Expected: with current ~80% candidate failure rate, loop drops from ~10 min → ~2–3 min for the same final selection.

### S2. Adaptive search mixer (replaces uniform grid)
- Keep grid only for *seed* points (~3 per family).
- After ~10 evaluations per family, switch the family's bucket from grid to a Bayesian sampler (Optuna TPE or simple GP/CMA-ES) over the same parameter ranges. Use ledger return as the objective.
- A UCB1/Thompson bandit allocates batch slots across families based on each family's recent best return + uncertainty. Replaces today's hand-tuned `family_quota_boost = (max_count - count) * 3.0`.

### S3. Feature + baseline + snapshot caches
- `_indicator_cache: dict[(snapshot_id, start_idx, end_idx, name, params)] -> ndarray` — keyed cache for `ema`, `rsi`, `rolling_max_exclusive`, etc.
- `_baseline_cache: dict[(snapshot_id, fold_id)] -> dict[name, dict]` — only `randomized_entry_same_exposure` stays per-spec; the rest are spec-independent.
- `_snapshot_subhash_cache` for `_split_research_and_holdout` so we don't re-hash 194K bars per spec.

### S4. Vectorize indicators
Rewrite `ema`, `rsi`, `rolling_max_exclusive`, `_intraday_realized_volatility_bps`, `_session_progress_stats` to NumPy. Wins ≥ 10× per call; biggest impact when feature cache misses.

### S5. Conditional cost stress
Move `_cost_scenario_metrics` out of `_evaluate_fold` and only call it after stage B, only for candidates that pass robustness.

### S6. Process-pool parallelism
Selected candidates are independent. Run stage B in a `ProcessPoolExecutor(max_workers=min(8, len(candidates)))`. Main process collects results and writes ledger/artifacts (avoids SQLite write contention).

### S7. Reduce neighbor count from 6 → 3 with weighted sampling
Pick 3 most informative axis-neighbors (highest parameter delta) instead of all 6. Use median-of-neighbors with bootstrap CI for stability check.

### S8. Drop overplan factor
Once stage-A pre-screen exists, `overplan_factor` of 4 is enough. Today's default 12 is paying preview costs that pre-screen makes unnecessary.

### S9. Block-bootstrap p-value for promotion
Instead of `randomized_entry_same_exposure` once, draw 500 random-entry baselines (block-resampled by trading day) and report the p-value of the strategy's return distribution. Use this as a promotion gate ("p < 0.10") instead of, or in addition to, `delta_buy_and_hold > 0.5%`.

### S10. Down-sample equity for storage
Store equity as per-day or per-30-min, not per-minute. Drop 90%+ of artifact bytes. Per-minute can be reconstructed from trades + bars on demand.

## Better Validation And Metrics

### V1. Exposure-adjusted benchmark
Add `delta_exposure_adjusted_buy_and_hold_pct` = strategy return − (exposure_pct × B&H_return_pct). A 5%-exposure strategy is now compared to 5% of B&H, not 100%. Promotion gate becomes `> 0` on this metric instead of raw delta.

### V2. Information ratio vs B&H
Per-trading-day P&L vector for both strategy and B&H; report `mean(diff) / std(diff) * sqrt(252)`. Add to promotion gate as `IR > 0.5`.

### V3. Per-trade quality metrics
Currently win_rate and profit_factor are present. Add: average bars-held, P&L distribution skew, maximum consecutive losses. These distinguish "lucky big winner" from "consistent small edge".

### V4. Regime-conditional returns
Split per-day P&L by realized regime (trend/chop/high-vol/low-vol) and report return-by-regime. A strategy that only works in chop is fine if scoped that way; today the metric averages those away.

### V5. Holdout p-value
Same block-bootstrap on holdout. Promotion to `candidate` requires holdout p < 0.10 *and* directional match with research folds.

### V6. Concentration check on top-N trades
If top-3 trades produce >50% of P&L, mark fragile. Today this is approximated only at the *monthly* level.

## Future-Trading Readiness (already partially in place)

The paper-trading scaffold is solid. Next:

- F1. Live signal warm-up / online state: refactor each `generate_regime` to also expose a streaming `update(bar) -> regime_state` so paper trading uses the exact same code path as research with no fork.
- F2. Latency budget: instrument from `bar_close → signal → order_submitted` and assert under 250 ms in tests.
- F3. End-of-day reconciliation: nightly job that re-runs the day's signals through the research evaluator on the same bars and asserts identical regime sequences vs the live audit log.
- F4. Strategy decay should run nightly with `--decay-promoted` and post a status to `data/research/reports/decay_*.md`.

## Prioritized Checklist

Ordered by impact-per-day-of-work. P0 = blocker, P1 = high impact, P2 = improves quality, P3 = enables new modes.

### P0 — fix what we burn time/money on now

- [x] **Stage-A cheap pre-screen before Stage-B robustness** (G2, S1). Single-fold, no baselines, no cost stress; reject on return ≤ 0, trade count ∉ `[10, 400]`, drawdown > 25%, exposure ∉ `[1%, 90%]`. Wire into `EvaluationRunner.evaluate_preview`.
  - Completed: 2026-04-25
  - Implementation: `EvaluationRunner.evaluate_preview` now runs a most-recent-fold Stage-A gate before full folds/robustness when robustness is enabled; Stage-A rejects return exploratory results without baselines, cost stress, robustness neighbors, or holdout.
  - Verification: focused runner/queue tests passed (`.venv/bin/pytest -q tests/test_robustness.py tests/test_holdout.py tests/test_research_queue.py`); full suite passed (`.venv/bin/pytest -q`); read-only verifier re-check found no blockers.
- [x] **Conditional cost stress** (S5). Move `_cost_scenario_metrics` out of `_evaluate_fold`; run only after Stage B passes base gates.
  - Completed: 2026-04-25
  - Implementation: `_evaluate_fold` now produces base metrics and baselines only; cost-stress metrics are added after robustness/promotion gating only for non-exploratory Stage-B results, and fold reports omit fake `cost_drag=0` when stress was skipped.
  - Verification: focused cost/runner/report tests passed (`.venv/bin/pytest -q tests/test_costs.py tests/test_robustness.py tests/test_ledger.py tests/test_holdout.py tests/test_research_queue.py`); full suite passed (`.venv/bin/pytest -q`); read-only verifier found no blockers.
- [x] **Per-(snapshot, fold) baseline cache** (S3). `buy_and_hold`, `regular_session_open_to_close_long`, `session_long_flat_at_close`, `always_flat` are spec-independent; cache them per fold per loop process.
  - Completed: 2026-04-25
  - Implementation: `EvaluationRunner` now caches fixed fold baselines per snapshot, fold window, and cost model while keeping `randomized_entry_same_exposure` per spec; fold-result caching is also snapshot-scoped to avoid stale reuse.
  - Verification: focused baseline/runner tests passed (`.venv/bin/pytest -q tests/test_baselines.py tests/test_data_quality.py tests/test_robustness.py tests/test_costs.py tests/test_research_queue.py`); full suite passed (`.venv/bin/pytest -q`); read-only verifier re-check found no blockers.
- [x] **Per-(snapshot, fold) indicator cache** (S3). EMA/RSI/rolling-high/rolling-low keyed by `(snapshot_id, fold_range, feature, params)`. Targets the biggest hit-rate when grids share lookback values.
  - Completed: 2026-04-25
  - Implementation: `EvaluationRunner` now owns a fold-scoped indicator cache used by `FeaturePipeline` for EMA, RSI, rolling-high, and rolling-low test slices; cached values are immutable internally and scoped by snapshot plus fold train/test window.
  - Verification: focused indicator/leakage/runner tests passed (`.venv/bin/pytest -q tests/test_indicator_cache.py tests/test_leakage.py tests/test_robustness.py tests/test_costs.py tests/test_baselines.py`); full suite passed (`.venv/bin/pytest -q`); read-only verifier found no blockers.
- [x] **Cap planner overplan factor at 4** once stage-A exists (S8). Reduce wasted preview/eval-key compute.
  - Completed: 2026-04-25
  - Implementation: loop default `DEFAULT_OVERPLAN_FACTOR` is now 4 while preserving the explicit `--overplan-factor` override and minimum planned-spec floor.
  - Verification: focused loop CLI tests passed (`.venv/bin/pytest -q tests/test_loop_cmd.py`); full suite passed (`.venv/bin/pytest -q`); read-only verifier found no blockers.
- [x] **Add `delta_exposure_adjusted_buy_and_hold_pct` and use it as the candidate gate** (V1). The current `delta_buy_and_hold > 0.5` is structurally unwinnable for low-exposure strategies in a bull market. This is the *single biggest reason* nothing has promoted.
  - Completed: 2026-04-25
  - Implementation: baseline deltas now include exposure-adjusted buy-and-hold edge, reports label it, the critic uses it when present, and promotion gates candidates on positive exposure-adjusted edge instead of raw buy-and-hold delta.
  - Verification: focused promotion/metric/ledger/runner tests passed (`.venv/bin/pytest -q tests/test_promotion.py tests/test_splits_metrics_registry.py tests/test_research_queue.py tests/test_ledger.py tests/test_costs.py tests/test_robustness.py tests/test_holdout.py`); full suite passed (`.venv/bin/pytest -q`); read-only verifier found no blockers.
- [x] **Fix vwap_deviation churn** (G10). Add `min_bars_between_entries` (cooldown) and `max_entries_per_session` to the signal params; default cooldown 30 bars, max 4 entries/day.
  - Completed: 2026-04-25
  - Implementation: `vwap_deviation` now normalizes and emits explicit churn-control params, enforces cooldown after exits, caps signal entries per session, resets state on session changes, and preserves cooldown/cap through planner grids and neighbors.
  - Verification: focused VWAP/planner/leakage tests passed (`.venv/bin/pytest -q tests/test_vwap_deviation.py tests/test_planner.py tests/test_leakage.py`); full suite passed (`.venv/bin/pytest -q`); read-only verifier re-check found no blockers.
- [x] **Add per-phase timing JSON** to loop output (rev 1 P0 carryover): planning, key compute, preview, queue scoring, stage A, stage B, robustness neighbors, artifact write, ledger write.
  - Completed: 2026-04-25
  - Implementation: loop output now includes `timings_sec` with stable phase keys for planning, evaluation-key compute, preview, queue scoring, Stage A, Stage B, robustness neighbors, artifact writes, and ledger writes.
  - Verification: focused timing/queue/runner tests passed (`.venv/bin/pytest -q tests/test_loop_cmd.py tests/test_research_queue.py tests/test_robustness.py tests/test_costs.py`); full suite passed (`.venv/bin/pytest -q`); read-only verifier found no blockers.

### P1 — make the loop a real ensemble factory

- [x] **Add `composite` signal handler** with `all` / `any` / `vote_k_of_n` / `primary_plus_confirmations` combiners (M1, M2). Reuse child signals as-is.
  - Completed: 2026-04-25
  - Implementation: Added a first-class `composite` signal handler that normalizes 2-4 child signal payloads, combines existing child regimes with deterministic `all`, `any`, `vote_k_of_n`, and `primary_plus_confirmations` modes, reports max child required history, and emits bounded one-child-at-a-time robustness neighbors.
  - Verification: focused composite/registry/leakage tests passed (`.venv/bin/pytest -q tests/test_composite_signal.py tests/test_splits_metrics_registry.py tests/test_leakage.py`); full suite passed (`.venv/bin/pytest -q`); read-only verifier found no blockers.
- [x] **Composite leakage tests** (M6) by transitively asserting children's leakage tests still hold under composition.
  - Completed: 2026-04-25
  - Implementation: Added composite prefix-invariance coverage that mutates only future test bars and asserts prior outputs remain unchanged across `all`, `any`, `vote_k_of_n`, and `primary_plus_confirmations` composites using real child signals.
  - Verification: focused composite/leakage tests passed (`.venv/bin/pytest -q tests/test_composite_signal.py tests/test_leakage.py`); full suite passed (`.venv/bin/pytest -q`); read-only verifier re-check found no blockers.
- [x] **Composite planner bucket** with the 6 canonical composites in M5 (NOT a full cross-product). Cap composite specs at ~24 per loop.
  - Completed: 2026-04-25
  - Implementation: Added a curated `composite_grid` planner path with six canonical recipes and three deterministic variants each, split into per-recipe buckets so the default loop budget reaches every recipe before raw grid expansion; `--signal-family composite` is now accepted and `composite.parameter_grid()` remains empty to avoid a full cross-product.
  - Verification: focused planner/composite/loop tests passed (`.venv/bin/pytest -q tests/test_planner.py tests/test_composite_signal.py tests/test_loop_cmd.py`); full suite passed (`.venv/bin/pytest -q`); read-only verifier re-check found no blockers.
- [x] **Replace family-quota boost with a UCB bandit allocator** (S2) keyed on per-family best-return and evaluation count.
  - Completed: 2026-04-25
  - Implementation: Replaced count-gap quota and family-quality scoring with a deterministic family UCB score based on best historical `return_pct` and evaluation count; cheap preview ordering and final candidate allocation both use virtual same-batch counts so preview budget and selected batches are allocated by the bandit score.
  - Verification: focused queue tests passed (`.venv/bin/pytest -q tests/test_research_queue.py`); full suite passed (`.venv/bin/pytest -q`); read-only verifier re-check found no blockers.
- [x] **Optuna/TPE sampler per family after seed grid is exhausted** (S2). Ledger-return as objective; persist study state in `data/research/optuna/`.
  - Completed: 2026-04-26
  - Implementation: `DeterministicPlanner` now adds per-family `optuna_tpe` candidate buckets after 10 completed seed evaluations, uses completed ledger `return_pct` as the Optuna objective, persists per-family study state under `data/research/optuna/`, and wires the loop to pass ledger history plus the Optuna storage directory while preserving queue dedupe/scoring and no-lookahead evaluation boundaries.
  - Verification: focused planner/loop/queue tests passed (`.venv/bin/pytest -q tests/test_planner.py tests/test_loop_cmd.py tests/test_research_queue.py`); full suite passed (`.venv/bin/pytest -q`); read-only verifier re-check found no blockers.
- [x] **Add information ratio metric** vs B&H per-trading-day (V2). Promotion gate: `IR > 0.5` *and* exposure-adjusted delta > 0.
  - Completed: 2026-04-26
  - Implementation: Added `information_ratio_vs_buy_and_hold` from per-session active returns versus buy-and-hold, recomputed aggregate IR from pooled fold backtests, surfaced the metric in reports, and required IR > 0.5 alongside positive exposure-adjusted buy-and-hold edge for candidate promotion.
  - Verification: focused metric/promotion/cost/robustness/queue/holdout tests passed (`.venv/bin/pytest -q tests/test_costs.py tests/test_splits_metrics_registry.py tests/test_promotion.py tests/test_robustness.py tests/test_research_queue.py tests/test_holdout.py`); full suite passed (`.venv/bin/pytest -q`); read-only verifier found no blockers.
- [x] **Block-bootstrap p-value baseline** replacing single random-entry (S9, V5). 500 daily-block resamples; report `p_value_vs_random_entry`.
  - Completed: 2026-04-26
  - Implementation: Random-entry baselines now compute a deterministic 500-sample daily-block bootstrap p-value, aggregate research p-values from fold bootstrap samples instead of averaging fold p-values, gate candidates on `p_value_vs_random_entry < 0.10`, demote research candidates when holdout p-value or directional match fails, and skip bootstrap baselines for robustness-neighbor metrics.
  - Verification: focused baseline/promotion/cost/holdout/robustness/queue tests passed (`.venv/bin/pytest -q tests/test_baselines.py tests/test_promotion.py tests/test_costs.py tests/test_holdout.py tests/test_robustness.py tests/test_research_queue.py`); full suite passed (`.venv/bin/pytest -q`); read-only verifier re-check found no blockers.
- [x] **Vectorize indicator primitives** to NumPy (S4). `ema`, `rsi`, `rolling_max_exclusive`, `rolling_min_exclusive`, `_intraday_realized_volatility_bps`, `_session_progress_stats`.
  - Completed: 2026-04-25
  - Implementation: Added NumPy as a runtime dependency, moved EMA/RSI/rolling primitive internals onto NumPy arrays while preserving list/`None` return contracts, vectorized exclusive rolling extrema, and rewrote intraday-volatility/session-progress helpers with array math scoped to current sessions.
  - Verification: focused primitive/leakage/regime/cache tests passed (`.venv/bin/pytest -q tests/test_feature_primitives.py tests/test_leakage.py tests/test_regime_filters.py tests/test_indicator_cache.py`); full suite passed (`.venv/bin/pytest -q`); read-only verifier found no blockers.
- [x] **Process-pool parallel Stage B** (S6). 4–8 workers, single-writer ledger.
  - Completed: 2026-04-25
  - Implementation: Stage-B candidate evaluation now runs through a 4-8 worker `ProcessPoolExecutor`; each worker opens its own `EvaluationRunner` against the same database and returns only the `ExperimentResult` plus phase timings, while critique, artifact writes, ledger writes, and suppression audit logging remain single-writer in the parent process.
  - Verification: focused loop/queue/robustness/ledger tests passed (`.venv/bin/pytest -q tests/test_loop_cmd.py tests/test_research_queue.py tests/test_robustness.py tests/test_ledger.py`); throwaway real ProcessPool smoke passed against a temporary SQLite DB; full suite passed (`.venv/bin/pytest -q`); read-only verifier found no blockers.
- [x] **Down-sample stored equity to 30-minute granularity** (S10). Per-minute reconstructable from `trades.json`.
  - Completed: 2026-04-25
  - Implementation: Artifact `result.json` and `equity.json` now store sampled equity points at the first bar, each 30-minute regular-session offset, and the final bar, with sampling metadata/source counts; default serializers remain full-resolution unless artifacts opt into sampling, and `trades.json` now uses the central trade serializer so cost fields needed for reconstruction are retained.
  - Verification: focused ledger/artifact tests passed (`.venv/bin/pytest -q tests/test_ledger.py`); full suite passed (`.venv/bin/pytest -q`); read-only verifier found and then re-checked the `trades.json` reconstruction blocker with no blockers remaining.

### P2 — better selection signal & reuse across runs

- [x] **Persist per-region critic memory** to disk so the next loop's planner can avoid bad regions across processes (today's `_critic_penalty_by_family` is only family-level).
  - Completed: 2026-04-25
  - Implementation: Added persistent `critic_memory.json` with deterministic per-region records derived from completed critique planning penalties, load/seed/rewrite wiring in the loop, and candidate-queue scoring that uses region penalties before falling back to family-level critic penalties when no memory exists.
  - Verification: focused critic-memory/queue/loop tests passed (`.venv/bin/pytest -q tests/test_critic_memory.py tests/test_research_queue.py tests/test_loop_cmd.py`); full suite passed (`.venv/bin/pytest -q`); read-only verifier found and then re-checked the startup-load blocker with no blockers remaining.
- [x] **Within-run failure memory** (rev 1 P2 carryover): after first 2 stage-A failures from a region, suppress remaining stage-B from that region in the *same* loop.
  - Completed: 2026-04-25
  - Implementation: Added serial Stage-A prescreening before process-pool Stage B, an in-memory same-region Stage-A failure suppressor using existing normalized parameter geometry, distinct `stage_a_suppressed` audit logging for selected candidates skipped before Stage B, and Stage-A pass carry-forward for survivors evaluated by workers.
  - Verification: focused suppressor/loop/robustness/queue tests passed (`.venv/bin/pytest -q tests/test_suppressor.py tests/test_loop_cmd.py tests/test_robustness.py tests/test_research_queue.py`); full suite passed (`.venv/bin/pytest -q`); read-only verifier found and then re-checked Stage-A pass/audit-type blockers with no blockers remaining.
- [x] **Holdout p-value** (V5) and direction-match gate before `candidate` promotion.
  - Completed: 2026-04-25
  - Implementation: Candidate promotion now fails closed when holdout bars are absent, and the runner keeps candidates only when holdout `p_value_vs_random_entry < 0.10` and holdout return direction matches positive research return.
  - Verification: focused holdout/promotion/robustness/cost tests passed (`.venv/bin/pytest -q tests/test_holdout.py tests/test_promotion.py tests/test_robustness.py tests/test_costs.py`); full suite passed (`.venv/bin/pytest -q`); read-only verifier found and then re-checked the missing-holdout fail-closed blocker with no blockers remaining.
- [x] **Regime-conditional return reporting** (V4).
  - Completed: 2026-04-25
  - Implementation: Added reporting-only online-safe trend/chop/high-vol/low-vol labels, per-session compounded return/day-count metrics by regime, fold/holdout wiring, aggregate compounding across folds, and report labels/formatting for the new metrics.
  - Verification: focused metrics/regime/runner/report tests passed (`.venv/bin/pytest -q tests/test_splits_metrics_registry.py tests/test_regime_filters.py tests/test_robustness.py tests/test_ledger.py`); full suite passed (`.venv/bin/pytest -q`); read-only verifier found no blockers in the V4 implementation and confirmed unrelated follow-up/data working-tree changes should stay out of this commit.
- [x] **Top-N trade concentration check** (V6) added to robustness.
  - Completed: 2026-04-25
  - Implementation: Added a fail-closed top-3 positive trade PnL concentration robustness gate, promotion requirement, report fields, and suppressor weighting for nearby failures.
  - Verification: focused robustness/promotion/suppressor/report/cost tests passed (`.venv/bin/pytest -q tests/test_robustness.py tests/test_promotion.py tests/test_suppressor.py tests/test_ledger.py tests/test_costs.py`); full suite passed (`.venv/bin/pytest -q`); read-only verifier found no blockers in the V6 implementation and confirmed unrelated follow-up/data working-tree changes should stay out of this commit.
- [x] **Snapshot subhash cache in `_split_research_and_holdout`** (S3) so research/holdout snapshot IDs aren't re-hashed per spec.
  - Completed: 2026-04-25
  - Implementation: Added a per-runner subhash cache keyed by full snapshot ID and exclusive slice bounds, and reused it for research and holdout snapshot IDs without changing split boundaries.
  - Verification: focused holdout/cache tests passed (`.venv/bin/pytest -q tests/test_holdout.py tests/test_indicator_cache.py tests/test_baselines.py`); full suite passed (`.venv/bin/pytest -q`); read-only verifier found no blockers in the S3 implementation and confirmed unrelated follow-up/data working-tree changes should stay out of this commit.
- [ ] **Reduce neighbor count to 3** with median + bootstrap CI (S7).

### P3 — open new search modes

- [ ] **Long+short or hedged variants**. Currently long-only; explore signed regime that goes flat-vs-short on bearish signals. Even just "short on `breakout AND day_type=mean_reversion`" expands the alpha surface meaningfully.
- [ ] **Fitted-weight ensembles** (M7) once deterministic composites are solid. Logistic regression on child boolean outputs trained on each fold's train slice.
- [ ] **Multi-fold warm restart for Optuna** so sampler state survives across loop runs.
- [ ] **Streaming `update(bar)` API** on every signal (F1) for paper-trading parity.
- [ ] **Latency budget assertion** (F2).
- [ ] **Nightly decay + reconciliation jobs** (F3, F4).
- [ ] **Tighten artifact retention**: keep only top-K by recent return + all promoted; archive others to compressed tarballs.

## Recommended Implementation Order

Do these in order; each step makes the next cheaper or more informative.

1. **P0 Stage-A pre-screen + conditional cost stress + baseline/indicator caches** — unblocks everything; loop becomes 3–5× faster on bad batches.
2. **P0 Exposure-adjusted gate + per-phase timing** — makes promotion possible and makes optimization measurable.
3. **P0 vwap_deviation cooldown** — stops a known-bad family from polluting results.
4. **P1 Composite signal + canonical composite planner bucket** — opens the multi-signal search the user explicitly asked for.
5. **P1 Information ratio + bootstrap p-value** — makes "is this real edge?" answerable.
6. **P1 NumPy vectorization + ProcessPool parallelism** — pure throughput win, no semantic change.
7. **P1 UCB family bandit + Optuna per-family sampler** — turns the loop from grid search into adaptive search.
8. **P2 Within-run failure memory + persistent critic + concentration checks** — better quality at the same throughput.
9. **P3 Long/short variants + fitted ensembles** — once we know the composite framework is sound.

## Definition Of "Done" For A Promotable Strategy (revised)

A strategy promotes only if it satisfies:

- All current robustness gates (`fold_consistency_pass`, `regime_pass`, `neighborhood_pass`, `drawdown_pass`).
- Positive aggregate OOS return AND positive `annualized_sharpe`.
- `trade_count ∈ [10, 400]` (lower bound prevents tiny-sample flukes; upper bound prevents cost-driven losers).
- `delta_exposure_adjusted_buy_and_hold_pct > 0` (new — replaces raw `delta_buy_and_hold > 0.5`).
- `information_ratio_vs_buy_and_hold > 0.5` (new).
- `bootstrap_p_value_vs_random_entry < 0.10` (new — block-bootstrap by day).
- Holdout pass: `bootstrap_p_value_holdout < 0.10` AND directionally consistent with research folds.
- Cost stress (+2 bps slip / +2 bps spread) does not flip return sign.
