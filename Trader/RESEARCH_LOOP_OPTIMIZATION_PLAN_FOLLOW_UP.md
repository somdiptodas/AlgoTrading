# Research Loop Optimization Follow-Up

- [ ] Consider updating ledger/top-experiment scalar ranking to use `delta_exposure_adjusted_buy_and_hold_pct` once enough new rows include it. Current SQLite ranking still uses raw `delta_buy_and_hold_return_pct`, which can keep favoring high-exposure bull-market strategies even though promotion now gates on exposure-adjusted edge.
- [ ] Add an explicit negated filter mode so `ema_cross AND NOT day_type=trend` can be represented exactly. The first composite planner bucket uses the existing `day_type=mean_reversion` filter as the smallest safe approximation.
- [ ] Benchmark the NumPy indicator path on real fold sizes and consider a dedicated EMA/RSI recurrence accelerator if those recursive primitives remain a hotspot after rolling-window and regime-helper vectorization.
- [ ] Benchmark process-pool Stage B with real batch sizes to confirm preview/result pickling overhead does not erase the expected wall-clock speedup.
- [ ] Decide whether `trades.json` should become a standalone reconstruction artifact; today exact per-minute equity reconstruction still needs bars/config from `result.json` in addition to the trade cost fields now stored in `trades.json`.
- [ ] Consider merging or rebuilding stale/malformed `critic_memory.json` from the ledger before queue scoring instead of using the persisted file as-is for that run.
- [ ] Add a full loop integration test that verifies a mixed batch writes both Stage-A-passed survivor artifacts and `stage_a_suppressed` suppression-log rows.

- [ ] Add a runner-level regression test proving aggregate `information_ratio_vs_buy_and_hold` is recomputed from pooled daily active returns across fold backtests, not averaged from fold-level IR values.
