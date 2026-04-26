# Research Loop Optimization Follow-Up

- [ ] Consider updating ledger/top-experiment scalar ranking to use `delta_exposure_adjusted_buy_and_hold_pct` once enough new rows include it. Current SQLite ranking still uses raw `delta_buy_and_hold_return_pct`, which can keep favoring high-exposure bull-market strategies even though promotion now gates on exposure-adjusted edge.
- [ ] Add an explicit negated filter mode so `ema_cross AND NOT day_type=trend` can be represented exactly. The first composite planner bucket uses the existing `day_type=mean_reversion` filter as the smallest safe approximation.

- [ ] Add a runner-level regression test proving aggregate `information_ratio_vs_buy_and_hold` is recomputed from pooled daily active returns across fold backtests, not averaged from fold-level IR values.
