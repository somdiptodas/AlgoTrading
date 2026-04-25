# Research Loop Optimization Follow-Up

- [ ] Consider updating ledger/top-experiment scalar ranking to use `delta_exposure_adjusted_buy_and_hold_pct` once enough new rows include it. Current SQLite ranking still uses raw `delta_buy_and_hold_return_pct`, which can keep favoring high-exposure bull-market strategies even though promotion now gates on exposure-adjusted edge.
