# Trader

Deterministic-first quantitative research system for bounded SPY strategy discovery.

The repository keeps raw market data in `data/market_data.db` and writes research outputs to:

- `data/research/ledger.db`
- `data/research/artifacts/`
- `data/research/reports/`

## Core boundaries

- Raw market data, execution semantics, split policy, metrics, baselines, robustness checks, and promotion rules are fixed and human-owned.
- The autonomous loop may only generate validated `StrategySpec` instances inside the registered search space.
- Phase 6 LLM planning is intentionally not implemented in this pass.

## Install

```bash
python3 -m pip install -e ".[dev]"
```

## CLI

```bash
python3 -m trader ingest --ticker SPY --months-back 6
python3 -m trader visualize --ticker SPY --days 5
python3 -m trader backtest --strategy ema_cross
python3 -m trader eval --spec-json '{"name":"ema_default","signal":{"name":"ema_cross","params":{"fast_length":20,"slow_length":80,"signal_buffer_bps":0.0}}}'
python3 -m trader ledger summary
python3 -m trader loop --batch-size 6 --folds 3
```

## Minimal v1 scope

- SPY only
- 1-minute bars only
- Strategy families: `ema_cross`, `breakout`
- Sizing rule: `full_notional`
- Fixed execution policy: next-bar open fills, commission/slippage, regular session only, flat at close
- Fixed walk-forward evaluator with `buy_and_hold` and `always_flat` baselines
- Separate ledger DB and artifact store
- Deterministic bounded search loop

## Testing

```bash
pytest
```
