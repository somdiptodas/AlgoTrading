# Autonomous SPY Research System — Canonical Implementation Plan

> Grounded in the current repository as of April 21, 2026. This plan upgrades the existing `Trader` repo into an autonomous, auditable, deterministic-first quantitative research system for SPY strategy discovery. It preserves the strongest parts of the current codebase, enforces strict evaluator boundaries, and delays LLM-driven planning until the research core is correct and reproducible.

---

## 1. Current repo understanding

### 1.1 Current repository structure

The repo is currently a small flat Python project with a script-style layout:

```text
Trader/
├── __main__.py
├── config.py
├── models.py
├── massive_client.py
├── ingest.py
├── storage.py
├── backtest.py
├── visualize.py
├── data/
│   ├── market_data.db
│   ├── ema_baseline_trades.csv
│   └── spy_1_minute_chart.html
└── .gitignore
```

Observed live state:
- The local database contains SPY 1-minute bars only.
- The current dataset spans roughly `2025-10-21` through `2026-04-20` UTC.
- There is no package metadata, no test suite, no experiment tracking, and no autonomous research loop.

### 1.2 Major systems already present

#### Data ingestion
- [ingest.py](./ingest.py) fetches aggregate bars in date windows and writes them to SQLite.
- [massive_client.py](./massive_client.py) wraps the Massive REST client and retries on rate limiting.
- [storage.py](./storage.py) maintains:
  - `aggregate_bars`
  - `ingest_checkpoints`

#### Raw market data storage
- The SQLite schema is clean and usable as the long-term raw market data source.
- Upserts are idempotent.
- Range fetches and SQL-side bucketed fetches already exist.

#### Backtesting
- [backtest.py](./backtest.py) contains a single long-only EMA cross backtest with:
  - signal computed from known bars only
  - entry and exit deferred to the next bar open for signal flips
  - optional regular-session-only filtering
  - optional flat-at-close behavior
  - commission and slippage
  - mark-to-market equity curve
  - trade log export

#### Visualization
- [visualize.py](./visualize.py) renders a self-contained HTML chart from local market data.

### 1.3 What is reusable as-is

- `models.AggregateBar`
- `massive_client.MassiveAggregatesClient`
- Most of `ingest.run_ingest`
- The raw SQLite schema in `storage.py`
- The current execution invariants embedded in `backtest.py`:
  - next-bar execution after signal generation
  - centralized slippage and commission assumptions
  - regular-session filter
  - forced close behavior
  - deterministic single-position bookkeeping

These execution semantics should become the seed of the fixed evaluator. They should not be reimplemented per strategy.

### 1.4 What is partially reusable with refactors

- `backtest.py`
  - Keep the event-loop semantics and PnL accounting.
  - Split out CLI parsing, strategy logic, simulator, metrics, and reporting.
- `config.py`
  - Keep basic path settings.
  - Remove the hard-coded default API key fallback.
  - Split infrastructure settings from research/evaluator settings.
- `visualize.py`
  - Keep as a raw-data viewer initially.
  - Later extend it for experiment reports and overlays.
- `__main__.py`
  - Keep as the top-level CLI entrypoint shape.
  - Replace command internals with package-based commands.

### 1.5 What is missing

The current repo has no:
- typed strategy schema / DSL
- bounded search space
- strategy registry
- fold-based evaluation harness
- leakage-safe feature pipeline
- baseline suite
- robustness checks
- promotion policy
- experiment ledger
- artifact storage
- frontier manager
- deterministic autonomous loop
- LLM planner / generator / critic
- tests
- package metadata

---

## 2. Gap analysis

| Capability | Current state | Target state | Gap | Priority |
| --- | --- | --- | --- | --- |
| Raw market data | SQLite SPY minute bars with checkpointed ingest | Stable source-of-truth data layer with snapshot identity | Need snapshot metadata and typed read access | High |
| Strategy representation | One hard-coded EMA CLI path | Typed declarative `StrategySpec` with bounded fields | No schema, no registry, no validation | High |
| Evaluator | One single-window backtest | Fixed fold-based evaluator with immutable policies | Current backtest is tightly coupled | High |
| Validation | Manual date slicing | Walk-forward / purged time-aware folds | No split engine | High |
| Leakage controls | Implicit for EMA only | Explicit feature/model fit-transform boundaries | No generalized leakage protection | High |
| Execution model | Commission/slippage/next-bar fills in one file | Centralized execution model used by all strategies | Must be extracted and frozen | High |
| Baselines | Informal buy-and-hold comparison in summary | Mandatory baseline suite in every run | Missing reusable baseline runner | High |
| Robustness checks | None | Sensitivity, regime, turnover, stability checks | Missing | Medium |
| Experiment tracking | Optional CSV trade log | Ledger DB + artifacts + lineage + dedupe | Missing | High |
| Reporting | Console output + raw chart | Experiment reports + ledger/frontier reports | Missing | Medium |
| Search loop | Manual CLI invocations | Deterministic autonomous search first | Missing | High |
| LLM autonomy | None | Provider-agnostic planner/critic later | Missing, but intentionally deferred | Medium |
| Promotion gates | None | Human-defined candidate promotion workflow | Missing | Medium |
| Packaging/tests | None | Installable package + regression/unit tests | Missing | High |
| Secrets hygiene | Hard-coded API key fallback | Env var only | Current config is unsafe | High |

---

## 3. Proposed target architecture

### 3.1 High-level architecture

The target system should be a real Python package with explicit boundaries:

```text
Trader/
├── pyproject.toml
├── README.md
├── trader/
│   ├── __init__.py
│   ├── __main__.py
│   ├── config.py
│   ├── cli/
│   │   ├── __init__.py
│   │   ├── ingest_cmd.py
│   │   ├── visualize_cmd.py
│   │   ├── eval_cmd.py
│   │   ├── ledger_cmd.py
│   │   └── loop_cmd.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── models.py
│   │   ├── massive_client.py
│   │   ├── ingest.py
│   │   ├── storage.py
│   │   └── view.py
│   ├── execution/
│   │   ├── __init__.py
│   │   ├── engine.py
│   │   ├── fills.py
│   │   └── position.py
│   ├── strategies/
│   │   ├── __init__.py
│   │   ├── spec.py
│   │   ├── registry.py
│   │   ├── signals/
│   │   │   ├── ema_cross.py
│   │   │   └── breakout.py
│   │   ├── sizers/
│   │   │   └── full_notional.py
│   │   └── filters/
│   │       └── session.py
│   ├── features/
│   │   ├── __init__.py
│   │   ├── primitives.py
│   │   └── pipeline.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── splits.py
│   │   ├── runner.py
│   │   ├── metrics.py
│   │   ├── baselines.py
│   │   ├── robustness.py
│   │   └── promotion.py
│   ├── ledger/
│   │   ├── __init__.py
│   │   ├── entry.py
│   │   ├── store.py
│   │   └── query.py
│   ├── artifacts/
│   │   ├── __init__.py
│   │   └── store.py
│   ├── reporting/
│   │   ├── __init__.py
│   │   ├── visualize.py
│   │   └── report.py
│   └── research/
│       ├── __init__.py
│       ├── planner.py
│       ├── generator.py
│       ├── critic.py
│       ├── frontier.py
│       └── prompts/
├── tests/
└── data/
    ├── market_data.db
    └── research/
        ├── ledger.db
        ├── artifacts/
        └── reports/
```

### 3.2 Responsibilities of each module

#### `trader.data`
- Owns raw market bars and ingestion.
- Exposes a read-only `DataView`.
- Computes deterministic snapshot identities for evaluation windows.

#### `trader.execution`
- Owns the frozen strategy-agnostic simulator.
- Owns fills, slippage, commissions, and position accounting.
- This is a human-controlled correctness boundary.

#### `trader.strategies`
- Defines the declarative search space.
- Registers allowed signal, sizing, and filter handlers.
- The agent can only choose from validated registered handlers.

#### `trader.features`
- Computes leak-safe features.
- Any trainable transform must fit only on training windows and apply to OOS windows.

#### `trader.evaluation`
- Owns folds, metrics, baselines, robustness checks, and promotion gates.
- This is the only legal path to evaluate a strategy.

#### `trader.ledger`
- Stores every attempted experiment, including failures.
- Handles dedupe, lineage, summary queries, and critique records.

#### `trader.artifacts`
- Stores heavy outputs on disk:
  - resolved spec
  - metrics JSON
  - trades
  - equity curves
  - reports

#### `trader.reporting`
- Generates per-run and cross-run human-readable reports.

#### `trader.research`
- Owns research loop logic.
- Phase 4 uses a deterministic planner.
- LLM-backed planning and criticism come later, through the same interfaces.

### 3.3 Data flow through the research loop

```text
raw bars
  -> DataView / snapshot
  -> StrategySpec proposal
  -> evaluator
       -> splitter
       -> features
       -> execution engine
       -> metrics + baselines
       -> robustness checks
  -> ledger entry + artifacts
  -> frontier update
  -> critique / next proposal
```

### 3.4 Human-owned vs agent-controlled boundaries

#### Human-owned and hard to mutate
- `trader.execution.*`
- `trader.evaluation.splits`
- `trader.evaluation.metrics`
- `trader.evaluation.baselines`
- `trader.evaluation.robustness`
- `trader.evaluation.promotion`
- `trader.data.*`
- schema of `StrategySpec`

#### Agent-controlled
- Which valid `StrategySpec` to try next
- Which part of the search space to explore
- Critiques and follow-up hypotheses
- Report narratives

#### Explicit rule
The agent may search **within** the schema. It may not rewrite the evaluator, cost model, split logic, or raw data.

---

## 4. Phased roadmap

### Phase 0 — Repo hardening and package reorg

**Objective**
- Convert the repo into a real installable package.
- Preserve behavior while making the codebase safe to extend.

**Deliverables**
- Add `pyproject.toml`
- Create `trader/` package and move current modules into package submodules
- Remove all `try/except ModuleNotFoundError` import shims
- Remove `DEFAULT_MASSIVE_API_KEY`
- Add initial tests and README
- Keep `market_data.db` untouched

**Files/modules to create or modify**
- Create:
  - `pyproject.toml`
  - `README.md`
  - `trader/__init__.py`
  - `trader/__main__.py`
  - `trader/cli/*`
  - `tests/conftest.py`
  - `tests/test_backtest_regression.py`
  - `tests/test_storage.py`
  - `tests/test_leakage.py`
- Move/refactor:
  - `config.py -> trader/config.py`
  - `models.py -> trader/data/models.py`
  - `massive_client.py -> trader/data/massive_client.py`
  - `ingest.py -> trader/data/ingest.py`
  - `storage.py -> trader/data/storage.py`
  - `backtest.py -> trader/backtest.py` as temporary compatibility shim
  - `visualize.py -> trader/reporting/visualize.py`

**Exit criteria**
- `pip install -e .` works
- Current CLI behavior still works through the new package
- Regression tests lock current EMA behavior

**Manual testing**
- Re-run the current EMA backtest and confirm matching metrics
- Re-run ingest and confirm checkpoint updates

### Phase 1 — Fixed experiment pipeline

**Objective**
- Extract the simulator and evaluator boundary without changing current trading semantics.

**Deliverables**
- Strategy-agnostic execution engine
- Fixed evaluator entrypoint
- Fold-based evaluation support
- Standard metrics module
- Standard baseline module

**Files/modules to create or modify**
- Create:
  - `trader/execution/engine.py`
  - `trader/execution/fills.py`
  - `trader/execution/position.py`
  - `trader/evaluation/splits.py`
  - `trader/evaluation/runner.py`
  - `trader/evaluation/metrics.py`
  - `trader/evaluation/baselines.py`
- Modify:
  - `trader/backtest.py`
  - `trader/__main__.py`
  - `trader/cli/eval_cmd.py`

**Exit criteria**
- One-fold eval reproduces legacy EMA behavior
- Multi-fold eval runs through the same evaluator
- Baselines are attached to each run

**Manual testing**
- One-fold parity
- 3-fold and 5-fold walk-forward runs
- Verify no overlap or lookahead across folds

### Phase 2 — Typed strategy specs and bounded search space

**Objective**
- Replace free-form strategy logic with a declarative schema and registry.

**Deliverables**
- `StrategySpec`
- signal/sizing/filter registries
- schema validation
- deterministic spec hashing
- support for two rules-based families in v1:
  - `ema_cross`
  - `breakout`

**Files/modules to create or modify**
- Create:
  - `trader/strategies/spec.py`
  - `trader/strategies/registry.py`
  - `trader/strategies/signals/ema_cross.py`
  - `trader/strategies/signals/breakout.py`
  - `trader/strategies/sizers/full_notional.py`
  - `trader/strategies/filters/session.py`
- Add spec loading support to CLI

**Exit criteria**
- Evaluator accepts validated specs instead of bespoke backtest flags
- Invalid specs fail before evaluation
- Same spec produces same canonical hash

**Manual testing**
- Run EMA and breakout through the same evaluator
- Verify malformed specs are rejected

### Phase 3 — Feature pipeline, ledger, artifacts, and reporting

**Objective**
- Make experiments reproducible, auditable, and comparable.

**Deliverables**
- Leak-safe feature primitives and pipeline
- Separate research ledger DB
- Artifact store
- Experiment reports and ledger reports
- Duplicate experiment detection

**Files/modules to create or modify**
- Create:
  - `trader/features/primitives.py`
  - `trader/features/pipeline.py`
  - `trader/ledger/entry.py`
  - `trader/ledger/store.py`
  - `trader/ledger/query.py`
  - `trader/artifacts/store.py`
  - `trader/reporting/report.py`
- Modify:
  - `trader/data/view.py`
  - `trader/reporting/visualize.py`
  - `.gitignore`

**Exit criteria**
- Every run writes:
  - ledger row
  - fold metrics
  - artifacts
  - data snapshot id
  - git SHA
  - environment hash
- Duplicate spec/data/eval combinations are skipped or reused

**Manual testing**
- Re-run identical experiment and verify dedupe
- Open a report and confirm fold and baseline sections are correct

### Phase 4 — Deterministic autonomous loop

**Objective**
- Add a practical autonomous loop without introducing LLM dependence.

**Deliverables**
- Deterministic planner
- Generator/validator
- Frontier manager
- Heuristic critic
- Loop CLI

**Files/modules to create or modify**
- Create:
  - `trader/research/planner.py`
  - `trader/research/generator.py`
  - `trader/research/frontier.py`
  - `trader/research/critic.py`
  - `trader/cli/loop_cmd.py`

**Behavior**
- The first autonomous loop should be deterministic.
- It should search bounded neighborhoods or finite grids, for example:
  - EMA lengths
  - breakout windows
  - signal thresholds
  - session filter combinations
- It should use frontier and ledger history to avoid re-running duplicates.

**Exit criteria**
- Running the loop produces a reproducible batch of experiments
- No invalid proposal reaches the evaluator
- Repeating the same deterministic loop yields the same ordered experiment set

**Manual testing**
- Run the loop twice with same planner settings and compare outputs
- Verify frontier updates deterministically

### Phase 5 — Robustness and promotion workflow

**Objective**
- Prevent best-backtest selection from dominating the process.

**Deliverables**
- parameter neighborhood scan
- regime decomposition
- stability checks
- promotion gates
- candidate states such as:
  - `exploratory`
  - `frontier`
  - `candidate`
  - `promoted_to_paper`

**Files/modules to create or modify**
- Create:
  - `trader/evaluation/robustness.py`
  - `trader/evaluation/promotion.py`
- Modify:
  - `trader/evaluation/runner.py`
  - `trader/reporting/report.py`
  - `trader/research/frontier.py`

**Exit criteria**
- Fragile strategies fail robustness gates
- Candidates cannot promote without passing explicit OOS criteria

**Manual testing**
- Hand-tuned peaky parameter set should fail neighborhood stability
- Baseline-like strategies should fail “meaningful improvement” gates

### Phase 6 — LLM planner / critic

**Objective**
- Introduce LLM assistance only after the deterministic research core is working.

**Deliverables**
- provider-agnostic planner interface
- prompt templates
- structured LLM outputs
- prompt/result hashing in ledger

**Files/modules to create or modify**
- Extend:
  - `trader/research/planner.py`
  - `trader/research/critic.py`
  - `trader/research/prompts/*`

**Exit criteria**
- LLM planner can be swapped in without changing evaluator or ledger interfaces
- Deterministic planner remains available as fallback

---

## 5. File-by-file implementation plan

| File or directory | Action | Why | What should change |
| --- | --- | --- | --- |
| `__main__.py` | Replace via package entrypoint | Current CLI router is too thin and tied to flat layout | Move logic into `trader/__main__.py` and `trader/cli/*` |
| `config.py` | Refactor | Current settings are too narrow and unsafe | Split runtime settings from evaluation settings; delete hard-coded API key |
| `models.py` | Move and keep | Good raw DTO | Move to `trader/data/models.py` |
| `massive_client.py` | Move and keep | Existing adapter is fine | Move to `trader/data/massive_client.py` |
| `ingest.py` | Move and refactor lightly | Ingest flow is reusable | Move to `trader/data/ingest.py`; add snapshot metadata/logging |
| `storage.py` | Move and refactor lightly | Raw data store is good | Move to `trader/data/storage.py`; keep bar store separate from research ledger |
| `backtest.py` | Replace by extraction | Too much coupling | Keep compatibility shim; move core logic to `execution` + `evaluation` + `strategies` |
| `visualize.py` | Move and extend | Useful viewer, wrong module location | Move to `trader/reporting/visualize.py`; later overlay experiments |
| `.gitignore` | Update | Need explicit research output ignores | Ignore `data/research/`, reports, artifacts |
| `data/market_data.db` | Keep | Existing raw data source | No schema rewrite early |
| `data/ema_baseline_trades.csv` | Deprecate | One-off output should become artifactized | Move future outputs into `data/research/artifacts/` |
| `pyproject.toml` | Add | Installability and tests require it | Define dependencies and package entrypoints |
| `README.md` | Add | Architecture and usage need documentation | Include core boundaries and workflows |
| `tests/` | Add | No safety net exists today | Add regression, leakage, split, metrics, ledger, registry tests |
| `trader/data/` | Add | Package raw data concerns cleanly | Move current data-related modules here |
| `trader/execution/` | Add | Need fixed simulator boundary | Extract event loop, fills, positions |
| `trader/strategies/` | Add | Need bounded search and handler registry | Add spec schema and handlers |
| `trader/features/` | Add | Need leak-safe features | Add primitives and pipeline |
| `trader/evaluation/` | Add | Need fixed evaluation harness | Add splits, metrics, baselines, robustness, promotion |
| `trader/ledger/` | Add | Need durable experiment history | Add ledger entry/store/query |
| `trader/artifacts/` | Add | Keep bulky outputs off DB | Add artifact writer and layout |
| `trader/reporting/` | Add | Need experiment and ledger reports | Add report generation and charting |
| `trader/research/` | Add | Need autonomous research orchestration | Add planner/generator/critic/frontier |

---

## 6. Core interfaces

### 6.1 Strategy spec

```python
from dataclasses import dataclass, field
from typing import Literal

@dataclass(frozen=True)
class SignalSpec:
    name: str
    params: dict[str, int | float | str | bool]

@dataclass(frozen=True)
class SizingSpec:
    name: str
    params: dict[str, int | float | str | bool] = field(default_factory=dict)

@dataclass(frozen=True)
class FilterSpec:
    name: str
    params: dict[str, int | float | str | bool] = field(default_factory=dict)

@dataclass(frozen=True)
class ExecConfig:
    initial_cash: float = 100_000.0
    commission_per_order: float = 0.0
    slippage_bps: float = 1.0
    regular_session_only: bool = True
    flat_at_close: bool = True

@dataclass(frozen=True)
class StrategySpec:
    name: str
    instrument: Literal["SPY"] = "SPY"
    multiplier: int = 1
    timespan: Literal["minute"] = "minute"
    signal: SignalSpec = field(default_factory=lambda: SignalSpec("ema_cross", {}))
    sizing: SizingSpec = field(default_factory=lambda: SizingSpec("full_notional"))
    filters: tuple[FilterSpec, ...] = ()
    exec_config: ExecConfig = field(default_factory=ExecConfig)
    feature_set: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    seed: int = 0
```

Rules:
- Canonical JSON serialization
- Stable hash over canonical JSON
- Only registry-known handlers are valid
- No arbitrary Python or code generation in strategy definitions

### 6.2 Evaluator input/output

```python
@dataclass(frozen=True)
class EvaluatorInput:
    experiment_id: str
    data_snapshot_id: str
    split_plan_id: str
    cost_model_id: str
    baseline_ids: tuple[str, ...]
    spec: StrategySpec
    artifact_dir: str

@dataclass(frozen=True)
class FoldResult:
    fold_id: str
    train_start_utc: str
    train_end_utc: str
    test_start_utc: str
    test_end_utc: str
    metrics: dict[str, float]
    baseline_deltas: dict[str, float]
    warnings: tuple[str, ...] = ()

@dataclass(frozen=True)
class ExperimentResult:
    experiment_id: str
    status: str
    aggregate_metrics: dict[str, float]
    fold_results: tuple[FoldResult, ...]
    robustness_checks: dict[str, float | bool]
    artifact_paths: dict[str, str]
```

### 6.3 Ledger entry

```python
@dataclass(frozen=True)
class LedgerEntry:
    experiment_id: str
    spec_hash: str
    parent_experiment_ids: tuple[str, ...]
    created_at_utc: str
    generator_kind: str
    data_snapshot_id: str
    split_plan_id: str
    cost_model_id: str
    git_sha: str
    env_hash: str
    status: str
    summary_metrics: dict[str, float]
    novelty_key: str
    artifact_paths: dict[str, str]
```

### 6.4 Frontier record

```python
@dataclass(frozen=True)
class FrontierRecord:
    experiment_id: str
    family: str
    score_vector: dict[str, float]
    dominates_baseline: bool
    stability_pass: bool
    promotion_stage: str
```

### 6.5 Data view

```python
class DataView:
    def bars(self, ticker: str, start_ms: int, end_ms: int) -> list[object]: ...
    def snapshot_hash(self) -> str: ...
```

This must be the only data interface the evaluator and strategies use.

---

## 7. Main risks and failure modes

### Leakage risk
- Future-aware features or fold-wide normalization can invalidate all results.
- Mitigation:
  - train-only fit / test-only transform discipline
  - synthetic leakage tests
  - read-only `DataView`

### Invalid walk-forward splits
- Overlapping train/test or insufficient embargo can create hidden leakage.
- Mitigation:
  - explicit fold objects
  - split validation tests
  - embargo minimum tied to max holding assumptions

### Hidden coupling from current backtest
- `backtest.py` currently mixes parsing, strategy logic, execution, metrics, and reporting.
- Mitigation:
  - extract simulator first
  - keep backtest CLI only as temporary shim

### Overly flexible agent behavior
- If the agent can write evaluator code, it will optimize the harness.
- Mitigation:
  - agent only emits validated specs
  - evaluator modules treated as human-owned

### Duplicate experiments
- Autonomous search will rediscover equivalent specs.
- Mitigation:
  - canonical spec hash
  - dedupe in generator and ledger
  - novelty keys and parent lineage

### Poor reproducibility
- Missing git SHA, data snapshot, env hash, or seed will make results hard to trust.
- Mitigation:
  - all four stored on every run
  - artifacts include resolved spec and manifest

### Slow feedback loops
- Rich reports and baselines can slow throughput.
- Mitigation:
  - keep v1 feature set small
  - separate heavy reporting from core evaluation path where possible
  - cache repeated baseline computations per data/split policy when safe

### Metrics that incentivize overfitting
- Raw return or headline Sharpe alone will reward fragile ideas.
- Mitigation:
  - rank on OOS fold metrics, stability, and baseline-relative value
  - require promotion gates later

### Thin historical coverage
- Current repo only has about six months of SPY minute data.
- Mitigation:
  - expand history before taking any model comparison seriously
  - treat early rankings as harness validation, not alpha discovery

### SQLite contention
- Concurrent experiment writes can lock the ledger.
- Mitigation:
  - keep v1 concurrency low
  - serialize ledger writes
  - consider WAL mode later if needed

---

## 8. Recommended implementation order

### Exact build order
1. Package reorg and repo hardening
2. Regression and leakage tests
3. Extract execution engine
4. Add fold-based evaluator
5. Add typed strategy specs and registry
6. Add feature pipeline and baseline suite
7. Add ledger and artifacts
8. Add reports and frontier queries
9. Add deterministic autonomous loop
10. Add robustness and promotion workflow
11. Add LLM planner and critic only after the loop is stable

### Week-by-week suggestion
- **Week 1**
  - Phase 0 package reorg
  - remove secret fallback
  - baseline regression tests
- **Week 2**
  - extract execution engine
  - keep legacy backtest shim
- **Week 3**
  - build splits, evaluator, metrics, baselines
- **Week 4**
  - build spec schema, registry, EMA + breakout
- **Week 5**
  - build features, ledger, artifacts
- **Week 6**
  - build reports and deterministic loop
- **Week 7**
  - build robustness and promotion
- **Week 8+**
  - provider-agnostic LLM planner / critic

---

## 9. Minimal v1

The smallest useful version that is still architecturally correct:

### Scope
- SPY only
- 1-minute bars only
- One raw data source: existing SQLite DB
- Two strategy families:
  - EMA cross
  - simple breakout
- One sizing rule:
  - full notional
- One execution policy:
  - current next-bar-open, slippage, commission, regular-session, flat-at-close
- Fixed walk-forward folds
- Baselines:
  - buy-and-hold
  - always-flat
- Metrics:
  - OOS return
  - Sharpe-like risk-adjusted metric
  - max drawdown
  - exposure
  - trade count
  - win rate
  - profit factor
- Separate ledger DB
- Artifact store
- Deterministic planner:
  - finite grids and local neighborhoods only
- Frontier summary over OOS metrics and stability

### What v1 deliberately excludes
- no LLM planner
- no ML model families
- no paper trading
- no multi-asset support
- no agent-authored new feature primitives or handlers

### V1 success criteria
- One command runs:
  - spec generation
  - validation
  - evaluation
  - ledger write
  - artifact write
- Same loop input produces same experiment ordering and same dedupe behavior
- The system is trustworthy enough to compare bounded rules-based SPY ideas without changing evaluator rules

---

## 10. Stretch roadmap

### V2 — richer search without changing core boundaries
- Add more strategy families
- Add richer regime filters
- Add more feature subsets
- Add ensemble combinations
- Add better frontier scoring
- Add deterministic critic heuristics driven by experiment history

### V2.5 — provider-agnostic LLM support
- Add LLM planner and critic behind stable interfaces
- Pass only summary ledger context into prompts
- Record prompt hashes, model id, and structured outputs
- Keep deterministic planner as fallback and regression oracle

### V3 — production-adjacent research workflow
- Promotion to paper trading
- paper-trading event trace and reconciliation
- multi-asset support
- more realistic execution assumptions
- stronger guardrails:
  - daily experiment budget
  - human approval for new handlers
  - prompt/result audit trail

---

## 11. Recommendations: what stays fixed vs what the agent may modify

### Must remain fixed by humans
- raw market data ingestion
- cost model
- split policy
- evaluator interfaces
- baseline set
- metric definitions
- promotion policy
- feature primitive implementations
- strategy handler implementations

### May be generated or modified by the agent
- valid `StrategySpec` instances
- bounded parameter sweeps
- experiment rationale
- critiques and follow-up suggestions
- report text and summaries

### May be added by the agent later, but only through human review
- new strategy handlers
- new feature primitives
- new model-family wrappers

---

## 12. Canonical design rules

1. The evaluator is more important than the planner.
2. The strategy search space must be typed and bounded.
3. The first autonomous loop must be deterministic.
4. LLMs are an optimization layer, not a foundation.
5. Every experiment must be reproducible from ledger metadata and artifacts.
6. Raw backtest return is never enough; promotion depends on OOS robustness.
7. The agent’s leverage is in choosing the next valid spec, not rewriting the judge.

---

## Appendix A — Prompt boundaries for later LLM integration

These are deferred until Phase 6, but the interfaces should be designed now.

### Planner input
- registered handlers and parameter ranges
- recent experiment summaries
- current frontier
- recent critiques

### Planner output
- structured spec proposals only
- bounded rationale text

### Critic input
- one experiment result
- nearest historical neighbors
- fold, baseline, and robustness summaries

### Critic output
- structured verdict
- overfitting risk
- regime dependence summary
- next-search suggestions

All prompts and outputs should be hashed and stored once LLM support is added.

---

## Appendix B — Dependency recommendations

### Minimum
- runtime:
  - `massive`
  - `urllib3`
- dev:
  - `pytest`

### Reasonable additions
- `polars` for feature and metric tables
- optional lightweight TOML support if needed for spec files

### Avoid in v1
- `mlflow`
- `wandb`
- heavy ML libraries
- broad autonomous code generation

---

## TL;DR

Reorganize the repo into a proper `trader/` package first. Extract the current backtest’s execution semantics into a frozen, strategy-agnostic evaluator. Add typed strategy specs, fold-based evaluation, baselines, ledger, artifacts, and reporting. Build the first autonomous loop as a deterministic planner over a bounded search space. Only after that loop is reproducible and trustworthy should LLM-based planning and criticism be introduced.
