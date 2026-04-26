from __future__ import annotations

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from trader.data.models import MarketBar
from trader.data.view import DataView
from trader.evaluation.runner import EvaluationRunner
from trader.strategies.decisions import SignalVote
from trader.strategies.registry import REGISTRY
from trader.strategies.signals import multi_signal
from trader.strategies.spec import SignalSpec, StrategySpec


NEW_YORK = ZoneInfo("America/New_York")


def _bar(index: int, close: float, *, volume: float = 1_000.0, vwap: float | None = 100.0) -> MarketBar:
    timestamp = datetime(2026, 1, 5, 9, 30, tzinfo=NEW_YORK) + timedelta(minutes=index)
    timestamp_utc = timestamp.astimezone(timezone.utc)
    return MarketBar(
        timestamp_ms=int(timestamp_utc.timestamp() * 1000),
        timestamp_utc=timestamp_utc.isoformat(),
        open=close,
        high=close + 0.25,
        low=close - 0.25,
        close=close,
        volume=volume,
        vwap=vwap,
    )


def _multi_signal_params() -> dict[str, object]:
    return {
        "entry_rule": {
            "combiner": "all",
            "signals": [
                {"name": "rsi_below", "params": {"length": "2", "threshold": "30"}},
                {"name": "ema_trend_up", "params": {"fast": "2", "slow": "3"}},
                {"name": "vwap_distance", "params": {"side": "below", "min_bps": "10"}},
            ],
        },
        "exit_rule": {
            "combiner": "any",
            "signals": [
                {"name": "rsi_above", "params": {"length": "2", "threshold": "70"}},
                {"name": "ema_trend_down", "params": {"fast": "2", "slow": "3"}},
                {"name": "vwap_reclaimed", "params": {"min_bps": "0"}},
            ],
        },
    }


class _FakePredicates:
    def __init__(self, votes: dict[str, list[bool]]) -> None:
        self.votes = votes

    def validate_params(self, name: str, params: dict[str, object]) -> dict[str, object]:
        if name not in self.votes:
            raise ValueError(f"Unknown atomic predicate: {name}")
        return dict(params)

    def generate_votes(
        self,
        name: str,
        history_bars: tuple[MarketBar, ...],
        test_bars: tuple[MarketBar, ...],
        params: dict[str, object],
    ) -> list[SignalVote]:
        return [
            SignalVote(name, passed, f"{name} vote {index}")
            for index, passed in enumerate(self.votes[name][: len(test_bars)])
        ]


def test_multi_signal_normalizes_rules_and_child_predicate_params() -> None:
    validated = REGISTRY.validate_spec(
        StrategySpec(
            name="multi_signal_validation",
            signal=SignalSpec("multi_signal", _multi_signal_params()),
        )
    )

    assert validated.signal.params == {
        "entry_rule": {
            "combiner": "all",
            "signals": [
                {"name": "ema_trend_up", "params": {"fast": 2, "slow": 3, "buffer_bps": 0.0}},
                {"name": "rsi_below", "params": {"length": 2, "threshold": 30.0}},
                {
                    "name": "vwap_distance",
                    "params": {"side": "below", "min_bps": 10.0, "max_bps": 100_000.0},
                },
            ],
        },
        "exit_rule": {
            "combiner": "any",
            "signals": [
                {"name": "ema_trend_down", "params": {"fast": 2, "slow": 3, "buffer_bps": 0.0}},
                {"name": "rsi_above", "params": {"length": 2, "threshold": 70.0}},
                {"name": "vwap_reclaimed", "params": {"min_bps": 0.0}},
            ],
        },
    }


def test_multi_signal_validation_rejects_bad_rule_shapes() -> None:
    with pytest.raises(ValueError, match="multi_signal.entry_rule must be a rule payload"):
        REGISTRY.validate_spec(
            StrategySpec(
                name="bad_entry_rule",
                signal=SignalSpec("multi_signal", {"exit_rule": _multi_signal_params()["exit_rule"]}),
            )
        )

    params = _multi_signal_params()
    entry_rule = dict(params["entry_rule"])  # type: ignore[arg-type]
    entry_rule["signals"] = [
        {"name": "missing", "params": {}},
        {"name": "ema_trend_up", "params": {"fast": 2, "slow": 3}},
        {"name": "vwap_distance", "params": {"side": "below"}},
    ]
    params["entry_rule"] = entry_rule
    with pytest.raises(ValueError, match="Unknown atomic predicate"):
        REGISTRY.validate_spec(
            StrategySpec(
                name="bad_child_predicate",
                signal=SignalSpec("multi_signal", params),
            )
        )


def test_multi_signal_rejects_entry_rules_with_fewer_than_three_signals() -> None:
    params = _multi_signal_params()
    entry_rule = dict(params["entry_rule"])  # type: ignore[arg-type]
    entry_rule["signals"] = entry_rule["signals"][:2]  # type: ignore[index]
    params["entry_rule"] = entry_rule

    with pytest.raises(ValueError, match="entry_rule\\.signals must contain at least 3"):
        REGISTRY.validate_spec(
            StrategySpec(
                name="short_entry_rule",
                signal=SignalSpec("multi_signal", params),
            )
        )


def test_multi_signal_rejects_exit_rules_with_fewer_than_three_signals() -> None:
    params = _multi_signal_params()
    exit_rule = dict(params["exit_rule"])  # type: ignore[arg-type]
    exit_rule["signals"] = exit_rule["signals"][:2]  # type: ignore[index]
    params["exit_rule"] = exit_rule

    with pytest.raises(ValueError, match="exit_rule\\.signals must contain at least 3"):
        REGISTRY.validate_spec(
            StrategySpec(
                name="short_exit_rule",
                signal=SignalSpec("multi_signal", params),
            )
        )


@pytest.mark.parametrize(
    ("combiner", "extra_params", "expected"),
    (
        ("all", {}, [True, False, False]),
        ("any", {}, [True, True, True]),
        ("k_of_n", {"k": 2}, [True, True, True]),
        ("k_of_n", {"k": 3}, [True, False, False]),
    ),
)
def test_multi_signal_combines_child_votes_into_trade_decisions(
    monkeypatch: pytest.MonkeyPatch,
    combiner: str,
    extra_params: dict[str, object],
    expected: list[bool],
) -> None:
    monkeypatch.setattr(
        multi_signal,
        "PREDICATES",
        _FakePredicates(
            {
                "a": [True, False, True],
                "b": [True, True, False],
                "c": [True, True, True],
                "x": [False, False, False],
                "y": [False, True, False],
                "z": [False, False, True],
            }
        ),
    )
    bars = tuple(_bar(index, 100.0 + index) for index in range(3))
    spec = StrategySpec(
        name=f"multi_signal_{combiner}",
        signal=SignalSpec(
            "multi_signal",
            {
                "entry_rule": {
                    "combiner": combiner,
                    **extra_params,
                    "signals": [{"name": "a", "params": {}}, {"name": "b", "params": {}}, {"name": "c", "params": {}}],
                },
                "exit_rule": {
                    "combiner": "any",
                    "signals": [{"name": "x", "params": {}}, {"name": "y", "params": {}}, {"name": "z", "params": {}}],
                },
            },
        ),
    )

    decisions = REGISTRY.generate_decisions(spec, tuple(), bars)

    assert [decision.entry.passed for decision in decisions] == expected
    assert [vote.name for vote in decisions[0].entry.votes] == ["a", "b", "c"]
    assert decisions[0].entry.votes[0].detail == "a vote 0"
    assert decisions[0].entry.reason.startswith(f"{combiner} passed: 3/3 signals")
    assert [decision.exit.passed for decision in decisions] == [False, True, True]


def test_multi_signal_rejects_invalid_k_of_n_rules() -> None:
    params = _multi_signal_params()
    entry_rule = dict(params["entry_rule"])  # type: ignore[arg-type]
    entry_rule["combiner"] = "k_of_n"
    entry_rule["k"] = 4
    params["entry_rule"] = entry_rule

    with pytest.raises(ValueError, match="multi_signal\\.k must be between 1 and signal count"):
        REGISTRY.validate_spec(
            StrategySpec(
                name="bad_k_of_n",
                signal=SignalSpec("multi_signal", params),
            )
        )


def test_multi_signal_supports_asymmetric_entry_and_exit_rules(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        multi_signal,
        "PREDICATES",
        _FakePredicates(
            {
                "entry_a": [True, False, True],
                "entry_b": [True, True, False],
                "entry_c": [True, True, True],
                "exit_a": [False, False, True],
                "exit_b": [False, True, False],
                "exit_c": [False, False, False],
            }
        ),
    )
    bars = tuple(_bar(index, 100.0 + index) for index in range(3))
    spec = StrategySpec(
        name="multi_signal_asymmetric_rules",
        signal=SignalSpec(
            "multi_signal",
            {
                "entry_rule": {
                    "combiner": "all",
                    "signals": [
                        {"name": "entry_a", "params": {}},
                        {"name": "entry_b", "params": {}},
                        {"name": "entry_c", "params": {}},
                    ],
                },
                "exit_rule": {
                    "combiner": "any",
                    "signals": [
                        {"name": "exit_a", "params": {}},
                        {"name": "exit_b", "params": {}},
                        {"name": "exit_c", "params": {}},
                    ],
                },
            },
        ),
    )

    decisions = REGISTRY.generate_decisions(spec, tuple(), bars)

    assert [decision.entry.passed for decision in decisions] == [True, False, False]
    assert [decision.exit.passed for decision in decisions] == [False, True, True]
    assert [vote.name for vote in decisions[0].entry.votes] == ["entry_a", "entry_b", "entry_c"]
    assert [vote.name for vote in decisions[0].exit.votes] == ["exit_a", "exit_b", "exit_c"]


def test_multi_signal_canonicalizes_child_order_for_stable_hashing() -> None:
    base = StrategySpec(
        name="multi_signal_order_a",
        signal=SignalSpec("multi_signal", _multi_signal_params()),
    )
    reordered_params = {
        "entry_rule": {
            "combiner": "all",
            "signals": [
                {"name": "vwap_distance", "params": {"min_bps": "10", "side": "below"}},
                {"name": "ema_trend_up", "params": {"slow": "3", "fast": "2"}},
                {"name": "rsi_below", "params": {"threshold": "30", "length": "2"}},
            ],
        },
        "exit_rule": {
            "combiner": "any",
            "signals": [
                {"name": "vwap_reclaimed", "params": {"min_bps": "0"}},
                {"name": "ema_trend_down", "params": {"slow": "3", "fast": "2"}},
                {"name": "rsi_above", "params": {"threshold": "70", "length": "2"}},
            ],
        },
    }
    reordered = StrategySpec(
        name="multi_signal_order_b",
        signal=SignalSpec("multi_signal", reordered_params),
    )

    validated_base = REGISTRY.validate_spec(base)
    validated_reordered = REGISTRY.validate_spec(reordered)

    assert validated_base.signal.params == validated_reordered.signal.params
    assert validated_base.spec_hash() == validated_reordered.spec_hash()


def test_multi_signal_required_history_uses_max_entry_and_exit_child_history() -> None:
    params = {
        "entry_rule": {
            "combiner": "all",
            "signals": [
                {"name": "rsi_below", "params": {"length": 4, "threshold": 30.0}},
                {"name": "ema_trend_up", "params": {"fast": 2, "slow": 3}},
                {"name": "vwap_distance", "params": {"side": "below"}},
            ],
        },
        "exit_rule": {
            "combiner": "any",
            "signals": [
                {"name": "intraday_volatility", "params": {"lookback": 2, "percentile_window": 5}},
                {"name": "rsi_above", "params": {"length": 2, "threshold": 70.0}},
                {"name": "vwap_reclaimed", "params": {"min_bps": 0.0}},
            ],
        },
    }

    spec = REGISTRY.validate_spec(
        StrategySpec(
            name="multi_signal_history",
            signal=SignalSpec("multi_signal", params),
        )
    )

    assert REGISTRY.required_history(spec) == 7


def test_registry_generate_decisions_wraps_legacy_regimes() -> None:
    bars = tuple(_bar(index, 100.0 + index) for index in range(6))
    decisions = REGISTRY.generate_decisions(
        StrategySpec(
            name="legacy_ema_decisions",
            signal=SignalSpec("ema_cross", {"fast_length": 2, "slow_length": 3}),
        ),
        bars[:3],
        bars[3:],
    )

    assert len(decisions) == 3
    assert decisions[0].entry.votes[0].name == "legacy_regime"


def test_multi_signal_decisions_do_not_change_prefix_when_future_bars_change() -> None:
    history = tuple(
        _bar(index, 100.0 + ((index % 5) - 2) * 0.35, volume=1_000.0 + index * 10, vwap=100.0)
        for index in range(8)
    )
    test = tuple(
        _bar(8 + index, 100.0 + ((index % 7) - 3) * 0.45, volume=1_200.0 + index * 25, vwap=100.25)
        for index in range(8)
    )
    spec = StrategySpec(
        name="multi_signal_no_lookahead",
        signal=SignalSpec(
            "multi_signal",
            {
                "entry_rule": {
                    "combiner": "k_of_n",
                    "k": 2,
                    "signals": [
                        {"name": "rsi_below", "params": {"length": 3, "threshold": 60.0}},
                        {"name": "vwap_distance", "params": {"side": "below", "min_bps": 1.0}},
                        {"name": "relative_volume", "params": {"lookback": 3, "min_ratio": 0.5}},
                    ],
                },
                "exit_rule": {
                    "combiner": "any",
                    "signals": [
                        {"name": "rsi_above", "params": {"length": 3, "threshold": 40.0}},
                        {"name": "vwap_reclaimed", "params": {"min_bps": 0.0}},
                        {"name": "ema_trend_down", "params": {"fast": 2, "slow": 4}},
                    ],
                },
            },
        ),
    )
    baseline = REGISTRY.generate_decisions(spec, history, test)

    for prefix_end in range(len(test) - 1):
        altered = list(test)
        for future_index in range(prefix_end + 1, len(altered)):
            altered[future_index] = _bar(
                8 + future_index,
                200.0 + future_index,
                volume=10_000.0 + future_index,
                vwap=50.0,
            )

        decisions = REGISTRY.generate_decisions(spec, history, tuple(altered))

        assert decisions[: prefix_end + 1] == baseline[: prefix_end + 1]


def test_evaluation_runner_routes_multi_signal_through_decision_engine(seeded_store) -> None:
    runner = EvaluationRunner(DataView(seeded_store.database_path), REGISTRY)
    spec = StrategySpec(
        name="runner_multi_signal",
        signal=SignalSpec(
            "multi_signal",
            {
                "entry_rule": {
                    "combiner": "any",
                    "signals": [
                        {"name": "rsi_below", "params": {"length": 3, "threshold": 60.0}},
                        {"name": "vwap_distance", "params": {"side": "below", "min_bps": 1.0}},
                        {"name": "relative_volume", "params": {"lookback": 3, "min_ratio": 0.5}},
                    ],
                },
                "exit_rule": {
                    "combiner": "any",
                    "signals": [
                        {"name": "rsi_above", "params": {"length": 3, "threshold": 40.0}},
                        {"name": "vwap_reclaimed", "params": {"min_bps": 0.0}},
                        {"name": "ema_trend_down", "params": {"fast": 2, "slow": 4}},
                    ],
                },
            },
        ),
    )

    result = runner.evaluate_single_window(spec)

    assert result.bars
    assert len(result.equity_curve) == len(result.bars)
