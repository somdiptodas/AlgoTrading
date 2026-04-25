from __future__ import annotations

from trader.cli.loop_cmd import DEFAULT_OVERPLAN_FACTOR, MIN_PLANNED_SPECS, _planned_spec_count


def test_planned_spec_count_uses_named_overplan_factor_and_floor() -> None:
    assert _planned_spec_count(batch_size=6, overplan_factor=DEFAULT_OVERPLAN_FACTOR) == 72
    assert _planned_spec_count(batch_size=1, overplan_factor=DEFAULT_OVERPLAN_FACTOR) == MIN_PLANNED_SPECS
    assert _planned_spec_count(batch_size=30, overplan_factor=3) == 90
