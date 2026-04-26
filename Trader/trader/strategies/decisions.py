from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SignalVote:
    name: str
    passed: bool
    detail: str = ""
