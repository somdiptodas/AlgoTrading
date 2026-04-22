from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_MASSIVE_API_KEY = "eR6AQ_GNXAI0E4xf_XjQlRLZ3zhEOxDI"


@dataclass(frozen=True)
class Settings:
    api_key: str
    database_path: Path
    default_ticker: str = "SPY"
    default_multiplier: int = 1
    default_timespan: str = "minute"
    default_chunk_days: int = 30
    default_limit: int = 50_000


def load_settings(database_path: str | None = None) -> Settings:
    api_key = os.environ.get("MASSIVE_API_KEY", DEFAULT_MASSIVE_API_KEY)
    if not api_key:
        raise RuntimeError("MASSIVE_API_KEY is not set")

    base_dir = Path(__file__).resolve().parent
    db_path = Path(database_path) if database_path else base_dir / "data" / "market_data.db"

    return Settings(
        api_key=api_key,
        database_path=db_path,
    )
