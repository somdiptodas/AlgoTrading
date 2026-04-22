from __future__ import annotations

import os
import platform
import sys
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    database_path: Path
    research_dir: Path
    ledger_path: Path
    artifacts_dir: Path
    reports_dir: Path
    default_ticker: str = "SPY"
    default_multiplier: int = 1
    default_timespan: str = "minute"
    default_chunk_days: int = 30
    default_limit: int = 50_000

    def require_massive_api_key(self) -> str:
        api_key = os.environ.get("MASSIVE_API_KEY")
        if not api_key:
            raise RuntimeError("MASSIVE_API_KEY is not set")
        return api_key

    def environment_hash(self) -> str:
        payload = "|".join(
            [
                platform.platform(),
                sys.version,
                str(self.database_path.resolve()),
                str(self.ledger_path.resolve()),
            ]
        )
        return sha256(payload.encode("utf-8")).hexdigest()


def load_settings(database_path: str | None = None) -> Settings:
    repo_root = Path(__file__).resolve().parent.parent
    data_dir = repo_root / "data"
    research_dir = data_dir / "research"
    db_path = Path(database_path).expanduser() if database_path else data_dir / "market_data.db"
    return Settings(
        database_path=db_path,
        research_dir=research_dir,
        ledger_path=research_dir / "ledger.db",
        artifacts_dir=research_dir / "artifacts",
        reports_dir=research_dir / "reports",
    )
