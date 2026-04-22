from __future__ import annotations

import time
from datetime import date

from massive import RESTClient
from urllib3.exceptions import MaxRetryError

from trader.data.models import AggregateBar


class MassiveAggregatesClient:
    def __init__(self, api_key: str, max_attempts: int = 5, backoff_seconds: int = 15) -> None:
        self._client = RESTClient(api_key=api_key)
        self._max_attempts = max_attempts
        self._backoff_seconds = backoff_seconds

    def list_aggregate_bars(
        self,
        ticker: str,
        multiplier: int,
        timespan: str,
        start_date: date,
        end_date: date,
        adjusted: bool = True,
        sort: str = "asc",
        limit: int = 50_000,
    ) -> list[AggregateBar]:
        attempt = 1
        while True:
            try:
                results: list[AggregateBar] = []
                for item in self._client.list_aggs(
                    ticker=ticker,
                    multiplier=multiplier,
                    timespan=timespan,
                    from_=start_date.isoformat(),
                    to=end_date.isoformat(),
                    adjusted=adjusted,
                    sort=sort,
                    limit=limit,
                ):
                    results.append(AggregateBar.from_sdk(ticker, multiplier, timespan, item))
                return results
            except MaxRetryError as exc:
                if attempt >= self._max_attempts:
                    raise RuntimeError(
                        f"Massive aggregate request failed after {self._max_attempts} attempts for "
                        f"{ticker} {start_date.isoformat()} -> {end_date.isoformat()}"
                    ) from exc
                delay = self._backoff_seconds * attempt
                print(
                    f"Rate limited for {ticker} {start_date.isoformat()} -> {end_date.isoformat()}; "
                    f"retrying in {delay}s (attempt {attempt}/{self._max_attempts})"
                )
                time.sleep(delay)
                attempt += 1
