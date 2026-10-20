"""Background cost worker: fills cost_usd on pending usage records."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable

from gateway.stats.db import StatsDb
from gateway.stats.pricing import Pricing

logger = logging.getLogger("llm-gateway.stats")

PricingResolver = Callable[[str], Pricing | None]


class CostWorker:
    def __init__(self, db: StatsDb, resolve: PricingResolver,
                 interval: float = 1.0, batch: int = 50) -> None:
        self._db = db
        self._resolve = resolve
        self._interval = interval
        self._batch = batch
        self._stop = threading.Event()

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="stats-cost-worker", daemon=True)
        self._thread.start()

    def stop(self, timeout: float | None = None) -> None:
        self._stop.set()
        if timeout is not None and self._thread is not None:
            self._thread.join(timeout)

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            try:
                self.process_pending()
            except Exception:
                logger.exception("stats cost worker iteration failed")

    def process_pending(self) -> int:
        """Fill costs for all rows with cost_usd IS NULL. Returns rows processed."""
        processed = 0
        while True:
            rows = self._db.pending(self._batch)
            if not rows:
                return processed
            for row in rows:
                pricing = self._resolve(row["model"])
                cost = pricing.cost_usd(
                    row["input_tokens"], row["cache_read_tokens"],
                    row["cache_write_tokens"], row["output_tokens"],
                ) if pricing is not None else 0.0
                self._db.set_cost(row["id"], cost)
            processed += len(rows)
