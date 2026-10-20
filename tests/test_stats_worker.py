import time

from gateway.stats.db import StatsDb
from gateway.stats.pricing import Pricing
from gateway.stats.worker import CostWorker


def test_process_pending_fills_costs(tmp_path):
    db = StatsDb(tmp_path / "usage.db")
    db.insert(ts=1000, base_url="https://a", model="m1", input_tokens=1_000_000,
              cache_read_tokens=0, cache_write_tokens=0, output_tokens=0, status=200)
    db.insert(ts=1001, base_url="https://a", model="m2", input_tokens=1,
              cache_read_tokens=0, cache_write_tokens=0, output_tokens=0, status=200)

    worker = CostWorker(db, resolve=lambda name: Pricing(3.0, 0.3, 3.75, 15.0) if name == "m1" else None)
    assert worker.process_pending() == 2
    assert db.pending() == []
    items, _ = db.list_requests(0, 2000, None, None, 10, 0)
    by_model = {i["model"]: i["cost_usd"] for i in items}
    assert by_model["m1"] == 3.0
    assert by_model["m2"] == 0.0   # unknown model → cost 0


def test_process_pending_noop_when_empty(tmp_path):
    db = StatsDb(tmp_path / "usage.db")
    worker = CostWorker(db, resolve=lambda name: None)
    assert worker.process_pending() == 0


def test_worker_runs_and_processes(tmp_path):
    db = StatsDb(tmp_path / "usage.db")
    worker = CostWorker(db, resolve=lambda name: Pricing(1.0, 0.1, 1.25, 5.0), interval=0.02)
    worker.start()
    try:
        db.insert(ts=1000, base_url="https://a", model="m1", input_tokens=1_000_000,
                  cache_read_tokens=0, cache_write_tokens=0, output_tokens=0, status=200)
        deadline = time.monotonic() + 5
        while db.pending() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert db.pending() == []
        items, _ = db.list_requests(0, 2000, None, None, 10, 0)
        assert items[0]["cost_usd"] == 1.0
    finally:
        worker.stop()
