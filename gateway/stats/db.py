"""SQLite storage for usage records. The requests table doubles as the
cost-computation queue: rows are inserted with cost_usd NULL and the worker
fills the price later (see worker.py)."""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts INTEGER NOT NULL,
    base_url TEXT NOT NULL,
    model TEXT NOT NULL,
    input_tokens INTEGER NOT NULL DEFAULT 0,
    cache_read_tokens INTEGER NOT NULL DEFAULT 0,
    cache_write_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    status INTEGER NOT NULL,
    cost_usd REAL
);
CREATE INDEX IF NOT EXISTS idx_requests_ts ON requests(ts);
CREATE INDEX IF NOT EXISTS idx_requests_base_model ON requests(base_url, model);
CREATE INDEX IF NOT EXISTS idx_requests_cost_null ON requests(cost_usd) WHERE cost_usd IS NULL;
"""

# Rows without ttft (non-streaming responses — TTFT is unobservable there,
# the upstream generates everything before sending the body) subtract this
# assumed observed TTFT from the duration when computing TPS; it collapses
# to 0 when the duration doesn't exceed it (division-by-zero guard).
_ASSUMED_TTFT_MS = 3_000


class StatsDb:
    def __init__(self, path: str | Path) -> None:
        self._lock = threading.Lock()
        # check_same_thread=False: the cost worker uses this connection from its
        # own thread; all access is serialized by self._lock instead.
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(_SCHEMA)
        self._migrate()
        self._conn.commit()

    def _migrate(self) -> None:
        with self._lock:
            cols = {r[1] for r in self._conn.execute("PRAGMA table_info(requests)")}
            if "duration_ms" not in cols:
                self._conn.execute(
                    "ALTER TABLE requests ADD COLUMN duration_ms INTEGER")
            if "ttft_ms" not in cols:
                self._conn.execute(
                    "ALTER TABLE requests ADD COLUMN ttft_ms INTEGER")

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def insert(self, *, ts: int, base_url: str, model: str, input_tokens: int,
               cache_read_tokens: int, cache_write_tokens: int,
               output_tokens: int, status: int,
               duration_ms: int | None = None,
               ttft_ms: int | None = None) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO requests (ts, base_url, model, input_tokens, cache_read_tokens,"
                " cache_write_tokens, output_tokens, status, cost_usd, duration_ms, ttft_ms)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?)",
                (ts, base_url, model, input_tokens, cache_read_tokens,
                 cache_write_tokens, output_tokens, status, duration_ms, ttft_ms),
            )
            self._conn.commit()

    def pending(self, limit: int = 50) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, model, input_tokens, cache_read_tokens, cache_write_tokens,"
                " output_tokens FROM requests WHERE cost_usd IS NULL ORDER BY id LIMIT ?",
                (limit,),
            ).fetchall()
        return [
            {"id": r[0], "model": r[1], "input_tokens": r[2],
             "cache_read_tokens": r[3], "cache_write_tokens": r[4], "output_tokens": r[5]}
            for r in rows
        ]

    def set_cost(self, request_id: int, cost: float) -> None:
        with self._lock:
            self._conn.execute("UPDATE requests SET cost_usd = ? WHERE id = ?",
                               (cost, request_id))
            self._conn.commit()

    def day_minutes(self, start_ts: int, end_ts: int) -> list[dict]:
        """Per-minute buckets in [start_ts, end_ts): [{ts, model, input,
        cache_read, cache_write, output, cost}]."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT (ts / 60) * 60 AS bucket, model,"
                " SUM(input_tokens), SUM(cache_read_tokens), SUM(cache_write_tokens),"
                " SUM(output_tokens), SUM(cost_usd)"
                " FROM requests WHERE ts >= ? AND ts < ?"
                " GROUP BY bucket, model ORDER BY bucket",
                (start_ts, end_ts),
            ).fetchall()
        return [
            {"ts": r[0], "model": r[1], "input": r[2], "cache_read": r[3],
             "cache_write": r[4], "output": r[5], "cost": r[6] or 0.0}
            for r in rows
        ]

    def model_totals(self, start_ts: int, end_ts: int) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT model, SUM(input_tokens + cache_read_tokens + cache_write_tokens"
                " + output_tokens) AS tokens, SUM(cost_usd), COUNT(*)"
                " FROM requests WHERE ts >= ? AND ts < ? GROUP BY model"
                " ORDER BY tokens DESC",
                (start_ts, end_ts),
            ).fetchall()
        return [{"model": r[0], "tokens": r[1], "cost": r[2] or 0.0, "calls": r[3]}
                for r in rows]

    def token_totals(self, start_ts: int, end_ts: int, base_url: str | None,
                     models: list[str] | None = None,
                     model: str | None = None) -> dict:
        """Aggregate token sums by kind (optional endpoint / model filters),
        plus the overall cache hit rate over the prompt side."""
        clause = ""
        args: list = [start_ts, end_ts]
        if base_url:
            clause += " AND base_url = ?"
            args.append(base_url)
        if models:
            clause += f" AND model IN ({','.join('?' * len(models))})"
            args.extend(models)
        if model:
            clause += " AND model = ?"
            args.append(model)
        with self._lock:
            r = self._conn.execute(
                "SELECT SUM(input_tokens), SUM(cache_read_tokens),"
                " SUM(cache_write_tokens), SUM(output_tokens)"
                f" FROM requests WHERE ts >= ? AND ts < ?{clause}",
                args,
            ).fetchone()
        total_input = r[0] or 0
        cache_read = r[1] or 0
        cache_write = r[2] or 0
        output = r[3] or 0
        prompt_side = total_input + cache_read + cache_write
        return {
            "input": total_input,
            "cache_read": cache_read,
            "cache_write": cache_write,
            "output": output,
            "cache_hit_rate": round(cache_read / prompt_side, 4) if prompt_side else 0.0,
        }

    def endpoint_models(self, start_ts: int, end_ts: int, base_url: str,
                        models: list[str] | None = None) -> list[dict]:
        clause = " AND base_url = ?"
        args: list = [start_ts, end_ts, base_url]
        if models:
            clause += f" AND model IN ({','.join('?' * len(models))})"
            args.extend(models)
        with self._lock:
            rows = self._conn.execute(
                "SELECT model,"
                " SUM(input_tokens + cache_read_tokens + cache_write_tokens + output_tokens) AS tokens,"
                " SUM(cost_usd), COUNT(*),"
                " SUM(cache_read_tokens) AS cr,"
                " SUM(input_tokens + cache_read_tokens + cache_write_tokens) AS prompt_side"
                f" FROM requests WHERE ts >= ? AND ts < ?{clause}"
                " GROUP BY model ORDER BY tokens DESC",
                args,
            ).fetchall()
        result = []
        for r in rows:
            prompt_side = r[5] or 0
            result.append({
                "model": r[0], "tokens": r[1], "cost": r[2] or 0.0, "calls": r[3],
                "cache_hit_rate": round(r[4] / prompt_side, 4) if prompt_side else 0.0,
            })
        return result

    def timeseries(self, start_ts: int, end_ts: int, base_url: str | None = None,
                   models: list[str] | None = None,
                   granularity: str = "minute") -> list[dict]:
        """Per-bucket per-model time series, optionally filtered by base_url
        and model names.  Granularity: minute (60s), hour (3600s), day (86400s)."""
        bucket_seconds = {"minute": 60, "hour": 3600, "day": 86400}[granularity]
        where = "ts >= ? AND ts < ?"
        args: list = [start_ts, end_ts]
        if base_url:
            where += " AND base_url = ?"
            args.append(base_url)
        if models:
            where += f" AND model IN ({','.join('?' * len(models))})"
            args.extend(models)
        with self._lock:
            rows = self._conn.execute(
                f"SELECT (ts / {bucket_seconds}) * {bucket_seconds} AS bucket, model,"
                f" SUM(input_tokens) AS input,"
                f" SUM(cache_read_tokens) AS cache_read,"
                f" SUM(cache_write_tokens) AS cache_write,"
                f" SUM(output_tokens) AS output,"
                f" SUM(cost_usd) AS cost"
                f" FROM requests WHERE {where}"
                f" GROUP BY bucket, model ORDER BY bucket, model",
                args,
            ).fetchall()
        return [
            {"ts": r[0], "model": r[1], "input": r[2], "cache_read": r[3],
             "cache_write": r[4], "output": r[5], "cost": r[6] or 0.0}
            for r in rows
        ]

    def list_requests(self, start_ts: int, end_ts: int, base_url: str | None,
                      model: str | None, limit: int, offset: int) -> tuple[list[dict], int]:
        where = "ts >= ? AND ts < ?"
        args: list = [start_ts, end_ts]
        if base_url:
            where += " AND base_url = ?"
            args.append(base_url)
        if model:
            where += " AND model = ?"
            args.append(model)
        with self._lock:
            total = self._conn.execute(
                f"SELECT COUNT(*) FROM requests WHERE {where}", args
            ).fetchone()[0]
            rows = self._conn.execute(
                f"SELECT ts, base_url, model, input_tokens, cache_read_tokens,"
                f" cache_write_tokens, output_tokens, status, cost_usd"
                f" FROM requests WHERE {where} ORDER BY ts DESC, id DESC LIMIT ? OFFSET ?",
                args + [limit, offset],
            ).fetchall()
        return [
            {"ts": r[0], "base_url": r[1], "model": r[2], "input_tokens": r[3],
             "cache_read_tokens": r[4], "cache_write_tokens": r[5],
             "output_tokens": r[6], "status": r[7], "cost_usd": r[8]}
            for r in rows
        ], total

    def model_tps(self, start_ts: int, end_ts: int, samples: int = 5) -> list[dict]:
        """Average decode tokens/second over each model's last `samples` requests
        (within [start_ts, end_ts)). Decode TPS = output / (duration - ttft) —
        excluding time-to-first-token so short responses aren't skewed by TTFT.
        Rows without ttft (non-streaming — TTFT unobservable) subtract an
        assumed 3s observed TTFT instead; the offset collapses to 0 when the
        duration doesn't exceed it.

        Models with fewer than `samples` usable requests (status 200, output>0,
        duration recorded) are omitted. avg_ttft_ms is None when no sampled
        row has ttft (and now includes queueing time — every TTFT is counted)."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT model, output_tokens, duration_ms, ttft_ms FROM ("
                "  SELECT model, output_tokens, duration_ms, ttft_ms,"
                "         ROW_NUMBER() OVER (PARTITION BY model ORDER BY id DESC) AS rn"
                "  FROM requests"
                "  WHERE ts >= ? AND ts < ? AND status = 200 AND output_tokens > 0"
                "    AND duration_ms IS NOT NULL AND duration_ms > 0"
                "    AND duration_ms >= output_tokens"
                ") WHERE rn <= ?",
                (start_ts, end_ts, samples),
            ).fetchall()
        by_model: dict[str, list[tuple[float, int | None]]] = {}
        for model, out_tokens, duration_ms, ttft_ms in rows:
            if ttft_ms is not None and 0 < ttft_ms < duration_ms:
                decode_s = (duration_ms - ttft_ms) / 1000.0
            else:
                # No usable ttft (non-streaming): subtract the assumed observed
                # TTFT when the duration exceeds it, else divide by the whole
                # duration.
                offset = _ASSUMED_TTFT_MS if duration_ms > _ASSUMED_TTFT_MS else 0
                decode_s = (duration_ms - offset) / 1000.0
            by_model.setdefault(model, []).append((out_tokens / decode_s, ttft_ms))
        result = []
        for model, entries in by_model.items():
            if len(entries) < samples:
                continue
            recent = entries[:samples]
            ttfts = [t for _, t in recent if t is not None]
            result.append({
                "model": model,
                "tps": round(sum(tps for tps, _ in recent) / len(recent), 1),
                "samples": len(recent),
                "avg_ttft_ms": (round(sum(ttfts) / len(ttfts), 1)
                                if ttfts else None),
            })
        result.sort(key=lambda r: r["tps"], reverse=True)
        return result
