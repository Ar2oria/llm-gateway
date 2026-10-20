"""Read-only usage stats API: /admin/api/stats/day, /admin/api/stats/endpoints,
/admin/api/requests. All date params are interpreted in server-local time."""

from __future__ import annotations

import datetime
import json

from fastapi import FastAPI, Query, Response

from gateway.stats.db import StatsDb

STATS_DISABLED_MESSAGE = "usage stats are disabled (no stats database path)"

DATE_FMT = "%Y-%m-%d"
DATETIME_FMT = "%Y-%m-%dT%H:%M"


def _json_response(data: object, status: int = 200) -> Response:
    return Response(
        content=json.dumps(data),
        status_code=status,
        media_type="application/json",
        headers={"Cache-Control": "no-store"},
    )


def _error(status: int, message: str) -> Response:
    return _json_response({"error": {"message": message}}, status=status)


def _parse_datetime(value: str) -> datetime.datetime:
    """Parse 'YYYY-MM-DD' or 'YYYY-MM-DDTHH:MM' (naive local time)."""
    text = value.strip()
    for fmt in (DATETIME_FMT, DATE_FMT):
        try:
            return datetime.datetime.strptime(text, fmt)
        except ValueError:
            continue
    raise ValueError("expected YYYY-MM-DD or YYYY-MM-DDTHH:MM")


def _day_range(date_str: str) -> tuple[int, int]:
    day = _parse_datetime(date_str)
    start = day.replace(hour=0, minute=0, second=0, microsecond=0)
    return int(start.timestamp()), int(start.timestamp()) + 86400


def _range(from_str: str, to_str: str) -> tuple[int, int]:
    start = _parse_datetime(from_str)
    end = _parse_datetime(to_str)
    if "T" in to_str:
        end += datetime.timedelta(minutes=1)   # THH:MM is inclusive of that minute
    else:
        end += datetime.timedelta(days=1)      # date-only: whole day
    return int(start.timestamp()), int(end.timestamp())


def register_stats_routes(app: FastAPI, db: StatsDb | None) -> None:
    @app.api_route("/admin/api/stats/day", methods=["GET"])
    async def stats_day(date: str | None = None) -> Response:
        if db is None:
            return _error(503, STATS_DISABLED_MESSAGE)
        if not date:
            return _error(400, "date query parameter is required (YYYY-MM-DD)")
        try:
            start, end = _day_range(date)
        except ValueError as exc:
            return _error(400, str(exc))
        minutes = db.day_minutes(start, end)
        totals = db.model_totals(start, end)
        buckets: dict[int, dict] = {}
        for m in minutes:
            bucket = buckets.setdefault(m["ts"], {"ts": m["ts"], "tokens": {}, "costs": {}})
            bucket["tokens"][m["model"]] = {
                "input": m["input"], "cache_read": m["cache_read"],
                "cache_write": m["cache_write"], "output": m["output"],
            }
            bucket["costs"][m["model"]] = m["cost"]
        return _json_response({
            "date": date,
            "models": [t["model"] for t in totals],
            "minutes": sorted(buckets.values(), key=lambda b: b["ts"]),
        })

    @app.api_route("/admin/api/stats/model-tps", methods=["GET"])
    async def model_tps(date: str | None = None) -> Response:
        if db is None:
            return _error(503, STATS_DISABLED_MESSAGE)
        if not date:
            return _error(400, "date query parameter is required (YYYY-MM-DD)")
        try:
            start, end = _day_range(date)
        except ValueError as exc:
            return _error(400, str(exc))
        return _json_response(db.model_tps(start, end))

    @app.api_route("/admin/api/stats/endpoints", methods=["GET"])
    async def stats_endpoints(date: str | None = None,
                              from_: str | None = Query(None, alias="from"),
                              to: str | None = Query(None, alias="to"),
                              base_url: str | None = None,
                              models: str | None = None,
                              granularity: str = "minute") -> Response:
        if db is None:
            return _error(503, STATS_DISABLED_MESSAGE)
        if not base_url:
            return _error(400, "base_url query parameter is required")
        if from_ and to:
            # Time-range mode: return per-bucket time-series data.
            try:
                start, end = _range(from_, to)
            except ValueError as exc:
                return _error(400, str(exc))
            if granularity not in ("minute", "hour", "day"):
                return _error(400, "granularity must be minute, hour, or day")
            model_list = [m for m in models.split(",") if m] if models else None
            minutes = db.timeseries(start, end, base_url, model_list, granularity)
            totals = db.endpoint_models(start, end, base_url, model_list)
            token_totals = db.token_totals(start, end, base_url, model_list)
            return _json_response({
                "base_url": base_url,
                "from": from_, "to": to,
                "models": [r["model"] for r in totals],
                "total_tokens": sum(r["tokens"] for r in totals),
                "total_cost": round(sum(r["cost"] for r in totals), 6),
                "total_input": token_totals["input"],
                "total_cache_read": token_totals["cache_read"],
                "total_cache_write": token_totals["cache_write"],
                "total_output": token_totals["output"],
                "cache_hit_rate": token_totals["cache_hit_rate"],
                "per_model": totals,
                "minutes": minutes,
                "granularity": granularity,
            })
        # Backward-compatible single-day mode.
        if not date:
            return _error(400, "date or from/to query parameters are required")
        try:
            start, end = _day_range(date)
        except ValueError as exc:
            return _error(400, str(exc))
        model_list = [m for m in models.split(",") if m] if models else None
        rows = db.endpoint_models(start, end, base_url, model_list)
        return _json_response({
            "base_url": base_url,
            "date": date,
            "models": [r["model"] for r in rows],
            "total_tokens": sum(r["tokens"] for r in rows),
            "total_cost": round(sum(r["cost"] for r in rows), 6),
            "per_model": rows,
        })

    @app.api_route("/admin/api/requests", methods=["GET"])
    async def stats_requests(from_: str | None = Query(None, alias="from"),
                             to: str | None = Query(None, alias="to"),
                             base_url: str | None = None,
                             model: str | None = None,
                             limit: int = 50, offset: int = 0) -> Response:
        if db is None:
            return _error(503, STATS_DISABLED_MESSAGE)
        if not from_ or not to:
            return _error(400, "from and to query parameters are required")
        try:
            start, end = _range(from_, to)
        except ValueError as exc:
            return _error(400, str(exc))
        if end <= start:
            return _error(400, "to must be after from")
        if limit < 1 or limit > 200:
            return _error(400, "limit must be between 1 and 200")
        if offset < 0:
            return _error(400, "offset must be non-negative")
        items, total = db.list_requests(start, end, base_url, model, limit, offset)
        totals = db.token_totals(start, end, base_url, model=model)
        return _json_response({"total": total, "items": items, "totals": totals})
