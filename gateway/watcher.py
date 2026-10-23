"""Background watcher: periodically fetch each endpoint's upstream /v1/models
and flag newly appeared ids as ``endpoints[].models_news`` (「模型上新」).
Baseline lists live in a JSON cache file in the system temp dir; failures skip
the endpoint (no baseline overwrite), keeping cache and config consistent."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

import httpx

from gateway.fetch import FetchError, fetch_models
from gateway.store import ConfigStore, StoreError

logger = logging.getLogger("llm-gateway.watcher")

DEFAULT_INTERVAL = 12 * 3600
DEFAULT_CACHE_FILENAME = "llm-gateway-models-cache.json"


def default_cache_path() -> Path:
    """Baseline cache location shared by the watcher and the admin
    fetch-models route (resolved once in build_app so both see one file)."""
    return Path(tempfile.gettempdir()) / DEFAULT_CACHE_FILENAME


def diff(old_ids: list[str] | None, new_ids: list[str]) -> tuple[list[str], list[str]]:
    """Ids newly present in new_ids (added) and no longer present
    (removed), preserving new_ids order for added. ``None`` old = no
    baseline -> nothing is new."""
    if old_ids is None:
        return ([], [])
    old = set(old_ids)
    new = set(new_ids)
    added = [i for i in new_ids if i not in old]
    removed = [i for i in old_ids if i not in new]
    return (added, removed)


def load_cache(path: Path) -> dict[str, dict]:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def save_cache(path: Path, cache: dict[str, dict]) -> None:
    data = json.dumps(cache, ensure_ascii=False, indent=2) + "\n"
    fd, tmp_name = tempfile.mkstemp(dir=str(Path(path).parent), prefix=".models-cache.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_name)
        raise


class ModelWatcher:
    def __init__(self, store: ConfigStore, http_client: httpx.AsyncClient,
                 interval: float = DEFAULT_INTERVAL,
                 cache_path: str | Path | None = None,
                 now_fn=time.time,
                 http_client_for: Callable[[str | None], httpx.AsyncClient] | None = None) -> None:
        self._store = store
        self._http = http_client
        self._http_client_for = http_client_for
        self._interval = interval
        self._cache_path = Path(cache_path) if cache_path is not None \
            else default_cache_path()
        self._now = now_fn
        self._task: asyncio.Task | None = None
        # Serializes scans: a manual trigger (admin scan-models route) must not
        # interleave with the background loop's own scan.
        self._scan_lock = asyncio.Lock()

    def start(self) -> None:
        self._task = asyncio.get_running_loop().create_task(self._run())

    def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            self._task = None

    async def _run(self) -> None:
        while True:
            try:
                await self.scan_once()
            except Exception:
                logger.exception("model watcher scan failed")
            await asyncio.sleep(self._interval)

    async def scan_once(self) -> list[dict]:
        """One pass over every endpoint. Returns per-endpoint results (used by
        the manual trigger route; the background loop ignores them)."""
        results: list[dict] = []
        async with self._scan_lock:
            cache = load_cache(self._cache_path)
            for entry in self._store.endpoint_dicts():
                base_url = entry["base_url"]
                try:
                    client = (
                        self._http_client_for(entry.get("socks5_proxy"))
                        if self._http_client_for is not None else self._http
                    )
                    ids = await fetch_models(base_url, entry["custom_headers"] or {}, client)
                except FetchError as exc:
                    logger.warning("model watcher: skipping %s (%s)", base_url, exc)
                    results.append({"base_url": base_url, "ok": False, "error": str(exc)})
                    continue
                record = cache.get(base_url)
                old_ids = record.get("ids") if isinstance(record, dict) else None
                added, removed = diff(old_ids, ids)
                news = [str(i) for i in entry.get("models_news") or []]
                if old_ids is not None and added:
                    # Accumulate onto unacknowledged news: the badge lists every
                    # model that appeared since the user last opened the modal.
                    merged = news + [i for i in added if i not in news]
                    try:
                        self._store.set_models_news(base_url, merged)
                    except StoreError as exc:
                        logger.warning(
                            "model watcher: cannot write config for %s (%s); cache not updated",
                            base_url, exc)
                        results.append({"base_url": base_url, "ok": False,
                                        "error": f"config write failed: {exc}"})
                        continue
                    news = merged
                cache[base_url] = {"ids": ids, "ts": int(self._now())}
                save_cache(self._cache_path, cache)
                results.append({
                    "base_url": base_url,
                    "ok": True,
                    "models": ids,
                    "added": added,
                    "removed": removed,
                    "models_news": news,
                })
        return results
