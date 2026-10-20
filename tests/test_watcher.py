import asyncio
import json

import httpx

from gateway.config import load_config
from gateway.store import ConfigStore
from gateway.watcher import ModelWatcher, diff, load_cache, save_cache


def _setup(tmp_path, models_by_url):
    p = tmp_path / "config.json"
    p.write_text(json.dumps({
        "server": {"api_key": "k"},
        "endpoints": [{
            "base_url": "https://a.com",
            "custom_headers": {"User-Agent": "ua"},
            "models": [{"name": "m1"}],
        }, {
            "base_url": "https://b.com",
            "models": [{"name": "n1"}],
        }],
    }))
    config = load_config(p)
    store = ConfigStore(config, p)

    def handler(request: httpx.Request) -> httpx.Response:
        for base, ids in models_by_url.items():
            if request.url.host == base.split("//")[1]:
                return httpx.Response(200, json={"data": [{"id": i} for i in ids]})
        return httpx.Response(404)

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    cache_path = tmp_path / "cache.json"
    return store, http_client, cache_path


def _news(store, base_url):
    return next(e for e in store.config.endpoints
                if e.base_url == base_url).models_news


# —— diff 纯函数 ——

def test_diff_add_remove_overlap():
    added, removed = diff(["a", "b"], ["b", "c"])
    assert added == ["c"]
    assert removed == ["a"]


def test_diff_empty_cases():
    assert diff([], []) == ([], [])
    assert diff(None, ["x"]) == ([], [])   # no baseline -> no news
    assert diff(["x"], []) == ([], ["x"])


# —— cache 文件 ——

def test_cache_roundtrip(tmp_path):
    path = tmp_path / "cache.json"
    save_cache(path, {"https://a.com": {"ids": ["x"], "ts": 123}})
    assert load_cache(path) == {"https://a.com": {"ids": ["x"], "ts": 123}}


def test_load_cache_missing_file_returns_empty(tmp_path):
    assert load_cache(tmp_path / "nope.json") == {}


def test_load_cache_garbage_returns_empty(tmp_path):
    path = tmp_path / "cache.json"
    path.write_text("not json at all")
    assert load_cache(path) == {}


# —— scan ——

def test_first_scan_creates_baseline_without_news(tmp_path):
    store, http_client, cache_path = _setup(tmp_path, {
        "https://a.com": ["m1", "m2"],
        "https://b.com": ["n1"],
    })
    w = ModelWatcher(store, http_client, interval=3600, cache_path=cache_path)
    asyncio.run(w.scan_once())
    cache = load_cache(cache_path)
    assert cache["https://a.com"]["ids"] == ["m1", "m2"]
    assert cache["https://b.com"]["ids"] == ["n1"]
    for ep in store.config.endpoints:
        assert ep.models_news == []


def test_new_models_set_news(tmp_path):
    store, http_client, cache_path = _setup(tmp_path, {
        "https://a.com": ["m1", "m2"],
        "https://b.com": ["n1"],
    })
    w = ModelWatcher(store, http_client, interval=3600, cache_path=cache_path)
    asyncio.run(w.scan_once())

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "a.com":
            return httpx.Response(200, json={"data": [{"id": "m1"}, {"id": "m2"}, {"id": "m3"}]})
        return httpx.Response(200, json={"data": [{"id": "n1"}]})

    w2 = ModelWatcher(store, httpx.AsyncClient(transport=httpx.MockTransport(handler)),
                      interval=3600, cache_path=cache_path)
    asyncio.run(w2.scan_once())
    assert _news(store, "https://a.com") == ["m3"]
    assert _news(store, "https://b.com") == []


def test_news_accumulates_across_scans(tmp_path):
    """Second detection round adds to, not replaces, the flag list."""
    store, http_client, cache_path = _setup(tmp_path, {"https://a.com": ["m1"]})
    w = ModelWatcher(store, http_client, interval=3600, cache_path=cache_path)
    asyncio.run(w.scan_once())

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"id": "m1"}, {"id": "m2"}]})

    w2 = ModelWatcher(store, httpx.AsyncClient(transport=httpx.MockTransport(handler)),
                      interval=3600, cache_path=cache_path)
    asyncio.run(w2.scan_once())

    def handler2(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"id": "m1"}, {"id": "m2"}, {"id": "m3"}]})

    w3 = ModelWatcher(store, httpx.AsyncClient(transport=httpx.MockTransport(handler2)),
                      interval=3600, cache_path=cache_path)
    asyncio.run(w3.scan_once())
    assert _news(store, "https://a.com") == ["m2", "m3"]


def test_removed_models_only_update_cache(tmp_path):
    store, http_client, cache_path = _setup(tmp_path, {
        "https://a.com": ["m1", "m2"],
        "https://b.com": ["n1"],
    })
    w = ModelWatcher(store, http_client, interval=3600, cache_path=cache_path)
    asyncio.run(w.scan_once())

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "a.com":
            return httpx.Response(200, json={"data": [{"id": "m1"}]})   # m2 gone
        return httpx.Response(200, json={"data": [{"id": "n1"}]})

    w2 = ModelWatcher(store, httpx.AsyncClient(transport=httpx.MockTransport(handler)),
                      interval=3600, cache_path=cache_path)
    asyncio.run(w2.scan_once())
    assert _news(store, "https://a.com") == []
    assert load_cache(cache_path)["https://a.com"]["ids"] == ["m1"]


def test_failed_endpoint_skipped_entirely(tmp_path):
    store, http_client, cache_path = _setup(tmp_path, {
        "https://a.com": ["m1"],
        "https://b.com": ["n1"],
    })
    w = ModelWatcher(store, http_client, interval=3600, cache_path=cache_path)
    asyncio.run(w.scan_once())

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "a.com":
            raise httpx.ConnectError("down")
        return httpx.Response(200, json={"data": [{"id": "n1"}, {"id": "n2"}]})

    w2 = ModelWatcher(store, httpx.AsyncClient(transport=httpx.MockTransport(handler)),
                      interval=3600, cache_path=cache_path)
    asyncio.run(w2.scan_once())
    # a.com unchanged in cache (old baseline kept), b.com updated with news
    cache = load_cache(cache_path)
    assert cache["https://a.com"]["ids"] == ["m1"]
    assert cache["https://b.com"]["ids"] == ["n1", "n2"]
    assert _news(store, "https://b.com") == ["n2"]
    assert _news(store, "https://a.com") == []


def test_store_write_failure_skips_cache_update(tmp_path):
    """A store that can't write config (read-only mode) must not have its
    cache entry updated either — cache and config stay consistent, so the
    next scan re-detects the same new models."""
    store, http_client, cache_path = _setup(tmp_path, {
        "https://a.com": ["m1"],
        "https://b.com": ["n1"],
    })
    # baseline via the writable store
    asyncio.run(ModelWatcher(store, http_client, interval=3600,
                             cache_path=cache_path).scan_once())
    # rebuild the store without a config path: set_models_news raises StoreError
    readonly = ConfigStore(load_config(tmp_path / "config.json"), None)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "a.com":
            return httpx.Response(200, json={"data": [{"id": "m1"}, {"id": "mX"}]})
        return httpx.Response(200, json={"data": [{"id": "n1"}]})

    w = ModelWatcher(readonly, httpx.AsyncClient(transport=httpx.MockTransport(handler)),
                     interval=3600, cache_path=cache_path)
    asyncio.run(w.scan_once())
    # cache NOT updated for a.com (stays consistent with config); b.com still
    # processed independently (baseline written, no news)
    cache = load_cache(cache_path)
    assert cache["https://a.com"]["ids"] == ["m1"]
    assert cache["https://b.com"]["ids"] == ["n1"]


def test_start_stop_lifecycle(tmp_path):
    store, http_client, cache_path = _setup(tmp_path, {
        "https://a.com": ["m1"],
        "https://b.com": ["n1"],
    })
    w = ModelWatcher(store, http_client, interval=3600, cache_path=cache_path)

    async def run():
        w.start()
        await asyncio.sleep(0.05)   # let the first scan run
        w.stop()
        await asyncio.sleep(0)

    asyncio.run(run())
    assert load_cache(cache_path) != {}


def test_scan_once_returns_results(tmp_path):
    store, http_client, cache_path = _setup(tmp_path, {
        "https://a.com": ["m1"],
        "https://b.com": ["n1"],
    })
    w = ModelWatcher(store, http_client, interval=3600, cache_path=cache_path)
    results = asyncio.run(w.scan_once())
    assert results == [
        {"base_url": "https://a.com", "ok": True, "models": ["m1"], "added": [],
         "removed": [], "models_news": []},
        {"base_url": "https://b.com", "ok": True, "models": ["n1"], "added": [],
         "removed": [], "models_news": []},
    ]


def test_scan_once_result_reports_added_and_news(tmp_path):
    store, http_client, cache_path = _setup(tmp_path, {
        "https://a.com": ["m1"],
        "https://b.com": ["n1"],
    })
    w = ModelWatcher(store, http_client, interval=3600, cache_path=cache_path)
    asyncio.run(w.scan_once())

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "a.com":
            return httpx.Response(200, json={"data": [{"id": "m1"}, {"id": "m2"}]})
        return httpx.Response(200, json={"data": [{"id": "n1"}]})

    w2 = ModelWatcher(store, httpx.AsyncClient(transport=httpx.MockTransport(handler)),
                      interval=3600, cache_path=cache_path)
    results = asyncio.run(w2.scan_once())
    assert results == [
        {"base_url": "https://a.com", "ok": True, "models": ["m1", "m2"],
         "added": ["m2"], "removed": [], "models_news": ["m2"]},
        {"base_url": "https://b.com", "ok": True, "models": ["n1"],
         "added": [], "removed": [], "models_news": []},
    ]


def test_scan_once_reports_failed_endpoint(tmp_path):
    store, http_client, cache_path = _setup(tmp_path, {
        "https://a.com": ["m1"],
        "https://b.com": ["n1"],
    })

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "a.com":
            raise httpx.ConnectError("down")
        return httpx.Response(200, json={"data": [{"id": "n1"}]})

    w = ModelWatcher(store, httpx.AsyncClient(transport=httpx.MockTransport(handler)),
                     interval=3600, cache_path=cache_path)
    results = asyncio.run(w.scan_once())
    assert results[0]["ok"] is False
    assert "error" in results[0]
    assert results[1]["ok"] is True
