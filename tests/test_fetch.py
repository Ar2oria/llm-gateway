import asyncio

import httpx
import pytest

from gateway.fetch import FetchError, fetch_models


def _client(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def test_returns_deduped_ids():
    client = _client(lambda r: httpx.Response(200, json={
        "data": [{"id": "a"}, {"id": "b"}, {"id": "a"}, {"object": "model"}],
    }))
    assert asyncio.run(fetch_models("https://u.example", {"User-Agent": "ua"}, client)) == ["a", "b"]


def test_bare_list_payload():
    client = _client(lambda r: httpx.Response(200, json=[{"id": "x"}]))
    assert asyncio.run(fetch_models("https://u.example", {}, client)) == ["x"]


def test_custom_headers_sent_lowercase():
    captured = {}

    def handler(request):
        captured["url"] = str(request.url)
        captured["ua"] = request.headers.get("user-agent")
        return httpx.Response(200, json={"data": []})

    client = _client(handler)
    assert asyncio.run(fetch_models("https://u.example", {"User-Agent": "ua"}, client)) == []
    assert captured["url"] == "https://u.example/v1/models"
    assert captured["ua"] == "ua"


def test_http_error_raises_fetch_error():
    client = _client(lambda r: httpx.Response(500))
    with pytest.raises(FetchError):
        asyncio.run(fetch_models("https://u.example", {}, client))


def test_transport_error_raises_fetch_error():
    def handler(request):
        raise httpx.ConnectError("boom")

    client = _client(handler)
    with pytest.raises(FetchError):
        asyncio.run(fetch_models("https://u.example", {}, client))


def test_non_json_raises_fetch_error():
    client = _client(lambda r: httpx.Response(200, text="not json"))
    with pytest.raises(FetchError):
        asyncio.run(fetch_models("https://u.example", {}, client))
