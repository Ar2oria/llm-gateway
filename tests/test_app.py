import asyncio
import gzip
import json
import logging
import time

import httpx
from fastapi.testclient import TestClient

import gateway.app as gateway_app
from gateway.app import build_app
from gateway.config import EndpointConfig, GatewayConfig, config_to_dict


def anthropic_system_block(text):
    """The shape the Anthropic system slot takes after an injected prompt."""
    return [{"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}]


def make_config() -> GatewayConfig:
    return GatewayConfig(
        host="127.0.0.1",
        port=8080,
        api_key="local-secret",
        endpoints=[
            EndpointConfig(
                model="gpt-4o",
                base_url="https://upstream.example",
                custom_headers={"Authorization": "Bearer real-key"},
                pricing={"input": 3.0},
            ),
            EndpointConfig(model="claude-3-5-sonnet", base_url="https://anthropic.example"),
            EndpointConfig(
                model="rewrite-model",
                base_url="https://upstream.example",
                upstream_model="real-upstream-model",
            ),
            EndpointConfig(
                model="system-model",
                base_url="https://upstream.example",
                system_prompt="configured system prompt",
            ),
            EndpointConfig(
                model="combined",
                base_url="https://upstream.example",
                upstream_model="real-combined",
                system_prompt="combined sys",
            ),
        ],
    )


def make_debug_config() -> GatewayConfig:
    """Single-model config with debug logging on (debug-log tests)."""
    return GatewayConfig(
        host="127.0.0.1",
        port=8080,
        api_key="local-secret",
        debug_log=True,
        endpoints=[
            EndpointConfig(
                model="m",
                base_url="https://upstream.example",
                custom_headers={"Authorization": "Bearer real-key"},
            ),
        ],
    )


def make_client(handler) -> TestClient:
    config = make_config()
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    app = build_app(config, client=http_client)
    return TestClient(app)


def make_client_with_prompts(handler, prompts_dir, tmp_path, config=None) -> TestClient:
    """Proxy client with a writable config and an explicit template dir.

    The admin inject route needs ``config_path`` (writes 503 in read-only
    mode), so this helper materialises the config in tmp_path.
    """
    config = config if config is not None else make_config()
    p = tmp_path / "config.json"
    p.write_text(json.dumps(config_to_dict(config)), encoding="utf-8")
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    app = build_app(config, client=http_client, config_path=p,
                    watcher_cache_path=tmp_path / "watcher-cache.json",
                    prompts_dir=prompts_dir)
    return TestClient(app)


def test_no_client_auth_sends_no_auth_upstream():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["headers"] = request.headers
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            stream=httpx.ByteStream(b"{}"),
        )

    client = make_client(handler)
    resp = client.post("/v1/chat/completions", json={"model": "gpt-4o", "messages": []})
    assert resp.status_code == 200
    # Auth is client-driven: without any client auth header, nothing is sent
    # upstream (the endpoint's configured auth is cleared, not a fallback).
    assert captured["headers"].get("authorization", "") == ""
    assert captured["headers"].get("x-api-key", "") == ""


def test_proxy_request_uses_endpoint_socks5_client(monkeypatch):
    proxy = "socks5://user:pass@127.0.0.1:1080"
    default_calls = []
    proxy_calls = []

    def default_handler(request: httpx.Request) -> httpx.Response:
        default_calls.append(request)
        return httpx.Response(200, json={})

    def proxy_handler(request: httpx.Request) -> httpx.Response:
        proxy_calls.append(request)
        return httpx.Response(200, json={})

    default_client = httpx.AsyncClient(transport=httpx.MockTransport(default_handler))
    proxy_client = httpx.AsyncClient(transport=httpx.MockTransport(proxy_handler))
    config = GatewayConfig(
        host="127.0.0.1",
        port=8080,
        api_key="local-secret",
        endpoints=[EndpointConfig(
            model="proxied-model",
            base_url="https://upstream.example",
            socks5_proxy=proxy,
        )],
    )
    app = build_app(config, client=default_client)

    def proxy_client_factory(*args, **kwargs):
        assert kwargs["proxy"] == proxy
        return proxy_client

    monkeypatch.setattr(gateway_app.httpx, "AsyncClient", proxy_client_factory)
    with TestClient(app) as client:
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "proxied-model", "messages": []},
        )

    assert resp.status_code == 200
    assert proxy_calls and not default_calls


def test_socks5_client_is_reused_and_closed_with_app(monkeypatch):
    proxy = "socks5://127.0.0.1:1080"
    factory_calls = []
    proxy_calls = []

    def proxy_handler(request: httpx.Request) -> httpx.Response:
        proxy_calls.append(request)
        return httpx.Response(200, json={})

    default_client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(500))
    )
    proxy_client = httpx.AsyncClient(transport=httpx.MockTransport(proxy_handler))
    config = GatewayConfig(
        host="127.0.0.1",
        port=8080,
        api_key="k",
        endpoints=[EndpointConfig(
            model="m1",
            base_url="https://upstream.example",
            socks5_proxy=proxy,
        )],
    )
    app = build_app(config, client=default_client)

    def proxy_client_factory(*args, **kwargs):
        factory_calls.append(kwargs["proxy"])
        return proxy_client

    monkeypatch.setattr(gateway_app.httpx, "AsyncClient", proxy_client_factory)
    with TestClient(app) as client:
        for _ in range(2):
            resp = client.post("/v1/chat/completions", json={"model": "m1", "messages": []})
            assert resp.status_code == 200

    assert factory_calls == [proxy]
    assert len(proxy_calls) == 2
    assert proxy_client.is_closed
    assert not default_client.is_closed
    asyncio.run(default_client.aclose())


def test_admin_socks5_proxy_update_takes_effect_next_request(tmp_path, monkeypatch):
    proxy_a = "socks5://127.0.0.1:1080"
    proxy_b = "socks5://127.0.0.1:1081"
    proxy_b_calls = []

    def default_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    def proxy_a_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    def proxy_b_handler(request: httpx.Request) -> httpx.Response:
        proxy_b_calls.append(request)
        return httpx.Response(200, json={})

    default_client = httpx.AsyncClient(transport=httpx.MockTransport(default_handler))
    proxy_a_client = httpx.AsyncClient(transport=httpx.MockTransport(proxy_a_handler))
    proxy_b_client = httpx.AsyncClient(transport=httpx.MockTransport(proxy_b_handler))
    config = GatewayConfig(
        host="127.0.0.1",
        port=8080,
        api_key="k",
        endpoints=[EndpointConfig(
            model="m1",
            base_url="https://upstream.example",
            socks5_proxy=proxy_a,
        )],
    )
    p = tmp_path / "config.json"
    p.write_text(json.dumps(config_to_dict(config)))
    app = build_app(config, client=default_client, config_path=str(p),
                    watcher_cache_path=tmp_path / "cache.json")

    def proxy_client_factory(*args, **kwargs):
        return {proxy_a: proxy_a_client, proxy_b: proxy_b_client}[kwargs["proxy"]]

    monkeypatch.setattr(gateway_app.httpx, "AsyncClient", proxy_client_factory)
    client = TestClient(app)
    resp = client.put(
        "/admin/api/endpoints?base_url=https%3A%2F%2Fupstream.example",
        json={"custom_headers": {}, "socks5_proxy": proxy_b},
    )
    assert resp.status_code == 200
    resp = client.post("/v1/chat/completions", json={"model": "m1", "messages": []})

    assert resp.status_code == 200
    assert proxy_b_calls
    for c in (default_client, proxy_a_client, proxy_b_client):
        asyncio.run(c.aclose())


def test_client_auth_forwarded_as_is():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["headers"] = request.headers
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            stream=httpx.ByteStream(b"{}"),
        )

    client = make_client(handler)
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "claude-3-5-sonnet", "messages": []},
        headers={"Authorization": "Bearer wrong-key"},
    )
    assert resp.status_code == 200
    assert captured["headers"]["authorization"] == "Bearer wrong-key"
    assert captured["headers"]["x-api-key"] == "wrong-key"


def test_forwards_openai_request_with_client_auth():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["auth"] = request.headers.get("authorization")
        captured["custom"] = request.headers.get("x-custom-user")
        captured["body"] = request.content
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            stream=httpx.ByteStream(
                json.dumps({"choices": [{"message": {"content": "hi"}}]}).encode()
            ),
        )

    client = make_client(handler)
    body = {"model": "claude-3-5-sonnet", "messages": [{"role": "user", "content": "hello"}]}
    resp = client.post(
        "/v1/chat/completions?x=1",
        json=body,
        headers={"Authorization": "Bearer client-key", "x-custom-user": "abc"},
    )
    assert resp.status_code == 200
    assert captured["url"] == "https://anthropic.example/v1/chat/completions?x=1"
    assert captured["auth"] == "Bearer client-key"
    assert captured["custom"] == "abc"
    assert json.loads(captured["body"]) == body
    assert resp.json() == {"choices": [{"message": {"content": "hi"}}]}


def test_forwards_anthropic_request():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            stream=httpx.ByteStream(
                json.dumps({"content": [{"text": "hi"}]}).encode()
            ),
        )

    client = make_client(handler)
    resp = client.post(
        "/v1/messages",
        json={"model": "claude-3-5-sonnet", "messages": []},
        headers={"x-api-key": "local-secret", "anthropic-version": "2023-06-01"},
    )
    assert resp.status_code == 200
    assert captured["url"] == "https://anthropic.example/v1/messages"
    assert resp.json() == {"content": [{"text": "hi"}]}


def test_unknown_model_returns_404():
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("should not be called")

    client = make_client(handler)
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "nope", "messages": []},
        headers={"Authorization": "Bearer local-secret"},
    )
    assert resp.status_code == 404
    assert "nope" in resp.json()["error"]["message"]


def test_missing_model_returns_400():
    gateway_app.working_model = None  # 重置全局回退，验证"首次缺 model 报 400"

    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("should not be called")

    client = make_client(handler)
    resp = client.post(
        "/v1/messages",
        json={"messages": []},
        headers={"Authorization": "Bearer local-secret"},
    )
    assert resp.status_code == 400
    assert "model" in resp.json()["error"]["message"]


def test_upstream_error_passthrough():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429,
            headers={"content-type": "application/json"},
            stream=httpx.ByteStream(
                json.dumps({"error": {"message": "rate limited"}}).encode()
            ),
        )

    client = make_client(handler)
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4o", "messages": []},
        headers={"Authorization": "Bearer local-secret"},
    )
    assert resp.status_code == 429
    assert resp.json() == {"error": {"message": "rate limited"}}


def test_upstream_connect_error_returns_502():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom")

    client = make_client(handler)
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4o", "messages": []},
        headers={"Authorization": "Bearer local-secret"},
    )
    assert resp.status_code == 502
    assert resp.json() == {"error": {"message": "failed to reach upstream"}}


def test_sse_stream_passthrough():
    chunks = [b"data: {\"a\": 1}\n\n", b"data: [DONE]\n\n"]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=httpx.ByteStream(b"".join(chunks)),
        )

    client = make_client(handler)
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={"model": "gpt-4o", "messages": [], "stream": True},
        headers={"Authorization": "Bearer local-secret"},
    ) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "text/event-stream"
        assert b"".join(resp.iter_bytes()) == b"".join(chunks)


def test_non_sse_response_still_passed_through():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            stream=httpx.ByteStream(json.dumps({"choices": []}).encode()),
        )

    client = make_client(handler)
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4o", "messages": []},
        headers={"Authorization": "Bearer local-secret"},
    )
    assert resp.status_code == 200
    assert resp.json() == {"choices": []}


def test_gzipped_upstream_passthrough():
    raw = gzip.compress(json.dumps({"choices": []}).encode())

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-encoding": "gzip", "content-type": "application/json"},
            stream=httpx.ByteStream(raw),
        )

    client = make_client(handler)
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={"model": "gpt-4o", "messages": []},
        headers={"Authorization": "Bearer local-secret"},
    ) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-encoding"] == "gzip"
        assert b"".join(resp.iter_raw()) == raw


class RecordingResponse(httpx.Response):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.aclose_called = False

    async def aclose(self):
        self.aclose_called = True
        await super().aclose()


class ChunkedStream(httpx.AsyncByteStream):
    def __init__(self, chunks):
        self._chunks = chunks

    async def __aiter__(self):
        for chunk in self._chunks:
            yield chunk
            await asyncio.sleep(0.01)

    async def aclose(self):
        pass


class DyingStream(httpx.AsyncByteStream):
    """An upstream SSE stream that yields a chunk then dies mid-body — some
    providers drop the connection without completing the chunked response
    (httpcore.RemoteProtocolError)."""

    def __init__(self, chunks):
        self._chunks = chunks

    async def __aiter__(self):
        for chunk in self._chunks:
            yield chunk
            await asyncio.sleep(0.01)
        raise httpx.RemoteProtocolError(
            "peer closed connection without sending complete message body "
            "(incomplete chunked read)"
        )

    async def aclose(self):
        pass


def test_sse_stream_cancel_closes_upstream():
    captured = {}
    chunks = [b"chunk1\n\n", b"chunk2\n\n", b"chunk3\n\n"]

    def handler(request: httpx.Request) -> httpx.Response:
        captured["resp"] = RecordingResponse(
            200,
            headers={"content-type": "text/event-stream"},
            stream=ChunkedStream(chunks),
        )
        return captured["resp"]

    config = make_config()
    upstream_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    app = build_app(config, client=upstream_client)

    async def run():
        body = json.dumps({"model": "gpt-4o", "messages": [], "stream": True}).encode()
        request_sent = False
        disconnect = asyncio.Event()
        scope = {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.3"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/v1/chat/completions",
            "raw_path": b"/v1/chat/completions",
            "query_string": b"",
            "root_path": "",
            "headers": [
                (b"authorization", b"Bearer local-secret"),
                (b"content-type", b"application/json"),
            ],
            "client": ("testclient", 50000),
            "server": ("testserver", 80),
        }

        async def receive():
            nonlocal request_sent
            if not request_sent:
                request_sent = True
                return {"type": "http.request", "body": body, "more_body": False}
            await disconnect.wait()
            return {"type": "http.disconnect"}

        async def send(message):
            if message["type"] == "http.response.body" and message.get("more_body"):
                disconnect.set()

        try:
            await app(scope, receive, send)
        except asyncio.CancelledError:
            pass

    asyncio.run(run())
    assert captured["resp"].aclose_called is True


def test_sse_upstream_mid_stream_error_is_graceful():
    """If the upstream drops the SSE connection mid-body (RemoteProtocolError,
    which some providers send), the gateway must stop streaming cleanly and
    close the upstream instead of crashing the ASGI app with an unhandled
    exception."""
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["resp"] = RecordingResponse(
            200,
            headers={"content-type": "text/event-stream"},
            stream=DyingStream([b"data: {\"choices\":[]}\n\n", b"data: {\"choices\":[]}\n\n"]),
        )
        return captured["resp"]

    config = make_config()
    upstream_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    app = build_app(config, client=upstream_client)
    client = TestClient(app)
    body = json.dumps({"model": "gpt-4o", "messages": [], "stream": True}).encode()
    with client.stream(
        "POST",
        "/v1/chat/completions",
        content=body,
        headers={"Authorization": "Bearer local-secret"},
    ) as resp:
        assert resp.status_code == 200
        # Partial SSE data was delivered before the upstream died; no
        # exception escapes the app.
        data = b"".join(resp.iter_raw())
    assert data
    assert captured["resp"].aclose_called is True


def test_sse_converted_upstream_dies_emits_terminator():
    """When a converting stream's upstream drops mid-body (opencode's
    RemoteProtocolError), the gateway must emit the target protocol's
    terminator (Anthropic message_stop) so the client's parser finishes
    instead of hanging on a dangling open content block."""
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        # Upstream speaks OpenAI chat (Responses-ish SSE); dies mid-body.
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=DyingStream([
                b'data: {"id":"c1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"hi"},"finish_reason":null}]}\n\n',
            ]),
        )

    client = _converted_client(handler)
    body = json.dumps({
        "model": "anth-to-oa",
        "max_tokens": 100,
        "stream": True,
        "messages": [{"role": "user", "content": "hi"}],
    }).encode()
    with client.stream(
        "POST",
        "/v1/messages",
        content=body,
        headers={"Authorization": "Bearer local-secret"},
    ) as resp:
        assert resp.status_code == 200
        data = b"".join(resp.iter_raw())
    # The converted Anthropic stream is delivered (partial), then the upstream
    # dies — the gateway must append message_stop so the client can finish.
    assert b"message_stop" in data
    assert b'"type": "message_stop"' in data


def test_x_api_key_auto_added_when_authorization_present():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["headers"] = request.headers
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            stream=httpx.ByteStream(b"{}"),
        )

    client = make_client(handler)
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "claude-3-5-sonnet", "messages": []},
        headers={"Authorization": "Bearer client-key"},
    )
    assert resp.status_code == 200
    assert captured["headers"]["authorization"] == "Bearer client-key"
    assert captured["headers"]["x-api-key"] == "client-key"


def test_x_api_key_and_authorization_synced():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["headers"] = request.headers
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            stream=httpx.ByteStream(b"{}"),
        )

    client = make_client(handler)
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "claude-3-5-sonnet", "messages": []},
        headers={"x-api-key": "client-key"},
    )
    assert resp.status_code == 200
    assert captured["headers"]["x-api-key"] == "client-key"
    assert captured["headers"]["authorization"] == "bearer client-key"


def test_sentinel_authorization_ignored_uses_config_auth():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["headers"] = request.headers
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            stream=httpx.ByteStream(b"{}"),
        )

    client = make_client(handler)
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4o", "messages": []},
        headers={"Authorization": "Bearer local-secret"},
    )
    assert resp.status_code == 200
    assert captured["headers"]["authorization"] == "Bearer real-key"
    assert captured["headers"]["x-api-key"] == "real-key"


def test_sentinel_x_api_key_derives_from_endpoint_auth():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["headers"] = request.headers
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            stream=httpx.ByteStream(b"{}"),
        )

    client = make_client(handler)
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4o", "messages": []},
        headers={"x-api-key": "local-secret"},
    )
    assert resp.status_code == 200
    # The sentinel is dropped entirely; the endpoint's authorization default
    # stands and the sibling x-api-key is derived from it — no sentinel leaks
    # upstream (an Anthropic-style upstream reading x-api-key authenticates).
    assert captured["headers"]["authorization"] == "Bearer real-key"
    assert captured["headers"]["x-api-key"] == "real-key"


def test_sentinel_in_either_auth_header_uses_endpoint_auth():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["headers"] = request.headers
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            stream=httpx.ByteStream(b"{}"),
        )

    client = make_client(handler)
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4o", "messages": []},
        headers={"Authorization": "Bearer client-key", "x-api-key": "local-secret"},
    )
    assert resp.status_code == 200
    # Either header matching the sentinel means "use the endpoint's auth":
    # both client auth values are dropped (real or not), the endpoint's
    # authorization default stands, and x-api-key is derived from it.
    assert captured["headers"]["authorization"] == "Bearer real-key"
    assert captured["headers"]["x-api-key"] == "real-key"


def test_sentinel_without_endpoint_auth_sends_no_auth():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["headers"] = request.headers
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            stream=httpx.ByteStream(b"{}"),
        )

    client = make_client(handler)
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "claude-3-5-sonnet", "messages": []},
        headers={"Authorization": "Bearer local-secret"},
    )
    assert resp.status_code == 200
    # The sentinel is dropped; with no endpoint auth configured there is
    # nothing to derive, so no auth headers are forwarded.
    assert captured["headers"].get("authorization", "") == ""
    assert captured["headers"].get("x-api-key", "") == ""


def test_client_real_auth_wins_endpoint_defaults():
    config = GatewayConfig(
        host="127.0.0.1",
        port=8080,
        api_key="local-secret",
        endpoints=[
            EndpointConfig(
                model="m",
                base_url="https://upstream.example",
                custom_headers={
                    "Authorization": "Bearer real-key",
                    "User-Agent": "my-agent/1.0",
                },
            ),
        ],
    )
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["headers"] = request.headers
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            stream=httpx.ByteStream(b"{}"),
        )

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = TestClient(build_app(config, client=http_client))
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "m", "messages": []},
        headers={
            "Authorization": "Bearer client-key",
            "User-Agent": "curl/8.0",
        },
    )
    assert resp.status_code == 200
    # The client's real auth wins over the endpoint default; non-auth headers
    # keep the endpoint's custom_headers as defaults.
    assert captured["headers"]["authorization"] == "Bearer client-key"
    assert captured["headers"]["user-agent"] == "my-agent/1.0"
    assert captured["headers"]["x-api-key"] == "client-key"


def test_client_missing_one_auth_header_gets_other_from_config():
    config = GatewayConfig(
        host="127.0.0.1",
        port=8080,
        api_key="local-secret",
        endpoints=[
            EndpointConfig(
                model="m",
                base_url="https://upstream.example",
                custom_headers={
                    "Authorization": "Bearer real-key",
                },
            ),
        ],
    )
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["headers"] = request.headers
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            stream=httpx.ByteStream(b"{}"),
        )

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = TestClient(build_app(config, client=http_client))
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "m", "messages": []},
        headers={"x-api-key": "client-key"},
    )
    assert resp.status_code == 200
    # The client's real x-api-key wins over the endpoint's authorization
    # default; the sibling authorization is derived from it.
    assert captured["headers"]["x-api-key"] == "client-key"
    assert captured["headers"]["authorization"] == "bearer client-key"


def test_upstream_model_rewritten_in_forwarded_body():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = request.content
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            stream=httpx.ByteStream(json.dumps({"choices": []}).encode()),
        )

    client = make_client(handler)
    body = {"model": "rewrite-model", "messages": [{"role": "user", "content": "hi"}]}
    resp = client.post(
        "/v1/chat/completions",
        json=body,
        headers={"Authorization": "Bearer local-secret"},
    )
    assert resp.status_code == 200
    forwarded = json.loads(captured["body"])
    assert forwarded["model"] == "real-upstream-model"
    assert forwarded["messages"] == body["messages"]


def test_upstream_model_rewritten_anthropic_path():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = request.content
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            stream=httpx.ByteStream(json.dumps({"content": []}).encode()),
        )

    client = make_client(handler)
    resp = client.post(
        "/v1/messages",
        json={"model": "rewrite-model", "messages": []},
        headers={"x-api-key": "local-secret"},
    )
    assert resp.status_code == 200
    forwarded = json.loads(captured["body"])
    assert forwarded["model"] == "real-upstream-model"


def test_debug_log_disabled_by_default(caplog):
    caplog.set_level(logging.DEBUG, logger="llm-gateway")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            stream=httpx.ByteStream(b"{}"),
        )

    client = make_client(handler)
    client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4o", "messages": []},
        headers={"Authorization": "Bearer client-key"},
    )
    assert "request:" not in caplog.text
    assert "client_headers:" not in caplog.text


def test_debug_log_logs_full_request(caplog):
    caplog.set_level(logging.DEBUG, logger="llm-gateway")
    config = make_debug_config()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            stream=httpx.ByteStream(b"{}"),
        )

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = TestClient(build_app(config, client=http_client))
    resp = client.post(
        "/v1/chat/completions?x=1",
        json={"model": "m", "messages": [{"role": "user", "content": "hello"}]},
        headers={"Authorization": "Bearer client-key", "x-custom-header": "custom-value"},
    )
    assert resp.status_code == 200
    assert "request: POST /v1/chat/completions?x=1" in caplog.text
    assert '"model": "m"' in caplog.text
    assert '"content": "hello"' in caplog.text
    assert "客户端请求头:" in caplog.text
    assert "转换后的请求头:" in caplog.text
    assert "客户端请求body:" in caplog.text
    assert "转换后的请求body:" in caplog.text


def test_debug_log_json_body_included(caplog):
    caplog.set_level(logging.DEBUG, logger="llm-gateway")
    config = make_debug_config()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            stream=httpx.ByteStream(b"{}"),
        )

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = TestClient(build_app(config, client=http_client))
    client.post(
        "/v1/chat/completions",
        json={"model": "m", "messages": [{"role": "user", "content": "hello"}]},
        headers={"Authorization": "Bearer client-key"},
    )
    assert '"model": "m"' in caplog.text
    assert '"content": "hello"' in caplog.text
    assert "客户端请求body:" in caplog.text


def test_debug_log_non_json_body_rejected(caplog):
    """A non-JSON body is rejected with 400 before any debug logging happens —
    the model is read from the JSON body, so the gateway must not proxy it."""
    caplog.set_level(logging.DEBUG, logger="llm-gateway")
    config = make_debug_config()

    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("upstream must not be called for a non-JSON body")

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = TestClient(build_app(config, client=http_client))
    resp = client.post(
        "/v1/chat/completions",
        content=b"not-json-body",
        headers={"Authorization": "Bearer client-key", "Content-Type": "text/plain"},
    )
    assert resp.status_code == 400
    assert "not valid JSON" in resp.json()["error"]["message"]


def test_system_prompt_replaces_first_openai_system_message():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = request.content
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            stream=httpx.ByteStream(b"{}"),
        )

    client = make_client(handler)
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "system-model",
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "system", "content": "old1"},
                {"role": "system", "content": "old2"},
            ],
        },
        headers={"Authorization": "Bearer client-key"},
    )
    assert resp.status_code == 200
    messages = json.loads(captured["body"])["messages"]
    # The injected prompt lands in slot 0; the client's system entries in the
    # first four slots are removed rather than rewritten in place.
    assert messages[0] == {"role": "system", "content": "configured system prompt"}
    assert messages[1] == {"role": "user", "content": "hi"}
    assert len(messages) == 2


def test_system_prompt_injected_when_missing_openai():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = request.content
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            stream=httpx.ByteStream(b"{}"),
        )

    client = make_client(handler)
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "system-model",
            "messages": [{"role": "user", "content": "hi"}],
        },
        headers={"Authorization": "Bearer client-key"},
    )
    assert resp.status_code == 200
    messages = json.loads(captured["body"])["messages"]
    assert messages[0] == {"role": "system", "content": "configured system prompt"}
    assert messages[1] == {"role": "user", "content": "hi"}


def test_system_prompt_replaces_anthropic_top_level():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = request.content
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            stream=httpx.ByteStream(b"{}"),
        )

    client = make_client(handler)
    resp = client.post(
        "/v1/messages",
        json={
            "model": "system-model",
            "system": "old system",
            "messages": [{"role": "user", "content": "hi"}],
        },
        headers={"x-api-key": "client-key"},
    )
    assert resp.status_code == 200
    data = json.loads(captured["body"])
    assert data["system"] == anthropic_system_block("configured system prompt")


def test_system_prompt_injected_when_missing_anthropic():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = request.content
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            stream=httpx.ByteStream(b"{}"),
        )

    client = make_client(handler)
    resp = client.post(
        "/v1/messages",
        json={
            "model": "system-model",
            "messages": [{"role": "user", "content": "hi"}],
        },
        headers={"x-api-key": "client-key"},
    )
    assert resp.status_code == 200
    data = json.loads(captured["body"])
    assert data["system"] == anthropic_system_block("configured system prompt")
    assert data["messages"] == [{"role": "user", "content": "hi"}]


def test_system_prompt_replaces_responses_instructions():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = request.content
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            stream=httpx.ByteStream(b"{}"),
        )

    client = make_client(handler)
    resp = client.post(
        "/v1/responses",
        json={
            "model": "system-model",
            "instructions": "old instructions",
            "input": "hi",
        },
        headers={"Authorization": "Bearer client-key"},
    )
    assert resp.status_code == 200
    data = json.loads(captured["body"])
    assert data["instructions"] == "configured system prompt"
    assert data["input"] == "hi"
    # responses payloads have no messages array — nothing must be injected
    assert "messages" not in data


def test_system_prompt_injected_when_missing_responses():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = request.content
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            stream=httpx.ByteStream(b"{}"),
        )

    client = make_client(handler)
    resp = client.post(
        "/v1/responses",
        json={"model": "system-model", "input": "hi"},
        headers={"Authorization": "Bearer client-key"},
    )
    assert resp.status_code == 200
    data = json.loads(captured["body"])
    assert data["instructions"] == "configured system prompt"
    assert "messages" not in data


def test_upstream_model_and_system_prompt_together():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = request.content
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            stream=httpx.ByteStream(b"{}"),
        )

    client = make_client(handler)
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "combined",
            "messages": [
                {"role": "system", "content": "old"},
                {"role": "user", "content": "hi"},
            ],
        },
        headers={"Authorization": "Bearer client-key"},
    )
    assert resp.status_code == 200
    data = json.loads(captured["body"])
    assert data["model"] == "real-combined"
    assert data["messages"][0] == {"role": "system", "content": "combined sys"}
    assert data["messages"][1] == {"role": "user", "content": "hi"}


def test_models_list_returns_configured_models():
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("models list must not hit upstream")

    client = make_client(handler)
    resp = client.get("/v1/models")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/json"
    assert resp.json() == {
        "data": [
            {"id": "claude-3-5-sonnet", "type": "model"},
            {"id": "combined", "type": "model"},
            {"id": "gpt-4o", "type": "model"},
            {"id": "rewrite-model", "type": "model"},
            {"id": "system-model", "type": "model"},
        ],
    }


def test_models_list_head_supported():
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("models list must not hit upstream")

    client = make_client(handler)
    resp = client.head("/v1/models")
    assert resp.status_code == 200


def test_models_list_post_is_not_proxied():
    """POST /v1/models is not a protocol path: it must 501, not be forwarded.

    /v1/models is a gateway-local listing endpoint (GET/HEAD only). A POST to
    it used to fall through to the catch-all and be proxied to
    {base_url}/v1/models, which is not something the upstream serves either.
    """
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return httpx.Response(200, json={})

    client = make_client(handler)
    resp = client.post("/v1/models", json={"model": "gpt-4o", "messages": []})
    assert resp.status_code == 501
    assert calls == []


def test_unsupported_path_returns_501_without_reading_multipart():
    """Files-API uploads (multipart/binary) are 501, never a 500 or a proxied 200.

    Regression: ``json.loads`` on a PNG body raised UnicodeDecodeError (500),
    and for a pinned model a parseable body was forwarded to /v1/responses
    with an empty ``input`` list — a bogus success. Neither may happen.
    """
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return httpx.Response(200, json={})

    client = make_client(handler)
    png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 400

    upload = client.post("/v1/files", files={"file": ("a.png", png, "image/png")})
    assert upload.status_code == 501
    assert "message" in upload.json()["error"]

    count = client.post("/v1/messages/count_tokens",
                        json={"model": "gpt-4o", "messages": []})
    assert count.status_code == 501
    assert calls == []


def test_nonstream_usage_recorded(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "hi"}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20,
                      "prompt_tokens_details": {"cached_tokens": 40}},
        })

    config = make_config()
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    app = build_app(config, client=http_client, stats_db_path=tmp_path / "usage.db")
    client = TestClient(app)
    resp = client.post("/v1/chat/completions", json={"model": "gpt-4o", "messages": []})
    assert resp.status_code == 200
    db = app.state.stats_db
    items, _ = db.list_requests(0, 2**31, None, None, 10, 0)
    assert len(items) == 1
    row = items[0]
    assert row["model"] == "gpt-4o"
    assert row["base_url"] == "https://upstream.example"
    assert row["input_tokens"] == 60
    assert row["cache_read_tokens"] == 40
    assert row["output_tokens"] == 20
    assert row["status"] == 200
    assert row["cost_usd"] is None          # worker not started (no lifespan)
    app.state.cost_worker.process_pending()
    items, _ = db.list_requests(0, 2**31, None, None, 10, 0)
    # (3*60 + 0.3*40 + 15*20) / 1e6
    assert items[0]["cost_usd"] == 0.000492


def test_sse_usage_recorded_and_bytes_forwarded(tmp_path):
    sse_body = (
        b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'
        b'data: {"choices":[],"usage":{"prompt_tokens":100,"completion_tokens":20,"cached_tokens":40}}\n\n'
        b'data: [DONE]\n\n'
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/event-stream"},
                              stream=httpx.ByteStream(sse_body))

    config = make_config()
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    app = build_app(config, client=http_client, stats_db_path=tmp_path / "usage.db")
    client = TestClient(app)
    with client.stream("POST", "/v1/chat/completions", json={"model": "gpt-4o", "messages": []}) as resp:
        got = b"".join(resp.iter_raw())
    assert resp.status_code == 200
    assert got == sse_body                  # forwarded byte-identical
    items, _ = app.state.stats_db.list_requests(0, 2**31, None, None, 10, 0)
    assert len(items) == 1
    assert items[0]["input_tokens"] == 60
    assert items[0]["cache_read_tokens"] == 40
    assert items[0]["output_tokens"] == 20


def test_non_json_response_recorded_with_zero_tokens(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(502, headers={"content-type": "text/html"},
                              stream=httpx.ByteStream(b"<html>bad gateway</html>"))

    config = make_config()
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    app = build_app(config, client=http_client, stats_db_path=tmp_path / "usage.db")
    client = TestClient(app)
    resp = client.post("/v1/chat/completions", json={"model": "gpt-4o", "messages": []})
    assert resp.status_code == 502
    items, _ = app.state.stats_db.list_requests(0, 2**31, None, None, 10, 0)
    assert len(items) == 1
    assert items[0]["status"] == 502
    assert items[0]["input_tokens"] == 0
    assert items[0]["output_tokens"] == 0


def test_gzip_response_records_usage(tmp_path):
    payload = b'{"choices": [], "usage": {"prompt_tokens": 100, "completion_tokens": 20, "cached_tokens": 40}}'

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "application/json",
                                            "content-encoding": "gzip"},
                              stream=httpx.ByteStream(gzip.compress(payload)))

    config = make_config()
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    app = build_app(config, client=http_client, stats_db_path=tmp_path / "usage.db")
    client = TestClient(app)
    resp = client.post("/v1/chat/completions", json={"model": "gpt-4o", "messages": []})
    assert resp.status_code == 200
    items, _ = app.state.stats_db.list_requests(0, 2**31, None, None, 10, 0)
    assert len(items) == 1
    assert items[0]["input_tokens"] == 60
    assert items[0]["cache_read_tokens"] == 40
    assert items[0]["output_tokens"] == 20


def test_gzip_sse_records_usage(tmp_path):
    sse = (b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'
           b'data: {"choices":[],"usage":{"prompt_tokens":100,"completion_tokens":20,"cached_tokens":40}}\n\n'
           b'data: [DONE]\n\n')
    gz = gzip.compress(sse)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/event-stream",
                                            "content-encoding": "gzip"},
                              stream=httpx.ByteStream(gz))

    config = make_config()
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    app = build_app(config, client=http_client, stats_db_path=tmp_path / "usage.db")
    client = TestClient(app)
    with client.stream("POST", "/v1/chat/completions", json={"model": "gpt-4o", "messages": []}) as resp:
        got = b"".join(resp.iter_raw())
    assert resp.status_code == 200
    assert got == gz            # compressed bytes forwarded unchanged
    items, _ = app.state.stats_db.list_requests(0, 2**31, None, None, 10, 0)
    assert len(items) == 1
    assert items[0]["input_tokens"] == 60
    assert items[0]["cache_read_tokens"] == 40
    assert items[0]["output_tokens"] == 20


def test_record_failure_does_not_break_proxy(tmp_path, monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [], "usage": {"prompt_tokens": 5,
                                                                  "completion_tokens": 1}})

    config = make_config()
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    app = build_app(config, client=http_client, stats_db_path=tmp_path / "usage.db")

    import gateway.stats.db as db_module

    def boom(*args, **kwargs):
        raise RuntimeError("db exploded")

    monkeypatch.setattr(db_module.StatsDb, "insert", boom)
    client = TestClient(app)
    resp = client.post("/v1/chat/completions", json={"model": "gpt-4o", "messages": []})
    assert resp.status_code == 200
    assert resp.json()["usage"]["prompt_tokens"] == 5


def test_worker_fills_cost_in_lifespan(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [],
                                         "usage": {"prompt_tokens": 1_000_000,
                                                   "completion_tokens": 0}})

    config = make_config()
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    app = build_app(config, client=http_client, stats_db_path=tmp_path / "usage.db")
    with TestClient(app) as client:         # lifespan runs: worker started
        client.post("/v1/chat/completions", json={"model": "gpt-4o", "messages": []})
        deadline = time.monotonic() + 5
        while True:
            items, _ = app.state.stats_db.list_requests(0, 2**31, None, None, 10, 0)
            if items and items[0]["cost_usd"] is not None:
                break
            assert time.monotonic() < deadline, "worker did not fill cost in time"
            time.sleep(0.05)
        assert items[0]["cost_usd"] == 3.0




def test_duration_recorded_for_streaming(tmp_path):
    sse = (b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'
           b'data: {"choices":[],"usage":{"prompt_tokens":100,"completion_tokens":20}}\n\n'
           b'data: [DONE]\n\n')

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/event-stream"},
                              stream=httpx.ByteStream(sse))

    config = make_config()
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    app = build_app(config, client=http_client, stats_db_path=tmp_path / "usage.db")
    client = TestClient(app)
    with client.stream("POST", "/v1/chat/completions", json={"model": "gpt-4o", "messages": []}):
        pass
    items, _ = app.state.stats_db.list_requests(0, 2**31, None, None, 10, 0)
    # duration exposed through model_tps after 3 samples; direct check via SQL
    import sqlite3
    conn = sqlite3.connect(tmp_path / "usage.db")
    dur = conn.execute("SELECT duration_ms FROM requests ORDER BY id DESC LIMIT 1").fetchone()[0]
    assert dur is not None and dur >= 0


def test_ttft_recorded_for_streaming(tmp_path):
    # Opening role-only delta (pre-generation) then a content delta then usage.
    sse = (b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n'
           b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'
           b'data: {"choices":[],"usage":{"prompt_tokens":100,"completion_tokens":20}}\n\n'
           b'data: [DONE]\n\n')

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/event-stream"},
                              stream=httpx.ByteStream(sse))

    config = make_config()
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    app = build_app(config, client=http_client, stats_db_path=tmp_path / "usage.db")
    client = TestClient(app)
    with client.stream("POST", "/v1/chat/completions", json={"model": "gpt-4o", "messages": []}):
        pass
    import sqlite3
    conn = sqlite3.connect(tmp_path / "usage.db")
    dur, ttft = conn.execute(
        "SELECT duration_ms, ttft_ms FROM requests ORDER BY id DESC LIMIT 1").fetchone()
    assert dur is not None and dur >= 0
    assert ttft is not None and 0 <= ttft <= dur


def test_ttft_null_when_no_content_event(tmp_path):
    # Stream with only protocol events (no content delta) -> TTFT stays NULL.
    sse = (b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n'
           b'data: {"choices":[],"usage":{"prompt_tokens":100,"completion_tokens":20}}\n\n'
           b'data: [DONE]\n\n')

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/event-stream"},
                              stream=httpx.ByteStream(sse))

    config = make_config()
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    app = build_app(config, client=http_client, stats_db_path=tmp_path / "usage.db")
    client = TestClient(app)
    with client.stream("POST", "/v1/chat/completions", json={"model": "gpt-4o", "messages": []}):
        pass
    import sqlite3
    conn = sqlite3.connect(tmp_path / "usage.db")
    ttft = conn.execute(
        "SELECT ttft_ms FROM requests ORDER BY id DESC LIMIT 1").fetchone()[0]
    assert ttft is None


def test_ttft_null_for_nonstreaming(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [],
                                         "usage": {"prompt_tokens": 10, "completion_tokens": 5}})

    config = make_config()
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    app = build_app(config, client=http_client, stats_db_path=tmp_path / "usage.db")
    client = TestClient(app)
    resp = client.post("/v1/chat/completions", json={"model": "gpt-4o", "messages": []})
    assert resp.status_code == 200
    import sqlite3
    conn = sqlite3.connect(tmp_path / "usage.db")
    dur, ttft = conn.execute(
        "SELECT duration_ms, ttft_ms FROM requests ORDER BY id DESC LIMIT 1").fetchone()
    assert dur is not None and dur >= 0
    assert ttft is None       # TTFT only meaningful for streaming


# --- routing primitives (merged from test_router.py; routing lives in gateway.config) ---

import pytest

from gateway.config import (
    ModelNotFoundError,
    ModelRouter,
    detect_format,
    load_config,
)


def test_detect_format():
    assert detect_format("/v1/messages") == "anthropic"
    assert detect_format("/v1/chat/completions") == "openai"
    assert detect_format("/v1/responses") == "responses"


def test_detect_format_accepts_trailing_slash_and_prefix():
    assert detect_format("/v1/messages/") == "anthropic"
    assert detect_format("/anthropic/v1/messages") == "anthropic"
    assert detect_format("/proxy/anthropic/v1/messages/") == "anthropic"
    assert detect_format("/v1/responses/") == "responses"
    assert detect_format("/openai/v1/responses") == "responses"
    assert detect_format("/v1/chat/completions/") == "openai"


def test_detect_format_rejects_every_other_path():
    """Only the three protocol endpoints are recognised — no OpenAI fallback.

    The old fallback classified anything unknown as OpenAI, which silently
    forwarded Files/count_tokens/batch requests to /v1/chat/completions (or,
    for a pinned model, rewrote the path to /v1/responses) with a converted
    body. Unsupported paths must be recognisable so the proxy can 501 them.
    """
    # Lookalikes must not match.
    assert detect_format("/v1/messages-extra") is None
    assert detect_format("/v1/responses-extra") is None
    assert detect_format("/v1/chat/completions-extra") is None
    # Real endpoints of the wider API surface: all unsupported.
    assert detect_format("/v1/models") is None
    assert detect_format("/v1/files") is None
    assert detect_format("/v1/files/file-abc/content") is None
    assert detect_format("/v1/messages/count_tokens") is None
    assert detect_format("/v1/batches") is None
    assert detect_format("/") is None


def test_router_exact_match():
    router = ModelRouter(make_config())
    ep = router.route("gpt-4o")
    assert ep.base_url == "https://upstream.example"
    assert ep.custom_headers == {"Authorization": "Bearer real-key"}


def test_router_missing_model():
    router = ModelRouter(make_config())
    with pytest.raises(ModelNotFoundError):
        router.route("unknown-model")


def test_router_list_models_sorted():
    router = ModelRouter(make_config())
    assert router.list_models() == ["claude-3-5-sonnet", "combined", "gpt-4o",
                                    "rewrite-model", "system-model"]


# ---------------------------------------------------------------------------
# Protocol pinning + conversion through the proxy
# ---------------------------------------------------------------------------

def _converted_client(handler) -> TestClient:
    """Client whose models include pinned-protocol entries:
    - oa-to-anth: OpenAI chat clients -> upstream speaks Anthropic.
    - anth-to-oa: Anthropic clients -> upstream speaks OpenAI chat.
    """
    config = GatewayConfig(
        host="127.0.0.1",
        port=8080,
        api_key="local-secret",
        endpoints=[
            EndpointConfig(
                model="oa-to-anth",
                base_url="https://upstream.example",
                protocol="anthropic_messages",
            ),
            EndpointConfig(
                model="anth-to-oa",
                base_url="https://upstream.example",
                protocol="openai_chat",
            ),
        ],
    )
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    app = build_app(config, client=http_client)
    return TestClient(app)


def test_openai_chat_to_anthropic_request_converted_and_response_converted():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["body"] = json.loads(request.content)
        # Upstream (Anthropic-speaking) returns an Anthropic response.
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            stream=httpx.ByteStream(json.dumps({
                "id": "msg_upstream",
                "type": "message",
                "role": "assistant",
                "model": "claude-upstream",
                "content": [{"type": "text", "text": "hi from anthropic"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 7, "output_tokens": 3},
            }).encode()),
        )

    client = _converted_client(handler)
    resp = client.post("/v1/chat/completions", json={
        "model": "oa-to-anth",
        "messages": [
            {"role": "system", "content": "be nice"},
            {"role": "user", "content": "hi"},
        ],
        "max_tokens": 50,
    })

    assert resp.status_code == 200
    # Request was converted to Anthropic format AND routed to the
    # Anthropic endpoint for the upstream (not the client's chat path).
    assert captured["path"] == "/v1/messages"
    assert captured["body"]["system"] == "be nice"
    assert captured["body"]["messages"] == [{"role": "user", "content": "hi"}]
    # C→A keeps the client's max_tokens verbatim (Anthropic accepts small
    # values; sub2api's floor only applies to Chat/Responses upstream params).
    assert captured["body"]["max_tokens"] == 50
    # Response converted back to OpenAI chat.
    data = resp.json()
    assert data["object"] == "chat.completion"
    assert data["choices"][0]["message"]["content"] == "hi from anthropic"
    assert data["choices"][0]["finish_reason"] == "stop"


def test_anthropic_to_openai_chat_request_converted_and_response_converted():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["body"] = json.loads(request.content)
        # Upstream (OpenAI-speaking) returns an OpenAI chat response.
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            stream=httpx.ByteStream(json.dumps({
                "id": "chatcmpl-up",
                "object": "chat.completion",
                "model": "gpt-upstream",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi oa"},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
            }).encode()),
        )

    client = _converted_client(handler)
    resp = client.post("/v1/messages", json={
        "model": "anth-to-oa",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 50,
        "system": "sys",
    })

    assert resp.status_code == 200
    assert captured["path"] == "/v1/chat/completions"
    assert captured["body"]["messages"][0]["role"] == "system"
    assert captured["body"]["messages"][1]["content"] == "hi"
    # A→C floors max_completion_tokens at 128 (sub2api
    # DefaultMinMaxOutputTokens — very small values cause upstream errors).
    assert captured["body"]["max_completion_tokens"] == 128
    data = resp.json()
    assert data["type"] == "message"
    assert data["content"] == [{"type": "text", "text": "hi oa"}]
    assert data["stop_reason"] == "end_turn"


def test_usage_record_captures_protocol_flow_and_final_reasoning_effort(tmp_path):
    """Usage details must describe the request actually sent upstream:
    client protocol -> pinned protocol, plus the final effort after extra_body
    overrides (not merely the client-supplied value)."""
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["reasoning_effort"] == "high"
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            json={
                "id": "chatcmpl-up",
                "object": "chat.completion",
                "model": "gpt-upstream",
                "choices": [{"index": 0,
                             "message": {"role": "assistant", "content": "hi"},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 4, "completion_tokens": 2,
                          "total_tokens": 6},
            },
        )

    config = GatewayConfig(
        host="127.0.0.1", port=8080, api_key="local-secret",
        extra_body={"openai_chat": {"reasoning_effort": "high"}},
        endpoints=[EndpointConfig(
            model="anth-to-oa",
            base_url="https://upstream.example",
            protocol="openai_chat",
        )],
    )
    app = build_app(
        config,
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        stats_db_path=tmp_path / "usage.db",
    )
    client = TestClient(app)

    resp = client.post("/v1/messages", json={
        "model": "anth-to-oa",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 50,
        "output_config": {"effort": "low"},
    })
    assert resp.status_code == 200

    items, _ = app.state.stats_db.list_requests(0, 2**31, None, None, 10, 0)
    assert len(items) == 1
    assert items[0]["client_protocol"] == "anthropic_messages"
    assert items[0]["upstream_protocol"] == "openai_chat"
    assert items[0]["reasoning_effort"] == "high"


def test_converting_request_missing_max_tokens_defaults_and_forwards():
    """A missing max_tokens defaults to 8192 (sub2api) instead of failing —
    the upstream still receives a valid Anthropic request."""
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            content=json.dumps({
                "id": "msg_1", "type": "message", "role": "assistant",
                "model": "oa-to-anth", "content": [{"type": "text", "text": "ok"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 1, "output_tokens": 1},
            }),
        )

    client = _converted_client(handler)
    resp = client.post("/v1/chat/completions", json={
        "model": "oa-to-anth",
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert resp.status_code == 200
    assert captured["body"]["max_tokens"] == 8192


def test_matching_protocol_is_direct_passthrough():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = request.content
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            stream=httpx.ByteStream(b"{}"),
        )

    config = GatewayConfig(
        host="127.0.0.1", port=8080, api_key="k",
        endpoints=[EndpointConfig(
            model="m", base_url="https://upstream.example", protocol="openai_chat",
        )],
    )
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    app = build_app(config, client=http_client)
    client = TestClient(app)
    body = json.dumps({"model": "m", "messages": [{"role": "user", "content": "hi"}]}).encode()
    resp = client.post("/v1/chat/completions", content=body)
    assert resp.status_code == 200
    # Byte-identical passthrough when format matches the pinned protocol.
    assert captured["body"] == body


def test_converted_response_strips_upstream_content_encoding_and_length():
    """A gzip-compressed upstream body is decompressed for conversion and
    re-emitted as plain JSON; the upstream's content-encoding/content-length
    headers no longer describe it and must not be forwarded — otherwise
    clients (e.g. Claude Code) try to inflate the body and fail with
    ZlibError."""
    raw = gzip.compress(json.dumps({
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "choices": [{"index": 0,
                     "message": {"role": "assistant", "content": "hi"},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
    }).encode())

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "application/json",
                     "content-encoding": "gzip",
                     "content-length": str(len(raw)),
                     "x-upstream-marker": "kept"},
            stream=httpx.ByteStream(raw),
        )

    client = _converted_client(handler)
    resp = client.post("/v1/messages", json={
        "model": "anth-to-oa",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert resp.status_code == 200
    assert "content-encoding" not in resp.headers
    assert resp.headers["content-length"] == str(len(resp.content))
    assert resp.headers["x-upstream-marker"] == "kept"
    assert resp.json()["role"] == "assistant"


def test_sse_stream_conversion_through_proxy():
    """OpenAI chat streaming client -> Anthropic-speaking upstream; the SSE
    stream is converted back into OpenAI chat chunks."""

    def handler(request: httpx.Request) -> httpx.Response:
        events = [
            'event: message_start\ndata: {"type":"message_start","message":{"id":"m1","model":"claude-3"}}',
            'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hi"}}',
            'event: message_stop\ndata: {"type":"message_stop"}',
        ]
        body = ("\n\n".join(events) + "\n\n").encode()
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=httpx.ByteStream(body),
        )

    client = _converted_client(handler)
    resp = client.post("/v1/chat/completions", json={
        "model": "oa-to-anth",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 50,
        "stream": True,
    })
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    text = resp.text
    # Converted back to OpenAI chat SSE: bare data: chunks + [DONE].
    assert 'data: {"id": "m1"' in text
    assert '"delta": {"content": "hi"}' in text
    assert "data: [DONE]" in text
    assert "event:" not in text


def test_watcher_runs_during_lifespan(tmp_path):
    # ModelWatcher 在 lifespan 启动时立即首轮扫描（建基线），随 lifespan 退出停止
    p = tmp_path / "config.json"
    p.write_text(json.dumps({
        "server": {"api_key": "k"},
        "endpoints": [{"base_url": "https://a.com", "models": [{"name": "m1"}]}],
    }))
    cache_path = tmp_path / "cache.json"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"id": "m1"}]})

    app = build_app(
        load_config(p),
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        config_path=str(p),
        watcher_cache_path=cache_path,
        watcher_interval=3600,
    )
    with TestClient(app) as client:
        assert client.get("/admin/api/endpoints").status_code == 200
        assert cache_path.exists()   # first scan ran at startup
    data = json.loads(cache_path.read_text())
    assert data["https://a.com"]["ids"] == ["m1"]


def test_watcher_not_started_without_config_path(tmp_path):
    # 只读模式（无 config_path）不启动 watcher，也不报错
    config = GatewayConfig(
        host="127.0.0.1", port=8080, api_key="k",
        endpoints=[EndpointConfig(model="m1", base_url="https://a.com")],
    )
    cache_path = tmp_path / "cache.json"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"id": "m1"}]})

    app = build_app(
        config,
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        watcher_cache_path=cache_path,
    )
    with TestClient(app) as client:
        assert client.get("/admin/api/endpoints").status_code == 200
    assert not cache_path.exists()


# --- injected prompt template body (request path) ---

def test_request_uses_injected_template_text(tmp_path):
    prompts = tmp_path / "prompts" / "system-prompt"
    prompts.mkdir(parents=True)
    (prompts / "prompt.md").write_text("TEMPLATE-BODY", encoding="utf-8")
    captures = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captures["body"] = request.content
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            stream=httpx.ByteStream(b"{}"),
        )

    client = make_client_with_prompts(handler, tmp_path / "prompts", tmp_path)
    resp = client.post("/admin/api/endpoints/models/prompt-template",
                       json={"base_url": "https://upstream.example", "name": "gpt-4o",
                             "template": "system-prompt"})
    assert resp.status_code == 200
    resp = client.post("/v1/messages", json={"model": "gpt-4o", "messages": []})
    assert resp.status_code == 200
    assert b"TEMPLATE-BODY" in captures["body"]


def test_request_uses_cached_template_after_dir_deleted(tmp_path):
    root = tmp_path / "prompts" / "system-prompt"
    root.mkdir(parents=True)
    (root / "prompt.md").write_text("TEMPLATE-BODY", encoding="utf-8")
    captures = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captures["body"] = request.content
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            stream=httpx.ByteStream(b"{}"),
        )

    client = make_client_with_prompts(handler, tmp_path / "prompts", tmp_path)
    resp = client.post("/admin/api/endpoints/models/prompt-template",
                       json={"base_url": "https://upstream.example", "name": "gpt-4o",
                             "template": "system-prompt"})
    assert resp.status_code == 200
    for f in root.iterdir():
        f.unlink()
    root.rmdir()
    client.post("/v1/messages", json={"model": "gpt-4o", "messages": []})
    assert b"TEMPLATE-BODY" in captures["body"]   # cached, still applies


def _config_with_injected_template(template: str) -> GatewayConfig:
    """One handwritten-prompt model pinned to a template name directly.

    Bypasses the admin inject route on purpose: it rejects names that are not
    on disk (400), while a name can become unavailable *after* injection
    (deleted folder) or resolve to a blank body — exactly the fallback cases.
    """
    return GatewayConfig(
        host="127.0.0.1",
        port=8080,
        api_key="local-secret",
        endpoints=[
            EndpointConfig(
                model="gpt-4o",
                base_url="https://upstream.example",
                system_prompt="hand",
                prompt_template=template,
            ),
        ],
    )


def _capture_handler(captures):
    def handler(request: httpx.Request) -> httpx.Response:
        captures["body"] = request.content
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            stream=httpx.ByteStream(b"{}"),
        )
    return handler


def test_request_falls_back_to_handwritten_when_template_missing(tmp_path):
    # The template name is configured but its folder never existed: get_prompt
    # returns None and the handwritten prompt must be used (no 500).
    prompts = tmp_path / "prompts"
    prompts.mkdir()
    captures = {}

    client = make_client_with_prompts(
        _capture_handler(captures), prompts, tmp_path,
        config=_config_with_injected_template("ghost"),
    )
    resp = client.post("/v1/messages", json={"model": "gpt-4o", "messages": []})
    assert resp.status_code == 200
    assert json.loads(captures["body"])["system"] == anthropic_system_block("hand")


def test_request_falls_back_to_handwritten_when_template_blank(tmp_path):
    # A whitespace-only prompt.md is "no template" for the store, so the
    # handwritten prompt applies rather than an empty system field.
    root = tmp_path / "prompts" / "blank"
    root.mkdir(parents=True)
    (root / "prompt.md").write_text("  \n\t\n ", encoding="utf-8")
    captures = {}

    client = make_client_with_prompts(
        _capture_handler(captures), tmp_path / "prompts", tmp_path,
        config=_config_with_injected_template("blank"),
    )
    resp = client.post("/v1/messages", json={"model": "gpt-4o", "messages": []})
    assert resp.status_code == 200
    assert json.loads(captures["body"])["system"] == anthropic_system_block("hand")


def test_request_after_restart_regenerates_missing_md_via_hook(tmp_path):
    # 规则 2B：重启后进程缓存为空，第一次用到已注入模板时目录里只有
    # prompt.py（prompt.md 被删了）。必须跑钩子生成正文再注入，而不是悄悄
    # 回落到手写提示词。
    folder = tmp_path / "prompts" / "py-only"
    folder.mkdir(parents=True)
    (folder / "prompt.py").write_text(
        "import pathlib\n"
        "def load_prompt():\n"
        "    pathlib.Path('prompt.md').write_text('RESTORED', encoding='utf-8')\n"
        "    return 0\n", encoding="utf-8")
    captures = {}

    client = make_client_with_prompts(
        _capture_handler(captures), tmp_path / "prompts", tmp_path,
        config=_config_with_injected_template("py-only"),
    )
    assert not (folder / "prompt.md").exists()     # 冷缓存 + 正文缺失
    resp = client.post("/v1/messages", json={"model": "gpt-4o", "messages": []})
    assert resp.status_code == 200
    assert json.loads(captures["body"])["system"] == anthropic_system_block("RESTORED")
    assert (folder / "prompt.md").exists()          # 钩子已把它写回来


def test_request_py_only_without_md_falls_back_to_handwritten(tmp_path):
    # 钩子跑成功却没写出 prompt.md：不注入，回落手写提示词，请求不报错。
    folder = tmp_path / "prompts" / "no-output"
    folder.mkdir(parents=True)
    (folder / "prompt.py").write_text("def load_prompt():\n    return 0\n",
                                      encoding="utf-8")
    captures = {}

    client = make_client_with_prompts(
        _capture_handler(captures), tmp_path / "prompts", tmp_path,
        config=_config_with_injected_template("no-output"),
    )
    resp = client.post("/v1/messages", json={"model": "gpt-4o", "messages": []})
    assert resp.status_code == 200
    assert json.loads(captures["body"])["system"] == anthropic_system_block("hand")


def test_hook_does_not_block_the_event_loop(tmp_path):
    # 规则 2B 的钩子在异步请求路径上跑：必须丢到线程里，否则一次首用会卡住
    # 整个事件循环（HOOK_TIMEOUT 长达 30s）。
    folder = tmp_path / "prompts" / "slow"
    folder.mkdir(parents=True)
    (folder / "prompt.py").write_text(
        "import time\ndef load_prompt():\n"
        "    time.sleep(0.6)\n"
        "    return 0\n", encoding="utf-8")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "application/json"},
                              stream=httpx.ByteStream(b"{}"))

    p = tmp_path / "config.json"
    p.write_text(json.dumps(config_to_dict(_config_with_injected_template("slow"))))
    app = build_app(
        _config_with_injected_template("slow"),
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        config_path=p, watcher_cache_path=tmp_path / "cache.json",
        prompts_dir=tmp_path / "prompts",
    )

    async def run():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport,
                                     base_url="http://test") as client:
            # 后台心跳：事件循环若被钩子阻塞，ticks 会远远少于预期。
            ticks = 0
            stop = asyncio.Event()

            async def heartbeat():
                nonlocal ticks
                while not stop.is_set():
                    ticks += 1
                    await asyncio.sleep(0.02)

            beat = asyncio.create_task(heartbeat())
            resp = await client.post("/v1/messages",
                                     json={"model": "gpt-4o", "messages": []})
            stop.set()
            await beat
            return resp, ticks

    resp, ticks = asyncio.run(run())
    assert resp.status_code == 200
    # 0.6s / 0.02s ≈ 30；被阻塞时只会有 0~1 次。
    assert ticks > 10, f"event loop was blocked (only {ticks} ticks)"
