from __future__ import annotations

import contextlib
import json
import logging
import time
import zlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse

from gateway.admin import register_admin_routes
from gateway.config import (
    EndpointConfig,
    GatewayConfig,
    ModelNotFoundError,
)
from gateway.convert import (
    ConversionFailed,
    UnsupportedPath,
    protocol_convert,
    convert_response_body,
    convert_sse_chunks,
    encode_stream_terminator,
    get_convert_stats,
    require_supported,
    ConvertStats,
    Protocol
)
from gateway.prompt_templates import DEFAULT_PROMPTS_DIR, PromptTemplateStore
from gateway.stats.api import register_stats_routes
from gateway.stats.capture import SseUsageScanner, Usage, extract_usage
from gateway.stats.db import StatsDb
from gateway.stats.pricing import Pricing, resolve_pricing
from gateway.stats.worker import CostWorker
from gateway.store import ConfigStore, StoreError
from gateway.watcher import ModelWatcher, default_cache_path

HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
    "host",
    "content-length"
}

WITH_BODY_METHODS = {"POST", "PUT"}
UPSTREAM_TIMEOUT = httpx.Timeout(connect=60, read=300, write=60, pool=60)

logger = logging.getLogger("llm-gateway")


def _decompressor(encoding: str) -> tuple[Callable[[bytes], bytes], Callable[[], bytes]] | None:
    """Build an incremental decompressor for usage parsing.

    Returns (feed, finish) where feed(bytes) decompresses a chunk and finish()
    flushes trailing output, or None when the encoding is not parseable with
    the stdlib (e.g. br). The forwarded bytes to the client are never touched —
    we decompress only a copy for usage extraction.
    """
    enc = encoding.lower()
    if enc in ("", "identity"):
        return (lambda b: b), (lambda: b"")
    if enc == "gzip":
        d = zlib.decompressobj(16 + zlib.MAX_WBITS)
    elif enc == "deflate":
        d = zlib.decompressobj()
    else:
        return None
    return d.decompress, d.flush


_DECODABLE_ENCODINGS = {"gzip", "deflate", "identity"}


def _rewrite_headers(client_headers: dict,
                     endpoint: EndpointConfig,
                     api_key: str) -> dict[str, str]:
    """Build upstream request headers from the client's (lowercased keys).

    Pipeline (order matters):
    1. Start from the client's headers minus hop-by-hop ones.
    2. Overlay endpoint.custom_headers — non-auth defaults override
       same-named client headers.
    3. Auth precedence: the client's real (non-sentinel) auth value is
       restored over the endpoint's auth default. A client value equal to
       `api_key` is the sentinel: it is left out and the endpoint's configured
       default for that header is used instead. A client that sent no auth at
       all gets no auth upstream (the endpoint default is not a fallback).
    4. Derive the missing sibling: exactly one of Authorization / x-api-key
       non-empty produces the other (empty values count as absent).
    """
    headers = {
        k.lower(): v for k, v in client_headers.items()
        if k.lower() not in HOP_BY_HOP_HEADERS
    }

    # 处理请求头中的编码
    accept_encoding = headers.get("accept-encoding", "")
    if accept_encoding:
        # The gateway must decompress upstream bodies for protocol
        # conversion and usage capture, but only handles gzip/deflate.
        # Strip br/zstd (and the '*'-any wildcard) from the forwarded
        # accept-encoding so the upstream never picks an encoding we
        # can't decode — otherwise buffered conversion fails with a
        # 502 "upstream returned non-JSON for protocol conversion".
        kept = [
            token.strip() for token in accept_encoding.split(",")
            if token.split(";", 1)[0].strip().lower() in _DECODABLE_ENCODINGS
        ]
        headers["accept-encoding"] = ", ".join(kept) if kept else "identity"

    custom_headers = {
        k.lower(): v for k, v in endpoint.custom_headers.items()
    }

    # reset api key
    request_auth = headers.get("authorization", "")
    request_apikey = headers.get("x-api-key", "")

    auth_apikey = request_auth[7:].strip() if request_auth.lower().startswith("bearer ") \
        else request_auth.strip()
    sentinel_sent = (
        auth_apikey == api_key
        or request_apikey.strip() == api_key
    )

    # 把传入的apikey视为本地网关的校验位，如果传入了apikey并且可以正常匹配，替换成自定义头中的apikey
    # 否则保持请求头中的内容正常传递
    if sentinel_sent:
        del headers["authorization"]
        del headers["x-api-key"]
        headers.update(custom_headers)
    else:
        headers.update(custom_headers)
        headers["authorization"] = request_auth
        headers["x-api-key"] = request_apikey

    # 兼容两种协议验证头
    if "x-api-key" not in headers or headers["x-api-key"] == "":
        auth = headers.get("authorization", "")
        key = auth[7:].strip() if auth.lower().startswith("bearer ") else auth.strip()
        if key:
            headers["x-api-key"] = key
    if "authorization" not in headers or headers["authorization"] == "":
        key = headers.get("x-api-key", "").strip()
        if key:
            headers["authorization"] = "bearer " + key

    return headers


def _requests_structured_output(data: dict, protocol: Protocol) -> bool:
    """Whether the request pins a JSON schema instead of free text.

    A structured-output call is machine-facing: the caller reads a parsed
    object, and its own system prompt is the instruction that produces that
    object. Replacing it with a persona template strips the only instruction
    the caller depends on, so the template is skipped for these requests.

    The three protocols spell the constraint differently — Anthropic
    ``output_format`` / ``output_config.format``, Responses ``text.format``,
    Chat ``response_format``. A ``{"type": "text"}`` format is the absence of
    a constraint and does not count.
    """
    if protocol == Protocol.ANTHROPIC_MESSAGES:
        candidates = [data.get("output_format")]
        config = data.get("output_config")
        if isinstance(config, dict):
            candidates.append(config.get("format"))
    elif protocol == Protocol.OPENAI_RESPONSES:
        text = data.get("text")
        candidates = [text.get("format") if isinstance(text, dict) else None]
    else:
        candidates = [data.get("response_format")]

    for candidate in candidates:
        if isinstance(candidate, dict) and candidate.get("type") not in (None, "text"):
            return True
    return False


def _rewrite_system_prompt(data, effective_prompt, final_protocol):
    # 分协议处理系统提示词重写
    if final_protocol == Protocol.ANTHROPIC_MESSAGES:
        # Anthropic accepts ``system`` as a string or as text blocks.
        # Emit a single block carrying a cache breakpoint: a bare string
        # would discard whatever cache_control the client sent and leave
        # the upstream with no marker for the system prefix.
        data["system"] = [
            {
                "type": "text",
                "text": effective_prompt,
                "cache_control": {"type": "ephemeral"},
            }
        ]
    elif final_protocol == Protocol.OPENAI_RESPONSES:
        # Responses has two system channels: the top-level
        # ``instructions`` string and SYSTEM-role items inside ``input``.
        # Both must go, or the client's prompt survives alongside the
        # injected one. Clients put those items at the head of the array;
        # only the first four are considered — anything later is real
        # conversation and stays untouched.
        data.pop("instructions", None)
        items = data.get("input")
        if isinstance(items, list):
            kept = [
                item
                for index, item in enumerate(items)
                if not (
                        index < 4
                        and isinstance(item, dict)
                        and item.get("role") in ("system", "developer")
                )
            ]
            if len(kept) != len(items):
                data["input"] = kept
        data["instructions"] = effective_prompt
    else:
        # Chat accepts system content only at the head of ``messages``,
        # and clients may send ``system`` or ``developer`` there. Drop
        # both from the first four entries, then put the injected prompt
        # in slot 0. Later system/developer turns are conversation, not a
        # system prompt, and stay untouched.
        messages = data.get("messages")
        if isinstance(messages, list):
            kept = [
                msg
                for index, msg in enumerate(messages)
                if not (
                        index < 4
                        and isinstance(msg, dict)
                        and msg.get("role") in ("system", "developer")
                )
            ]
            if len(kept) != len(messages):
                messages = kept
            messages.insert(0, {"role": "system", "content": effective_prompt})
            data["messages"] = messages
        else:
            data["messages"] = [{"role": "system", "content": effective_prompt}]


async def _rewrite_body(app: FastAPI,
                        endpoint: EndpointConfig,
                        data: dict,
                        convert_stats: ConvertStats,
                        extra_body: dict):
    """Rewrite the request body per model config, in place.

    `upstream_model` replaces the `model` field; the `system_prompt` argument
    targets the format's system field — Anthropic top-level `system`,
    Responses top-level `instructions`, OpenAI the first `role: system`
    message (prepended if absent). `extra_body` (per-protocol) overrides
    fields after conversion, with a `null` value deleting the field (see the
    merge loop below).

    The `system_prompt` argument is the request-time override (the injected
    prompt template's body). ``None`` falls back to ``endpoint.system_prompt``;
    an empty string counts as "no prompt": nothing is written, so neither the
    client's own system field nor ``endpoint.system_prompt`` is replaced.
    """

    if endpoint.upstream_model:
        data["model"] = endpoint.upstream_model

    final_protocol = convert_stats.target or convert_stats.source
    if not _requests_structured_output(data, final_protocol):
        # Injected template body wins over endpoint.system_prompt; a
        # missing/blank/unreadable template (get_prompt → None) falls
        # back to the handwritten prompt inside _rewrite_body. Reads go
        # through the store's in-memory cache, so only a cold miss
        # touches the disk.
        prompt_body = None
        if endpoint.prompt_template:
            # aget_prompt: 冷缓存时可能执行模板钩子（最长 30s），必须让出
            # 事件循环 —— 同步版会阻塞所有并发请求。
            prompt_body = await app.state.prompt_templates.aget_prompt(
                endpoint.prompt_template
            )
        prompt_body = prompt_body if prompt_body is not None else endpoint.system_prompt
        _rewrite_system_prompt(data, prompt_body, final_protocol)

    if extra_body and convert_stats.need_convert():
        # Extra params win over converted defaults (e.g. a pinned
        # reasoning.effort from output_config vs a configured one).
        # A value of `null` DELETES the field rather than setting it to
        # null — some upstreams (e.g. opencode's Go backend) reject the
        # parameter name itself (even `stop: null` / `stop: ""` → 400
        # "unknown parameter 'stop'"), so users configure
        # {"openai_responses": {"stop": null}} to strip the field.
        extra = extra_body.get(final_protocol.value)
        if extra:
            for key, value in extra.items():
                if value is None:
                    data.pop(key, None)
                else:
                    data[key] = value


def _reasoning_effort(data: dict) -> str | None:
    """Return the reasoning effort explicitly present in the final upstream body."""
    effort = data.get("reasoning_effort")
    if isinstance(effort, str) and effort:
        return effort

    reasoning = data.get("reasoning")
    if isinstance(reasoning, dict):
        effort = reasoning.get("effort")
        if isinstance(effort, str) and effort:
            return effort

    output_config = data.get("output_config")
    if isinstance(output_config, dict):
        effort = output_config.get("effort")
        if isinstance(effort, str) and effort:
            return effort

    thinking = data.get("thinking")
    if isinstance(thinking, dict):
        thinking_type = thinking.get("type")
        if thinking_type == "disabled":
            return "none"
        if thinking_type == "adaptive":
            return "adaptive"
        budget = thinking.get("budget_tokens")
        if isinstance(budget, (int, float)) and not isinstance(budget, bool) and budget > 0:
            return f"budget:{int(budget)}"
    return None


def _debug_log_request(client_headers: dict, final_headers: dict,
                       client_body: dict, final_body: dict) -> None:
    """Log the full incoming request (bodies pretty-printed when JSON).

    Logs real API keys — enable only in trusted contexts.
    """

    logger.debug(
        "客户端请求头:\n%s\n转换后的请求头:\n%s\n客户端请求body:\n%s\n转换后的请求body:\n%s\n" + "="*20,
        json.dumps(client_headers), json.dumps(final_headers),
        json.dumps(client_body), json.dumps(final_body)
    )


@dataclass(frozen=True)
class _UsageCapture:
    """Per-request usage recorder; a no-op when stats are disabled.

    Holds everything needed to insert one stats row once the response
    completes. `decomp` decompresses a *copy* of the upstream bytes purely
    for usage extraction — the bytes forwarded to the client are untouched,
    and capture is skipped entirely for encodings the stdlib can't parse
    (e.g. br).
    """

    stats_db: StatsDb | None
    endpoint: EndpointConfig
    start_time: float
    status: int
    decomp: tuple[Callable[[bytes], bytes], Callable[[], bytes]] | None
    client_protocol: str | None
    upstream_protocol: str | None
    reasoning_effort: str | None

    @property
    def active(self) -> bool:
        return self.stats_db is not None and self.decomp is not None

    def record_stream(self, scanner: SseUsageScanner, content_at: float | None) -> None:
        """Record usage for an SSE response.

        TTFT = time to the first event carrying model output (content/thinking/
        tool delta), not the first byte — protocol-opening events arrive before
        generation begins.
        """
        ttft = (int((content_at - self.start_time) * 1000)
                if content_at is not None else None)
        self._insert(scanner.usage,
                     duration_ms=int((time.monotonic() - self.start_time) * 1000),
                     ttft_ms=ttft)

    def record_buffered(self, content: bytes) -> None:
        """Record usage for a buffered (non-SSE) response."""
        usage = Usage()
        if self.active:
            try:
                data = json.loads(self.decomp[0](content))
            except (ValueError, TypeError, zlib.error):
                data = None
            if isinstance(data, dict):
                usage = extract_usage(data)
        self._insert(usage,
                     duration_ms=int((time.monotonic() - self.start_time) * 1000))

    def _insert(self, usage: Usage, duration_ms: int,
                ttft_ms: int | None = None) -> None:
        if self.stats_db is None:
            return
        try:
            self.stats_db.insert(
                ts=int(time.time()),
                base_url=self.endpoint.base_url,
                model=self.endpoint.model,
                input_tokens=usage.input_tokens,
                cache_read_tokens=usage.cache_read_tokens,
                cache_write_tokens=usage.cache_write_tokens,
                output_tokens=usage.output_tokens,
                status=self.status,
                duration_ms=duration_ms,
                ttft_ms=ttft_ms,
                client_protocol=self.client_protocol,
                upstream_protocol=self.upstream_protocol,
                reasoning_effort=self.reasoning_effort,
            )
        except Exception:
            logger.exception("failed to record usage stats")


def _sse_response(upstream_resp: httpx.Response, resp_headers: dict[str, str],
                  capture: _UsageCapture, convert_stats: ConvertStats) -> StreamingResponse:
    """Stream an SSE response chunk-by-chunk, scanning a decompressed copy
    for usage; upstream_resp.aclose() is guaranteed in the finally block —
    important for client-disconnect cleanup. When ``converting``, the chunks
    are converted from the model's pinned protocol into the client's format
    before being forwarded."""

    async def _stream_upstream() -> AsyncIterator[bytes]:
        scanner = SseUsageScanner() if capture.active else None
        content_at: float | None = None
        upstream = upstream_resp.aiter_raw()
        chunks: AsyncIterator[bytes] = (
            convert_sse_chunks(upstream, convert_stats)
            if convert_stats.need_convert() else upstream
        )
        try:
            async for chunk in chunks:
                if scanner is not None:
                    scanner.feed(capture.decomp[0](chunk))
                    if content_at is None and scanner.has_content:
                        content_at = time.monotonic()
                yield chunk
        except httpx.HTTPError:
            logger.debug("upstream stream ended prematurely", exc_info=True)
            if convert_stats.need_convert():
                yield encode_stream_terminator(convert_stats.source)
        finally:
            if scanner is not None:
                scanner.feed(capture.decomp[1]())
                scanner.flush()
                capture.record_stream(scanner, content_at)
            await upstream_resp.aclose()

    return StreamingResponse(
        _stream_upstream(),
        status_code=upstream_resp.status_code,
        headers=resp_headers,
    )


async def _buffered_response(upstream_resp: httpx.Response, resp_headers: dict[str, str],
                             capture: _UsageCapture, convert_stats: ConvertStats) -> Response:
    """Buffer a non-SSE response whole and return it unchanged (gzip and
    other encodings pass through untouched); usage is extracted from a
    decompressed copy. When ``converting``, the buffered body is converted
    from the model's pinned protocol into the client's format first."""
    if upstream_resp.is_stream_consumed:
        # Pre-buffered upstream (e.g. a caller-provided client whose
        # transport already read the body): content is cached.
        content = upstream_resp.content
    else:
        content = b"".join([chunk async for chunk in upstream_resp.aiter_raw()])
    await upstream_resp.aclose()
    capture.record_buffered(content)
    if convert_stats.need_convert():
        try:
            content = convert_response_body(content, convert_stats)
        except ConversionFailed as exc:
            return Response(
                content=json.dumps({"error": {"message": str(exc)}}).encode(),
                status_code=502,
                headers={"content-type": "application/json"},
            )
    return Response(
        content=content,
        status_code=upstream_resp.status_code,
        headers=resp_headers,
    )


working_model: str | None = None


def build_app(
    config: GatewayConfig,
    client: httpx.AsyncClient | None = None,
    config_path: str | Path | None = None,
    stats_db_path: str | Path | None = None,
    watcher_cache_path: str | Path | None = None,
    watcher_interval: float = 12 * 3600,
    prompts_dir: str | Path | None = None,
) -> FastAPI:

    logger.setLevel(logging.INFO)
    if not logger.handlers:
        logger.addHandler(logging.StreamHandler())
    if config.debug_log:
        logger.setLevel(logging.DEBUG)

    store = ConfigStore(config, config_path=config_path)
    db_path: Path | None = None
    if stats_db_path is not None:
        db_path = Path(stats_db_path)
    elif config_path is not None:
        db_path = Path(config_path).parent / "usage.db"
    stats_db: StatsDb | None = None
    if db_path is not None:
        try:
            stats_db = StatsDb(db_path)
        except Exception as e:
            raise StoreError(f"failed to open stats database at {db_path}; usage stats disabled") from e

    def _pricing_resolver(model_name: str) -> Pricing | None:
        for ep in store.config.endpoints:
            if ep.model == model_name:
                return resolve_pricing(ep.pricing)
        return None

    cost_worker = CostWorker(stats_db, _pricing_resolver) if stats_db is not None else None
    own_client = client is None
    http_client = (
        client
        if client is not None
        else httpx.AsyncClient(timeout=UPSTREAM_TIMEOUT)
    )
    proxied_clients: dict[str, httpx.AsyncClient] = {}

    def upstream_client_for(socks5_proxy: str | None) -> httpx.AsyncClient:
        if not socks5_proxy:
            return http_client
        proxied = proxied_clients.get(socks5_proxy)
        if proxied is None:
            proxied = httpx.AsyncClient(
                proxy=socks5_proxy,
                timeout=UPSTREAM_TIMEOUT,
            )
            proxied_clients[socks5_proxy] = proxied
        return proxied

    # Read-only mode (no config_path) never starts the watcher: it could not
    # persist models_news anyway. The baseline cache path is resolved here —
    # once — so the watcher and the admin fetch-models route always share the
    # same file (passing None from __main__ must not silently split them).
    if watcher_cache_path is not None:
        watcher_cache: Path | None = Path(watcher_cache_path)
    elif config_path is not None:
        watcher_cache = default_cache_path()
    else:
        watcher_cache = None
    model_watcher = (
        ModelWatcher(
            store,
            http_client,
            interval=watcher_interval,
            cache_path=watcher_cache,
            http_client_for=upstream_client_for,
        )
        if config_path is not None else None
    )

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if cost_worker is not None:
            cost_worker.start()
        if model_watcher is not None:
            model_watcher.start()
        yield
        if model_watcher is not None:
            model_watcher.stop()
        if cost_worker is not None:
            cost_worker.stop(timeout=2.0)
        if stats_db is not None:
            stats_db.close()
        for proxied in proxied_clients.values():
            await proxied.aclose()
        if own_client:
            await http_client.aclose()

    app = FastAPI(title="llm-gateway", docs_url=None, redoc_url=None, lifespan=lifespan)
    app.state.stats_db = stats_db
    app.state.cost_worker = cost_worker
    app.state.model_watcher = model_watcher
    app.state.watcher_cache_path = watcher_cache
    # Template body reads are lazy, so a missing/broken prompt store never
    # fails boot. Task 6 (proxy path) reuses this same object.
    prompt_store = PromptTemplateStore(
        prompts_dir if prompts_dir is not None else DEFAULT_PROMPTS_DIR
    )
    app.state.prompt_templates = prompt_store

    def error(status: int, message: str) -> Response:
        return Response(
            content=json.dumps({"error": {"message": message}}),
            status_code=status,
            media_type="application/json",
        )

    # Must be registered before the catch-all route so it is matched first.
    @app.api_route("/v1/models", methods=["GET", "HEAD"])
    async def models_list() -> Response:
        return Response(
            content=json.dumps(
                {"data": [{"id": name, "type": "model"} for name in store.router.list_models()]}
            ),
            media_type="application/json",
        )

    register_admin_routes(app, store, http_client=http_client,
                          http_client_for=upstream_client_for,
                          watcher_cache_path=watcher_cache,
                          model_watcher=model_watcher,
                          prompt_store=prompt_store)
    register_stats_routes(app, stats_db)

    @app.api_route(
        "/{full_path:path}",
        methods=["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"],
    )
    async def proxy_request(request: Request) -> Response:
        global working_model

        # Only the three inference protocols are proxied. Reject everything
        # else before touching the body: Files/count_tokens/batch requests
        # carry multipart or binary payloads that json.loads would blow up on
        # (500), and on paths like /v1/files a *successful* parse would be
        # worse — conversion rewrote the path to /v1/responses and the body to
        # an empty `input` list, so the client got a bogus 200 instead of an
        # error. Decline honestly instead.
        #
        # Classification happens here because the body is unreadable for
        # unsupported paths; `source_format` is reused below so the path is
        # matched exactly once per request.
        path = request.url.path
        try:
            source_format = require_supported(path)
        except UnsupportedPath:
            return error(
                501,
                f"unsupported path '{path}': this gateway only proxies "
                "/v1/messages, /v1/responses , /v1/chat/completions "
                "and /v1/models",
            )

        raw = "{}"
        if request.method not in WITH_BODY_METHODS:
            if working_model is None:
                return error(400, "request body is Empty")
        else:
            raw = await request.body()

        # request body
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError):
            # UnicodeDecodeError: a non-UTF-8 (multipart/binary) body that
            # json.loads rejects before JSON parsing even starts.
            return error(400, "request body is not valid JSON")
        if not isinstance(data, dict):
            return error(400, "request body must be a JSON object")

        # request model with fallback
        model = data.get("model") or working_model
        if model is None or model == "":
            return error(
                400,
                "request body must contain a top-level 'model' string field",
            )
        working_model = model

        # mapping endpoint
        try:
            endpoint = store.router.route(model)
        except ModelNotFoundError:
            return error(404, f"model '{model}' is not configured")

        # request path
        if store.config.debug_log:
            qs = ("?" + request.url.query) if request.url.query else ""
            logger.debug("request: %s %s%s, model: %s",
                         request.method, path, qs, working_model)

        # final url
        convert_stats = get_convert_stats(path, endpoint.protocol, source_format)
        upstream_path_ = convert_stats.get_target_path()
        upstream_url = endpoint.base_url + upstream_path_
        if request.url.query:
            upstream_url += "?" + request.url.query

        # final headers
        origin_headers = {k.lower(): v for k, v in request.headers.items()}
        headers = _rewrite_headers(origin_headers, endpoint, store.config.api_key)

        # final body
        body = {}
        if len(data.keys()) > 0:
            try:
                body = protocol_convert(convert_stats, data)
            except ConversionFailed as exc:
                return error(400, str(exc))

            await _rewrite_body(app, endpoint, body, convert_stats, store.config.extra_body)
        if store.config.debug_log:
            _debug_log_request(origin_headers, headers, data, body)

        start_time = time.monotonic()
        try:
            upstream_client = upstream_client_for(endpoint.socks5_proxy)
            upstream_resp = await upstream_client.send(
                upstream_client.build_request(
                    request.method, upstream_url, headers=headers, content=json.dumps(body).encode(),
                ),
                stream=True,
            )
        except httpx.HTTPError as exc:
            logger.error(
                "upstream request failed for %s %s (model=%s): %s: %s",
                request.method, upstream_url, working_model,
                type(exc).__name__, exc,
            )
            return error(502, "failed to reach upstream")

        capture = _UsageCapture(
            stats_db=stats_db,
            endpoint=endpoint,
            start_time=start_time,
            status=upstream_resp.status_code,
            decomp=(_decompressor(upstream_resp.headers.get("content-encoding", ""))
                    if stats_db is not None else None),
            client_protocol=convert_stats.source.value,
            upstream_protocol=(convert_stats.target or convert_stats.source).value,
            reasoning_effort=_reasoning_effort(body),
        )

        resp_headers = {
            k: v for k, v in upstream_resp.headers.items()
            if k.lower() not in HOP_BY_HOP_HEADERS
        }
        if convert_stats.need_convert():
            resp_headers.pop("content-encoding", None)
            resp_headers.pop("content-length", None)

        if "text/event-stream" in upstream_resp.headers.get("content-type", ""):
            return _sse_response(upstream_resp, resp_headers, capture, convert_stats)
        return await _buffered_response(upstream_resp, resp_headers, capture, convert_stats)

    return app
