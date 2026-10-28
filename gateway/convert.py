"""Protocol matching + request/response body conversion for the proxy path.

Routing rule: a model with a pinned ``protocol`` is compared against the
request path's detected format. A match (or an unpinned model) proxies
directly. A mismatch converts the request body into the model's pinned
protocol, forwards it, then converts the upstream response (streaming or
not) back into the client's original format.

The conversion logic lives in the ``apicompat`` SDK (wire behavior pinned to
sub2api's apicompat by the golden parity suite shipped with that package);
this module only decides *when* to convert and glues conversion into the
httpx/SSE machinery.
"""

import json
import zlib
from typing import AsyncIterator

from apicompat import (
    Protocol,
    astream,
    convert_request,
    convert_response,
    encode_stream_terminator,
)
from gateway.config import detect_format
from dataclasses import dataclass

# Client-facing format (from detect_format) -> converter protocol.
FORMAT_TO_PROTOCOL = {
    "anthropic": Protocol.ANTHROPIC_MESSAGES,
    "openai": Protocol.OPENAI_CHAT,
    "responses": Protocol.OPENAI_RESPONSES,
}

#: The request path's detected format routes to the endpoint the upstream
#: expects for that protocol (the client-facing path cannot be used verbatim
#: when converting: e.g. an Anthropic request must be sent to the upstream's
#: /v1/responses, not /v1/messages).
_PROTOCOL_PATH = {
    Protocol.ANTHROPIC_MESSAGES: "/v1/messages",
    Protocol.OPENAI_RESPONSES: "/v1/responses",
    Protocol.OPENAI_CHAT: "/v1/chat/completions",
}


@dataclass
class ConvertStats:
    path: str = ""
    source: Protocol | None = None
    target: Protocol | None = None

    def need_convert(self):
        return self.target is not None and self.source != self.target

    def get_target_path(self):
        if not self.need_convert():
            return self.path
        return _PROTOCOL_PATH[self.target]

class ConversionFailed(Exception):
    """Request/response conversion failed (client-visible; do not proxy)."""


class UnsupportedPath(Exception):
    """The request path is not one of the three proxied protocol endpoints.

    Raised by :func:`require_supported` (and by :func:`get_convert_stats` when
    it has to detect the format itself) so callers cannot accidentally build a
    ConvertStats with an unknown source protocol — the old code fell back to
    OpenAI, which silently forwarded files/batch/count_tokens requests to the
    chat endpoint.
    """


def require_supported(path: str) -> str:
    """Return the path's source format, or raise :class:`UnsupportedPath`.

    The proxy calls this *before* reading the request body (an unsupported
    path may carry a multipart/binary payload), then hands the returned format
    to :func:`get_convert_stats` so the path is classified exactly once.
    """
    fmt = detect_format(path)
    if fmt is None:
        raise UnsupportedPath(path)
    return fmt


def get_convert_stats(path: str, pinned_protocol: str | None,
                      source_format: str | None = None) -> ConvertStats:
    """Build the conversion plan for one request.

    ``source_format`` is the value a previous :func:`require_supported` call
    returned; pass it to avoid detecting the path twice. When omitted the
    format is detected here (and an unsupported path raises).
    """
    fmt = require_supported(path) if source_format is None else source_format
    target = Protocol[pinned_protocol.upper()] if pinned_protocol is not None else None
    return ConvertStats(path, FORMAT_TO_PROTOCOL[fmt], target)


def convert_request_body(
    convert_stats: ConvertStats,
    data: dict,
) -> dict:
    """Convert a request body from the path's format into the model's pinned
    protocol, then rewrite the model name (upstream_model wins over the
    converter's passthrough of the client-facing name).

    ``extra_body`` (from the global config) is a per-protocol map:
    ``{"anthropic_messages": {...}, "openai_chat": {...}, "openai_responses": {...}}``.
    Only the sub-dict matching the pinned protocol is merged into the
    converted request — used to supply protocol-specific params the source
    format cannot express (e.g. Responses-only ``store``/``include``/``text``
    when converting from Anthropic).

    Raises ConversionFailed when conversion is impossible (e.g. the client
    sent no max_tokens but the target requires one).
    """

    if not convert_stats.need_convert():
        return data

    source = convert_stats.source
    target = convert_stats.target

    stream = bool(data.get("stream"))
    try:
        converted = convert_request(source, target, data, stream=stream)
    except Exception as exc:
        raise ConversionFailed(f"request conversion to {target} failed: {exc}") from exc

    return converted


def _decompress_content(content: bytes) -> bytes:
    """Decompress a gzip/deflate-encoded upstream body for conversion.

    Upstreams often gzip responses; the forwarded bytes stay untouched but
    the conversion layer needs the decoded JSON.
    """
    if content[:2] == b"\x1f\x8b":  # gzip magic
        return zlib.decompress(content, 16 + zlib.MAX_WBITS)
    if content[:2] == b"\x78\x9c":  # zlib/deflate magic
        return zlib.decompress(content)
    return content


def convert_response_body(
    content: bytes,
    convert_stats: ConvertStats
) -> bytes:
    """Convert a non-streaming upstream response body (in the model's pinned
    protocol) back into the client's original format. Accepts gzip/deflate
    compressed bodies."""
    try:
        data = json.loads(_decompress_content(content))
    except (json.JSONDecodeError, UnicodeDecodeError, zlib.error) as exc:
        raise ConversionFailed("upstream returned non-JSON for protocol conversion") from exc
    if not isinstance(data, dict):
        raise ConversionFailed("upstream returned a non-object response")
    # Upstream error responses (e.g. opencode's 403
    # {"model": ..., "error": {...}}) carry no output/choices to convert —
    # converting them as a normal body yields an empty assistant message and
    # the real cause (e.g. user_blocked) is lost. Surface the error in the
    # client protocol's error envelope instead. OpenAI-style targets already
    # speak this error shape, so only Anthropic needs a rewrite.
    if (isinstance(data.get("error"), dict)
            and "output" not in data and "choices" not in data):
        err = data["error"]
        if convert_stats.source == Protocol.ANTHROPIC_MESSAGES:
            return json.dumps({
                "type": "error",
                "error": {
                    "type": err.get("type", "api_error"),
                    "message": err.get("message", ""),
                },
            }).encode()
        return json.dumps(data).encode()
    try:
        converted = convert_response(convert_stats.target, convert_stats.source, data)
    except Exception as exc:
        raise ConversionFailed(f"response conversion from {convert_stats.target} failed: {exc}") from exc
    return json.dumps(converted).encode()


async def convert_sse_chunks(
    chunks: AsyncIterator[bytes],
    convert_stats: ConvertStats
) -> AsyncIterator[bytes]:
    """Convert an upstream SSE stream (in the model's pinned protocol) into
    the client's original format.

    Delegates to the SDK's :func:`apicompat.astream`, which buffers partial
    lines across chunks, re-emits converted events in small pieces so the
    client sees streaming start immediately, and synthesizes a termination
    sequence when the upstream stops without one (sub2api's Finalize pass).
    """

    async for out in astream(convert_stats.target, convert_stats.source, chunks):
        yield out

