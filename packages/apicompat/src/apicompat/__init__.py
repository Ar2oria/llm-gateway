"""apicompat — convert LLM API payloads between OpenAI Chat, OpenAI Responses
and Anthropic Messages.

The three protocols are translated through a shared intermediate
representation, so every direction is supported by construction::

    from apicompat import convert_request, convert_response, convert_stream

    # request: client speaks Anthropic, upstream speaks Responses
    upstream_body = convert_request("anthropic_messages", "openai_responses", body)

    # response (non-streaming): convert the upstream body back
    client_body = convert_response("openai_responses", "anthropic_messages", upstream)

    # response (streaming): bytes in, bytes out
    for chunk in convert_stream("openai_responses", "anthropic_messages", upstream_chunks):
        write(chunk)

Async callers use :func:`astream`; anything that needs finer control can
drive :class:`StreamCodec` directly.

Protocol names are the strings ``"openai_chat"``, ``"openai_responses"`` and
``"anthropic_messages"`` (the :class:`Protocol` enum is accepted too).

Wire behaviour is pinned to sub2api's ``apicompat`` package by the golden
parity suite in ``tests/parity``.
"""

from __future__ import annotations

from .converters import convert_request, convert_response
from .exceptions import (
    CapabilityNotSupportedError,
    ConversionError,
    ValidationError,
)
from .ir import (
    IRContentBlock,
    IRMessage,
    IRRequest,
    IRResponse,
    IRToolDeclaration,
    IRUsage,
)
from .protocol import Protocol
from .stream import (
    StreamCodec,
    astream,
    convert_stream,
    encode_stream_terminator,
)

__version__ = "0.1.0"

__all__ = [
    # Conversion — requests and non-streaming responses
    "convert_request",
    "convert_response",
    # Conversion — streaming
    "convert_stream",
    "astream",
    "StreamCodec",
    "encode_stream_terminator",
    # Types
    "Protocol",
    "IRRequest",
    "IRResponse",
    "IRMessage",
    "IRContentBlock",
    "IRToolDeclaration",
    "IRUsage",
    # Errors
    "ConversionError",
    "CapabilityNotSupportedError",
    "ValidationError",
]
