"""
Protocol Converters Module

Provides functions for converting between different LLM API protocols.
Supports all 6 conversion directions for requests, responses, and streams.
"""

from typing import Any, Dict, List, Optional, Union

from .ir import IRRequest
from .protocol import Protocol

from .openai_chat import (
    OpenAIChatEncoder,
    OpenAIChatDecoder,
)
from .openai_responses import (
    OpenAIResponsesEncoder,
    OpenAIResponsesDecoder,
)
from .anthropic_messages import (
    AnthropicMessagesEncoder,
    AnthropicMessagesDecoder,
)
from .exceptions import (
    ConversionError,
    CapabilityNotSupportedError,
    ValidationError,
)


# Encoder/Decoder registry
_ENCODERS = {
    Protocol.OPENAI_CHAT: OpenAIChatEncoder(),
    Protocol.OPENAI_RESPONSES: OpenAIResponsesEncoder(),
    Protocol.ANTHROPIC_MESSAGES: AnthropicMessagesEncoder(),
}

# Decoder classes for creating fresh instances (needed for stateful streaming)
_DECODER_CLASSES = {
    Protocol.OPENAI_CHAT: OpenAIChatDecoder,
    Protocol.OPENAI_RESPONSES: OpenAIResponsesDecoder,
    Protocol.ANTHROPIC_MESSAGES: AnthropicMessagesDecoder,
}

# Encoder classes for creating fresh instances. Stream encoding is stateful
# (sequence numbers, item/usage accumulation — see OpenAIResponsesEncoder),
# so every stream conversion must build its own encoder; the singletons
# above are only for stateless request/response encoding.
_ENCODER_CLASSES = {
    Protocol.OPENAI_CHAT: OpenAIChatEncoder,
    Protocol.OPENAI_RESPONSES: OpenAIResponsesEncoder,
    Protocol.ANTHROPIC_MESSAGES: AnthropicMessagesEncoder,
}


def _get_protocol(protocol: Union[Protocol, str]) -> Protocol:
    """Convert string to Protocol enum if needed."""
    if isinstance(protocol, str):
        protocol_map = {
            "openai_chat": Protocol.OPENAI_CHAT,
            "openai_classic": Protocol.OPENAI_CHAT,
            "openai_responses": Protocol.OPENAI_RESPONSES,
            "anthropic_messages": Protocol.ANTHROPIC_MESSAGES,
            "anthropic": Protocol.ANTHROPIC_MESSAGES,
        }
        return protocol_map.get(protocol.lower(), Protocol(protocol))
    return protocol


# =============================================================================
# Generic Conversion Functions
# =============================================================================

def convert_request(
    source_protocol: Union[Protocol, str],
    target_protocol: Union[Protocol, str],
    payload: Dict[str, Any],
    *,
    stream: bool = False,
    options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Convert a request from source protocol to target protocol.

    Args:
        source_protocol: The source API protocol
        target_protocol: The target API protocol
        payload: The request payload in source format
        stream: Whether this is a streaming request
        options: Optional conversion options

    Returns:
        The converted request payload in target format

    Raises:
        ConversionError: If conversion fails
        CapabilityNotSupportedError: If target protocol doesn't support a required feature
    """
    source = _get_protocol(source_protocol)
    target = _get_protocol(target_protocol)
    # The source protocol drives a few encoder decisions that sub2api bakes
    # into per-direction bridges (e.g. Responses text.verbosity comes from
    # the Anthropic bridge only; unsigned thinking wraps in <thinking> tags
    # only on the Chat bridge).
    options = {**(options or {}), "source": source.value}

    # Decode to IR
    decoder = _DECODER_CLASSES[source]()
    ir_request = decoder.decode_request(payload)

    # The payload's own stream flag wins; the caller's flag can only turn
    # streaming on (it is derived from the same payload in the gateway).
    if stream:
        ir_request.stream = True
    _apply_default_parameters(ir_request, options)

    # Encode to target
    encoder = _ENCODERS[target]
    return encoder.encode_request(ir_request, options=options)


def _apply_default_parameters(
    ir_request: IRRequest, options: Dict[str, Any]
) -> None:
    default_params = options.get("default_parameters")
    if not isinstance(default_params, dict):
        return

    config = ir_request.generation_config
    mapping = {
        "temperature": "temperature",
        "top_p": "top_p",
        "top_k": "top_k",
        "max_tokens": "max_tokens",
    }
    for key, attr in mapping.items():
        if key not in default_params:
            continue
        if getattr(config, attr) is not None:
            continue
        setattr(config, attr, default_params[key])


def convert_response(
    source_protocol: Union[Protocol, str],
    target_protocol: Union[Protocol, str],
    payload: Dict[str, Any],
    *,
    options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Convert a response from source protocol to target protocol.

    Args:
        source_protocol: The source API protocol
        target_protocol: The target API protocol
        payload: The response payload in source format
        options: Optional conversion options

    Returns:
        The converted response payload in target format

    Raises:
        ConversionError: If conversion fails
    """
    source = _get_protocol(source_protocol)
    target = _get_protocol(target_protocol)
    options = {**(options or {}), "source": source.value}

    # Decode to IR
    decoder = _DECODER_CLASSES[source]()
    ir_response = decoder.decode_response(payload)

    # Client-facing model override (sub2api's converter `model` parameter):
    # the client's model name wins over the upstream response's.
    if options.get("model"):
        ir_response.model = options["model"]

    # Encode to target
    encoder = _ENCODERS[target]
    return encoder.encode_response(ir_response, options=options)


__all__ = [
    # Generic converters
    "convert_request",
    "convert_response",
    # Encoder/Decoder classes
    "OpenAIChatEncoder",
    "OpenAIChatDecoder",
    "OpenAIResponsesEncoder",
    "OpenAIResponsesDecoder",
    "AnthropicMessagesEncoder",
    "AnthropicMessagesDecoder",
    # Exceptions
    "ConversionError",
    "CapabilityNotSupportedError",
    "ValidationError",
]
