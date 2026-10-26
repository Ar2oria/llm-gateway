"""
OpenAI Responses API Encoder/Decoder

Converts between OpenAI Responses API format and the Intermediate
Representation.

The wire behavior mirrors sub2api's ``internal/pkg/apicompat`` Anthropic ↔
Responses bridges (conflicts between reference projects resolve in
sub2api's favor):

- Requests: system → developer message items, ``store: false``,
  ``include: ["reasoning.encrypted_content"]``, ``parallel_tool_calls:
  true``, ``text: {verbosity: "medium"}``, ``reasoning`` always emitted,
  ``max_output_tokens`` floored at 128, sampling params dropped for
  reasoning models, tool_choice in its string forms.
- Responses: reasoning items → thinking blocks carrying
  ``encrypted_content`` as ``signature``; usage keeps all four Anthropic
  buckets (``input_tokens`` excludes cache reads AND cache creation).
- Streams: stateful converters that maintain block pairing, emit
  ``signature_delta`` before closing thinking blocks, recover text that
  only arrived on done/terminal payloads, and can finalize a truncated
  stream into a clean termination sequence.
"""

import base64
import json
import re
import secrets
import time
from typing import Any, Dict, List, Optional, Tuple, Union

from .ir import (
    IRContentBlock,
    IRDocumentBlock,
    IRGenerationConfig,
    IRImageBlock,
    IRMessage,
    IRRequest,
    IRResponse,
    IRResponseFormat,
    IRServerToolUseBlock,
    IRStreamEvent,
    IRTextBlock,
    IRThinkingBlock,
    IRThinkingConfig,
    IRToolChoice,
    IRToolDeclaration,
    IRToolResultBlock,
    IRToolUseBlock,
    IRUsage,
    IRWebSearchToolResultBlock,
    ImageSourceType,
    Role,
    StopReason,
    StreamEventType,
    ToolChoiceType,
)

#: Prefix of the opaque envelope that carries a signed Anthropic thinking
#: block inside a Responses ``reasoning.encrypted_content`` field (sub2api's
#: bridge format — only emitted/consumed for Claude 5.5-family models).
ANTHROPIC_THINKING_ENVELOPE_PREFIX = "anthropic-thinking-v1:"

#: Floor for ``max_output_tokens`` — very small values cause upstream API
#: errors (sub2api's DefaultMinMaxOutputTokens).
MIN_MAX_OUTPUT_TOKENS = 128

#: Schema name emitted when the source protocol carried none. Responses
#: requires the field on every ``json_schema`` format; Anthropic's
#: ``output_format`` has no name slot, so a converted request always lands
#: here.
DEFAULT_SCHEMA_NAME = "structured_output"

#: Default ``max_tokens`` when the source protocol does not carry one
#: (Anthropic requires the field).

_JSON_SEPARATORS = (",", ":")


def _dump_compact(obj: Any) -> str:
    """Serialize without inter-token whitespace (matches Go's json.Marshal)."""
    return json.dumps(obj, separators=_JSON_SEPARATORS, ensure_ascii=False)


def _is_reasoning_model(model: str) -> bool:
    """Reasoning models reject sampling params via the Responses API.

    ``gpt-N`` with N >= 5 and the gpt-6 sol/luna spellings are
    reasoning-only (sub2api's isReasoningModel).
    """
    lowered = (model or "").strip().lower()
    if not lowered.startswith("gpt-"):
        return False
    for base in ("gpt-6-sol", "gpt-6-luna"):
        if lowered == base or lowered.startswith(base + "-"):
            return True
    rest = lowered[4:]
    digits = ""
    for ch in rest:
        if not ch.isdigit():
            break
        digits += ch
    if not digits:
        return False
    try:
        return int(digits) >= 5
    except ValueError:
        return False


def _is_claude_55(model: str) -> bool:
    """Claude 5.5 family — the only models that round-trip thinking
    signatures through the envelope (sub2api's IsOpus55/IsSonnet55/...)."""
    return bool(
        re.search(r"claude-(opus|sonnet|haiku)-5-5", (model or "").lower())
    )


def _from_responses_call_id_response(call_id: str) -> str:
    """Responses → Anthropic (response side): strip a legacy ``fc_`` wrapper
    when it hides a native Anthropic id; other ids pass through verbatim."""
    if call_id.startswith("fc_"):
        rest = call_id[3:]
        if rest.startswith("toolu_") or rest.startswith("call_"):
            return rest
    return call_id


def _from_responses_call_id_request(call_id: str) -> str:
    """Responses → Anthropic (request side): same unwrapping, plus a
    synthetic ``toolu_`` prefix for ids that are not Anthropic-shaped
    (sub2api's fromResponsesCallIDToAnthropic)."""
    unwrapped = _from_responses_call_id_response(call_id)
    if not unwrapped.startswith("toolu_") and not unwrapped.startswith("call_"):
        return "toolu_" + unwrapped
    return unwrapped


def _normalize_tool_parameters(schema: Any) -> Dict[str, Any]:
    """Responses requires ``properties`` on object tool schemas."""
    if not isinstance(schema, dict) or not schema:
        return {"type": "object", "properties": {}}
    cleaned = dict(schema)
    if cleaned.get("type") != "object":
        return cleaned
    if "properties" not in cleaned:
        cleaned["properties"] = {}
    return cleaned


def _parse_data_url(data_url: str) -> Optional[Tuple[str, str]]:
    """Parse ``data:<media>;base64,<data>`` into (media_type, data)."""
    match = re.match(r"data:([^;]+);base64,(.+)", data_url or "", re.DOTALL)
    if not match:
        return None
    return match.group(1), match.group(2)


def _data_url_to_image_block(data_url: str) -> Optional[IRImageBlock]:
    parsed = _parse_data_url(data_url)
    if parsed is None:
        return None
    return IRImageBlock(
        source_type=ImageSourceType.BASE64,
        base64_data=parsed[1],
        media_type=parsed[0],
    )


def _image_block_to_data_uri(block: IRImageBlock) -> Optional[str]:
    """base64 images become data URIs; URL sources are not convertible
    (sub2api drops them)."""
    if block.source_type != ImageSourceType.BASE64 or not block.base64_data:
        return None
    media_type = block.media_type or "image/png"
    return f"data:{media_type};base64,{block.base64_data}"


def _seed_tool_arguments(raw: Any) -> str:
    """Inline tool_use input on content_block_start → streaming seed.

    Empty/absent/no-argument payloads return "" so the "{}" fallback
    applies and no empty delta is synthesized.
    """
    if isinstance(raw, dict):
        raw = _dump_compact(raw)
    trimmed = str(raw or "").strip()
    if trimmed in ("", "{}", "null"):
        return ""
    return trimmed


def _sanitize_read_tool_input(name: str, raw: str) -> str:
    """Drop the ``pages: ""`` quirk Claude Code sends on Read tool calls."""
    if name != "Read" or not raw:
        return raw
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        return raw
    if not isinstance(obj, dict) or obj.get("pages") != "":
        return raw
    obj.pop("pages", None)
    return _dump_compact(obj)


def _encode_thinking_envelope(block: IRThinkingBlock) -> str:
    """Wrap a signed thinking block in the opaque bridge envelope."""
    payload: Dict[str, Any] = {"type": "thinking", "thinking": block.thinking}
    if block.signature:
        payload["signature"] = block.signature
    if block.is_redacted and block.redacted_data:
        payload = {
            "type": "redacted_thinking",
            "data": block.redacted_data,
        }
    encoded = base64.b64encode(
        _dump_compact(payload).encode("utf-8")
    ).decode("ascii")
    return ANTHROPIC_THINKING_ENVELOPE_PREFIX + encoded.rstrip("=")


def _decode_thinking_envelope(encrypted: str) -> Optional[IRThinkingBlock]:
    """Reverse of :func:`_encode_thinking_envelope`; None when invalid."""
    if not encrypted.startswith(ANTHROPIC_THINKING_ENVELOPE_PREFIX):
        return None
    raw = encrypted[len(ANTHROPIC_THINKING_ENVELOPE_PREFIX):]
    padding = "=" * (-len(raw) % 4)
    try:
        decoded = base64.b64decode(raw + padding)
        block = json.loads(decoded)
    except (ValueError, json.JSONDecodeError):
        return None
    if not isinstance(block, dict):
        return None
    if block.get("type") == "thinking" and block.get("signature"):
        return IRThinkingBlock(
            thinking=block.get("thinking", ""),
            signature=block.get("signature"),
        )
    if block.get("type") == "redacted_thinking" and block.get("data"):
        return IRThinkingBlock(
            is_redacted=True,
            redacted_data=block.get("data"),
        )
    return None


def _usage_from_responses(usage: Dict[str, Any]) -> IRUsage:
    """Convert Responses usage into Anthropic bucket semantics.

    Responses ``input_tokens`` is the total (cached included); Anthropic's
    ``input_tokens`` excludes cache reads AND cache creation.
    """
    input_details = usage.get("input_tokens_details") or {}
    cached = (
        input_details.get("cached_tokens") or 0
        if isinstance(input_details, dict)
        else 0
    )
    cache_creation = usage.get("cache_creation_input_tokens") or 0
    input_tokens = max(0, usage.get("input_tokens", 0) - cached - cache_creation)
    ir_usage = IRUsage(
        input_tokens=input_tokens,
        output_tokens=usage.get("output_tokens", 0),
        cache_read_tokens=cached,
        cache_creation_tokens=cache_creation,
    )
    output_details = usage.get("output_tokens_details") or {}
    if isinstance(output_details, dict):
        ir_usage.reasoning_tokens = output_details.get("reasoning_tokens", 0)
    return ir_usage


def _usage_to_responses(usage: IRUsage) -> Dict[str, Any]:
    """Convert Anthropic bucket semantics back into Responses usage
    (``input_tokens`` re-includes cache reads and cache creation)."""
    total_input = usage.input_tokens + usage.cache_read_tokens + usage.cache_creation_tokens
    out: Dict[str, Any] = {
        "input_tokens": total_input,
        "output_tokens": usage.output_tokens,
        "total_tokens": total_input + usage.output_tokens,
    }
    if usage.cache_creation_tokens:
        out["cache_creation_input_tokens"] = usage.cache_creation_tokens
    if usage.cache_read_tokens > 0:
        out["input_tokens_details"] = {"cached_tokens": usage.cache_read_tokens}
    return out


class OpenAIResponsesDecoder:
    """Decodes OpenAI Responses API format to IR."""

    def __init__(self) -> None:
        self._reset_stream_state()

    # ------------------------------------------------------------------
    # Request decoding (Responses request → IR)
    # ------------------------------------------------------------------

    def decode_request(self, payload: Dict[str, Any]) -> IRRequest:
        """Decode an OpenAI Responses request to IR."""
        ir = IRRequest(
            model=payload.get("model", ""),
            stream=payload.get("stream", False),
        )
        # Remember the model for the reasoning-envelope gate (Claude 5.5
        # family only, sub2api rule).
        self._current_model = ir.model

        # instructions + developer/system items → system parts (joined
        # "\n\n" for ``system``, kept separately in ``system_parts``).
        system_parts: List[str] = []
        instructions = payload.get("instructions")
        if isinstance(instructions, str) and instructions.strip():
            system_parts.append(instructions.strip())

        input_data = payload.get("input", "")
        if isinstance(input_data, str):
            ir.messages = [
                IRMessage(
                    role=Role.USER,
                    content=[IRTextBlock(text=input_data)],
                    plain_text=True,
                )
            ]
        else:
            ir.messages = self._decode_input(input_data, system_parts)

        if system_parts:
            ir.system = "\n\n".join(system_parts)
            ir.system_parts = system_parts

        # Decode generation config
        ir.generation_config = self._decode_generation_config(payload)

        # Decode tools
        if "tools" in payload:
            ir.tools = self._decode_tools(payload["tools"])

        # Decode tool choice
        if "tool_choice" in payload:
            ir.tool_choice = self._decode_tool_choice(payload["tool_choice"])

        # Decode response format (text.format)
        if "text" in payload and isinstance(payload["text"], dict):
            if "format" in payload["text"]:
                ir.response_format = self._decode_response_format(
                    payload["text"]["format"]
                )

        # Reasoning effort → IR thinking config (verbatim; the Anthropic
        # encoder maps xhigh → max).
        reasoning = payload.get("reasoning")
        if isinstance(reasoning, dict) and reasoning.get("effort"):
            ir.thinking = IRThinkingConfig(effort=str(reasoning["effort"]))

        return ir

    def _decode_input(
        self, input_data: List[Dict[str, Any]], system_parts: List[str]
    ) -> List[IRMessage]:
        """Decode the Responses input item array to IR messages.

        Mirrors sub2api's convertResponsesInputToAnthropic: role checks come
        first (system/developer → system parts), then typed items
        (function_call / function_call_output / reasoning), then user /
        assistant messages, then a whitelist conversion for anything else.
        """
        messages: List[IRMessage] = []

        for item in input_data:
            if not isinstance(item, dict):
                continue
            role = item.get("role", "")
            item_type = item.get("type", "message")

            if role in ("system", "developer"):
                # Position-preserving SYSTEM message (role fidelity in
                # ``name``): each encoder applies its target protocol's
                # semantics — Anthropic folds all of them into the top-level
                # system, Chat merges leading / downgrades mid-conversation,
                # Responses keeps the role on the item.
                text = self._text_from_content(item.get("content"))
                if text:
                    messages.append(
                        IRMessage(
                            role=Role.SYSTEM,
                            name=role,
                            content=[IRTextBlock(text=text)],
                        )
                    )
                continue

            if item_type == "function_call":
                messages.append(
                    IRMessage(
                        role=Role.ASSISTANT,
                        content=[
                            IRToolUseBlock(
                                id=_from_responses_call_id_request(
                                    item.get("call_id", "")
                                ),
                                name=item.get("name", ""),
                                input=self._parse_json_safely(
                                    item.get("arguments") or "{}"
                                ),
                            )
                        ],
                    )
                )
                continue

            if item_type == "function_call_output":
                messages.append(
                    IRMessage(
                        role=Role.TOOL,
                        content=[
                            IRToolResultBlock(
                                tool_use_id=_from_responses_call_id_request(
                                    item.get("call_id", "")
                                ),
                                content=self._function_output_content(item),
                            )
                        ],
                    )
                )
                continue

            if item_type == "reasoning":
                # Marked Anthropic bridge envelopes replay only for Claude
                # 5.5-family models (sub2api rule); plain summary text
                # always survives as a thinking block so the Chat bridge can
                # replay it as reasoning_content.
                block = None
                encrypted = item.get("encrypted_content", "")
                if _is_claude_55(self._current_model or "") and encrypted:
                    block = _decode_thinking_envelope(str(encrypted))
                if block is None:
                    summary_text = ""
                    for s in item.get("summary", []) or []:
                        if isinstance(s, dict) and s.get("type") == "summary_text":
                            summary_text += s.get("text", "")
                    if summary_text:
                        block = IRThinkingBlock(thinking=summary_text)
                if block is not None:
                    messages.append(
                        IRMessage(role=Role.ASSISTANT, content=[block])
                    )
                continue

            if role == "user":
                raw_content = item.get("content")
                messages.append(
                    IRMessage(
                        role=Role.USER,
                        content=self._user_content(raw_content),
                        plain_text=isinstance(raw_content, str),
                    )
                )
                continue

            if role == "assistant":
                raw_content = item.get("content")
                messages.append(
                    IRMessage(
                        role=Role.ASSISTANT,
                        content=self._assistant_content(raw_content),
                        plain_text=isinstance(raw_content, str),
                    )
                )
                continue

            # Unknown role/type — whitelist-convert as user content.
            if item.get("content") is None:
                continue
            messages.append(
                IRMessage(
                    role=Role.USER,
                    content=self._user_content(item.get("content")),
                )
            )

        return messages

    def _text_from_content(self, content: Any) -> str:
        """Extract text from a string-or-parts content field."""
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            texts = []
            for part in content:
                if not isinstance(part, dict):
                    continue
                if part.get("type") in ("input_text", "output_text", "text"):
                    if part.get("text"):
                        texts.append(part["text"])
            return "\n\n".join(texts)
        return ""

    def _user_content(self, content: Any) -> List[IRContentBlock]:
        """Responses user content → IR blocks (whitelist: text, data-URI
        images, data-URI files)."""
        if content is None:
            return []
        if isinstance(content, str):
            return [IRTextBlock(text=content)]
        blocks: List[IRContentBlock] = []
        if not isinstance(content, list):
            return []
        for part in content:
            if isinstance(part, str):
                blocks.append(IRTextBlock(text=part))
                continue
            if not isinstance(part, dict):
                continue
            part_type = part.get("type", "")
            if part_type in ("input_text", "text"):
                if part.get("text"):
                    blocks.append(IRTextBlock(text=part["text"]))
            elif part_type == "input_image":
                image = _data_url_to_image_block(part.get("image_url", ""))
                if image is not None:
                    blocks.append(image)
            elif part_type == "input_file":
                parsed = _parse_data_url(part.get("file_data", ""))
                if parsed is not None:
                    blocks.append(
                        IRDocumentBlock(
                            source_type="base64",
                            data=parsed[1],
                            media_type=parsed[0],
                        )
                    )
        return blocks

    def _assistant_content(self, content: Any) -> List[IRContentBlock]:
        """Responses assistant content → IR text blocks."""
        if content is None:
            return []
        if isinstance(content, str):
            return [IRTextBlock(text=content)]
        blocks: List[IRContentBlock] = []
        if not isinstance(content, list):
            return []
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") in ("output_text", "text"):
                if part.get("text"):
                    blocks.append(IRTextBlock(text=part["text"]))
        return blocks

    def _function_output_content(self, item: Dict[str, Any]) -> Any:
        """function_call_output.output (string or parts) → tool_result
        content."""
        output = item.get("output")
        if isinstance(output, str):
            return output if output else "(empty)"
        if isinstance(output, list):
            blocks: List[IRContentBlock] = []
            for part in output:
                if not isinstance(part, dict):
                    continue
                if part.get("type") in ("input_text", "output_text", "text"):
                    if part.get("text"):
                        blocks.append(IRTextBlock(text=part["text"]))
                elif part.get("type") == "input_image":
                    image = _data_url_to_image_block(part.get("image_url", ""))
                    if image is not None:
                        blocks.append(image)
            if blocks:
                return blocks
            return "(empty)" if not output else ""
        return "(empty)"

    def _decode_generation_config(self, payload: Dict[str, Any]) -> IRGenerationConfig:
        """Decode generation configuration."""
        config = IRGenerationConfig()

        if "temperature" in payload:
            config.temperature = payload["temperature"]
        if "top_p" in payload:
            config.top_p = payload["top_p"]
        if "max_output_tokens" in payload:
            config.max_tokens = payload["max_output_tokens"]

        return config

    def _decode_tools(self, tools: List[Dict[str, Any]]) -> List[IRToolDeclaration]:
        """Decode tool declarations."""
        ir_tools: List[IRToolDeclaration] = []
        for tool in tools:
            tool_type = tool.get("type", "function")
            if tool_type in ("web_search", "google_search", "web_search_20250305"):
                ir_tools.append(
                    IRToolDeclaration(
                        name="web_search",
                        type="web_search_20250305",
                    )
                )
                continue
            ir_tools.append(
                IRToolDeclaration(
                    name=tool.get("name", ""),
                    description=tool.get("description"),
                    parameters=tool.get("parameters", {}),
                    strict=tool.get("strict", False),
                    type=None if tool_type == "function" else tool_type,
                )
            )
        return ir_tools

    def _decode_tool_choice(self, tool_choice: Any) -> IRToolChoice:
        """Decode tool choice (string or object forms)."""
        if isinstance(tool_choice, str):
            if tool_choice == "required":
                return IRToolChoice(type=ToolChoiceType.ANY)
            if tool_choice == "none":
                return IRToolChoice(type=ToolChoiceType.NONE)
            return IRToolChoice(type=ToolChoiceType.AUTO)

        if isinstance(tool_choice, dict):
            choice_type = tool_choice.get("type", "auto")
            if choice_type == "required":
                return IRToolChoice(type=ToolChoiceType.ANY)
            if choice_type == "none":
                return IRToolChoice(type=ToolChoiceType.NONE)
            if choice_type == "function":
                name = tool_choice.get("name") or ""
                if not name:
                    nested = tool_choice.get("function") or {}
                    name = nested.get("name", "")
                return IRToolChoice(
                    type=ToolChoiceType.SPECIFIC,
                    name=name,
                )
            if choice_type == "any":
                return IRToolChoice(type=ToolChoiceType.ANY)
            return IRToolChoice(type=ToolChoiceType.AUTO)

        return IRToolChoice(type=ToolChoiceType.AUTO)

    def _decode_response_format(self, fmt: Dict[str, Any]) -> IRResponseFormat:
        """Decode response format configuration."""
        fmt_type = fmt.get("type", "text")
        ir_format = IRResponseFormat(type=fmt_type)

        if fmt_type == "json_schema":
            ir_format.json_schema = fmt.get("schema")
            ir_format.schema_name = fmt.get("name")
            ir_format.strict = fmt.get("strict", False)

        return ir_format

    # ------------------------------------------------------------------
    # Response decoding (Responses response → IR)
    # ------------------------------------------------------------------

    def decode_response(self, payload: Dict[str, Any]) -> IRResponse:
        """Decode an OpenAI Responses response to IR."""
        ir = IRResponse(
            id=payload.get("id", ""),
            model=payload.get("model", ""),
            created=payload.get("created_at"),
        )

        # Decode output items
        output = payload.get("output", [])
        for item in output:
            if not isinstance(item, dict):
                continue
            ir.content.extend(self._decode_output_item(item))

        # Map status to stop reason: "incomplete" is only max_tokens when the
        # incomplete_details say so (sub2api rule). The content_filter
        # incomplete reason is preserved in metadata — the Anthropic
        # direction reports end_turn, the Chat direction content_filter.
        status = payload.get("status", "completed")
        details = payload.get("incomplete_details")
        if status == "incomplete":
            if isinstance(details, dict) and details.get("reason") == "max_output_tokens":
                ir.stop_reason = StopReason.MAX_TOKENS
            else:
                ir.stop_reason = StopReason.END_TURN
                if isinstance(details, dict) and details.get("reason") == "content_filter":
                    ir.metadata["incomplete_reason"] = "content_filter"
        elif status == "failed":
            ir.stop_reason = StopReason.ERROR
        else:
            ir.stop_reason = StopReason.END_TURN

        # Preserve the upstream cost (OpenAI Responses has it; Anthropic
        # responses also carry a "cost" field the encoder can emit).
        if payload.get("cost") is not None:
            ir.metadata["cost"] = payload["cost"]

        # Decode usage
        usage = payload.get("usage")
        if usage:
            ir.usage = _usage_from_responses(usage)

        return ir

    def _decode_output_item(self, item: Dict[str, Any]) -> List[IRContentBlock]:
        """Decode an output item to IR content blocks."""
        item_type = item.get("type", "message")
        blocks: List[IRContentBlock] = []

        if item_type == "message":
            content = item.get("content", [])
            for c in content:
                if isinstance(c, dict):
                    c_type = c.get("type", "text")
                    if c_type in ("text", "output_text"):
                        if c.get("text"):
                            blocks.append(IRTextBlock(text=c["text"]))
                    # Refusal content: surface the refusal text — Anthropic
                    # has no refusal block, so it becomes a plain text block;
                    # the Chat encoder routes flagged blocks to
                    # message.refusal instead.
                    elif c_type == "refusal":
                        if c.get("refusal"):
                            blocks.append(
                                IRTextBlock(text=c["refusal"], refusal=True)
                            )

        elif item_type == "function_call":
            arguments = item.get("arguments") or "{}"
            if item.get("name") == "Read":
                arguments = _sanitize_read_tool_input(item.get("name", ""), arguments)
            blocks.append(
                IRToolUseBlock(
                    id=_from_responses_call_id_response(item.get("call_id", "")),
                    name=item.get("name", ""),
                    input=self._parse_json_safely(arguments),
                )
            )

        elif item_type == "reasoning":
            # Summary text is the primary carrier; encrypted_content rides
            # along as the thinking signature so multi-turn clients can send
            # it back. Plaintext reasoning_text content (opencode quirk) is a
            # fallback when no summary was produced.
            summary_text = ""
            for s in item.get("summary", []) or []:
                if isinstance(s, dict) and s.get("type") == "summary_text":
                    summary_text += s.get("text", "")
            encrypted = str(item.get("encrypted_content") or "").strip()
            if summary_text or encrypted:
                blocks.append(
                    IRThinkingBlock(
                        thinking=summary_text,
                        signature=encrypted or None,
                    )
                )
            else:
                reasoning_texts = []
                content = item.get("content", [])
                if isinstance(content, list):
                    for c in content:
                        if (
                            isinstance(c, dict)
                            and c.get("type") == "reasoning_text"
                            and c.get("text")
                        ):
                            reasoning_texts.append(c["text"])
                if reasoning_texts:
                    blocks.append(
                        IRThinkingBlock(thinking="\n".join(reasoning_texts))
                    )

        elif item_type == "web_search_call":
            # Synthesize the server_tool_use + web_search_tool_result pair
            # so Claude Code can count the searches performed (sub2api).
            item_id = item.get("id", "")
            query = ""
            action = item.get("action")
            if isinstance(action, dict):
                query = action.get("query", "")
            tool_use_id = "srvtoolu_" + item_id
            blocks.append(
                IRServerToolUseBlock(
                    id=tool_use_id,
                    name="web_search",
                    input={"query": query},
                )
            )
            blocks.append(
                IRWebSearchToolResultBlock(
                    tool_use_id=tool_use_id,
                    content=[],
                )
            )

        return blocks

    # ------------------------------------------------------------------
    # Stream decoding (Responses SSE → IR stream events, stateful)
    # ------------------------------------------------------------------

    def _reset_stream_state(self) -> None:
        """Reset per-stream state (sub2api's ResponsesEventToAnthropicState)."""
        # Whether message_start / message_stop have been emitted.
        self._s_started: bool = False
        self._s_stopped: bool = False
        # Current content block index; increments when a block closes.
        self._s_block_index: int = 0
        self._s_block_open: bool = False
        self._s_block_type: Optional[str] = None  # text | thinking | tool_use
        # Responses output_index → Anthropic block index.
        self._s_output_to_block: Dict[int, int] = {}
        # Thinking signature captured from the reasoning item; emitted as a
        # signature_delta just before the block closes.
        self._s_pending_sig: str = ""
        self._s_has_tool_call: bool = False
        self._s_tool_args_seen: bool = False
        self._s_tool_args_buffer: str = ""
        self._s_tool_name: str = ""
        # Text already delivered per (output_index, content_index) part, for
        # reconciling done payloads against the deltas.
        self._s_text_by_part: Dict[Tuple[int, int], str] = {}
        self._s_text_delivered: bool = False
        # Terminal usage / stop bookkeeping.
        self._s_response_id: str = ""
        self._s_model: str = ""
        self._current_model: Optional[str] = None

    def decode_stream_event(
        self, event: Union[Dict[str, Any], str]
    ) -> List[IRStreamEvent]:
        """Decode a streaming event to IR events."""
        # Handle SSE text format
        if isinstance(event, str):
            event = self._parse_sse_event(event)
            if event is None:
                return []

        ir_events: List[IRStreamEvent] = []
        event_type = event.get("type", "")

        if event_type == "response.created":
            self._handle_created(event, ir_events)

        elif event_type == "response.output_item.added":
            self._handle_item_added(event, ir_events)

        elif event_type in ("response.output_text.delta", "response.text.delta"):
            self._handle_text_delta(event, ir_events)

        elif event_type in (
            "response.reasoning_summary_text.delta",
            "response.reasoning_text.delta",
        ):
            self._handle_reasoning_delta(event, ir_events)

        elif event_type in (
            "response.function_call_arguments.delta",
            "response.custom_tool_call_input.delta",
        ):
            self._handle_args_delta(event, ir_events)

        elif event_type == "response.function_call_arguments.done":
            self._handle_args_done(event, ir_events)

        elif event_type == "response.output_text.done":
            self._handle_text_done(event, ir_events)

        elif event_type == "response.output_item.done":
            self._handle_item_done(event, ir_events)

        elif event_type in (
            "response.done",
            "response.completed",
            "response.incomplete",
            "response.failed",
        ):
            self._handle_completed(event, ir_events)

        elif event_type == "error":
            error = event.get("error", {})
            ir_events.append(
                IRStreamEvent(
                    type=StreamEventType.ERROR,
                    error_type=error.get("type", "error"),
                    error_message=error.get("message", "Unknown error"),
                )
            )

        return ir_events

    def finalize(self) -> List[IRStreamEvent]:
        """Synthesize a clean termination for a stream that ended without a
        completion event (sub2api's FinalizeResponsesAnthropicStream)."""
        if not self._s_started or self._s_stopped:
            return []
        ir_events: List[IRStreamEvent] = []
        self._close_block(ir_events)
        stop_reason = (
            StopReason.TOOL_USE if self._s_has_tool_call else StopReason.END_TURN
        )
        ir_events.append(
            IRStreamEvent(type=StreamEventType.MESSAGE_DELTA, stop_reason=stop_reason)
        )
        ir_events.append(IRStreamEvent(type=StreamEventType.DONE))
        self._s_stopped = True
        return ir_events

    # --- stream event handlers -----------------------------------------

    def _handle_created(
        self, event: Dict[str, Any], ir_events: List[IRStreamEvent]
    ) -> None:
        response_data = event.get("response") or {}
        if response_data.get("id"):
            self._s_response_id = response_data["id"]
        if not self._s_model and response_data.get("model"):
            self._s_model = response_data["model"]
        if self._s_started:
            return
        self._s_started = True
        ir_events.append(
            IRStreamEvent(
                type=StreamEventType.MESSAGE_START,
                response=IRResponse(
                    id=self._s_response_id,
                    model=self._s_model,
                ),
            )
        )

    def _handle_item_added(
        self, event: Dict[str, Any], ir_events: List[IRStreamEvent]
    ) -> None:
        item = event.get("item")
        if not isinstance(item, dict):
            return
        item_type = item.get("type", "")
        output_index = event.get("output_index", 0)

        if item_type in ("function_call", "custom_tool_call"):
            self._close_block(ir_events)
            idx = self._s_block_index
            self._s_output_to_block[output_index] = idx
            self._s_block_open = True
            self._s_block_type = "tool_use"
            self._s_tool_name = item.get("name", "")
            self._s_tool_args_seen = False
            self._s_tool_args_buffer = ""
            self._s_has_tool_call = True
            ir_events.append(
                IRStreamEvent(
                    type=StreamEventType.CONTENT_BLOCK_START,
                    index=idx,
                    content_block=IRToolUseBlock(
                        id=_from_responses_call_id_response(item.get("call_id", "")),
                        name=item.get("name", ""),
                    ),
                )
            )

        elif item_type == "reasoning":
            self._close_block(ir_events)
            idx = self._s_block_index
            self._s_output_to_block[output_index] = idx
            self._s_block_open = True
            self._s_block_type = "thinking"
            self._s_pending_sig = str(item.get("encrypted_content") or "").strip()
            ir_events.append(
                IRStreamEvent(
                    type=StreamEventType.CONTENT_BLOCK_START,
                    index=idx,
                    content_block=IRThinkingBlock(),
                )
            )

        # message items open their text block lazily on the first delta.

    def _handle_text_delta(
        self, event: Dict[str, Any], ir_events: List[IRStreamEvent]
    ) -> None:
        text = event.get("delta", "")
        if not text:
            return
        self._emit_text(
            text,
            (event.get("output_index", 0), event.get("content_index", 0)),
            ir_events,
        )

    def _emit_text(
        self,
        text: str,
        part: Tuple[int, int],
        ir_events: List[IRStreamEvent],
    ) -> None:
        """Open a text block when needed, emit the delta, and record the
        delivered text for later reconciliation."""
        if not text:
            return
        if not self._s_block_open or self._s_block_type != "text":
            self._close_block(ir_events)
            idx = self._s_block_index
            self._s_block_open = True
            self._s_block_type = "text"
            ir_events.append(
                IRStreamEvent(
                    type=StreamEventType.CONTENT_BLOCK_START,
                    index=idx,
                    content_block=IRTextBlock(),
                )
            )
        self._s_text_by_part[part] = self._s_text_by_part.get(part, "") + text
        self._s_text_delivered = True
        ir_events.append(
            IRStreamEvent(
                type=StreamEventType.CONTENT_BLOCK_DELTA,
                index=self._s_block_index,
                delta_type="text",
                delta_text=text,
            )
        )

    def _recover_text(
        self,
        text: str,
        part: Tuple[int, int],
        ir_events: List[IRStreamEvent],
    ) -> None:
        """Emit the tail of a finished payload that never reached the client
        as deltas; never repeats what was already delivered."""
        known = part in self._s_text_by_part
        if not known and self._s_text_delivered:
            # Cannot establish which part this payload finishes; repeating an
            # answer is worse than leaving a partial one alone.
            return
        delivered = self._s_text_by_part.get(part, "") if known else ""
        if text == delivered or not text.startswith(delivered):
            return
        self._emit_text(text[len(delivered):], part, ir_events)

    def _handle_text_done(
        self, event: Dict[str, Any], ir_events: List[IRStreamEvent]
    ) -> None:
        if self._s_stopped:
            # The message is already terminated; a late payload cannot be
            # delivered without emitting a block after message_stop.
            self._close_block(ir_events)
            return
        self._recover_text(
            event.get("text", ""),
            (event.get("output_index", 0), event.get("content_index", 0)),
            ir_events,
        )
        self._close_block(ir_events)

    def _handle_reasoning_delta(
        self, event: Dict[str, Any], ir_events: List[IRStreamEvent]
    ) -> None:
        text = event.get("delta", "")
        if not text:
            return
        idx = self._s_output_to_block.get(event.get("output_index", 0))
        if idx is None:
            return
        ir_events.append(
            IRStreamEvent(
                type=StreamEventType.CONTENT_BLOCK_DELTA,
                index=idx,
                delta_type="thinking",
                delta_text=text,
            )
        )

    def _handle_args_delta(
        self, event: Dict[str, Any], ir_events: List[IRStreamEvent]
    ) -> None:
        delta = event.get("delta", "")
        if not delta:
            return
        if self._s_block_type == "tool_use" and self._s_tool_name == "Read":
            # Claude Code's Read quirk: buffer and emit only once the
            # accumulated arguments first become valid JSON.
            self._s_tool_args_buffer += delta
            if self._s_tool_args_seen or not _is_valid_json(self._s_tool_args_buffer):
                return
            idx = self._s_output_to_block.get(event.get("output_index", 0))
            if idx is None:
                return
            self._s_tool_args_seen = True
            sanitized = _sanitize_read_tool_input(
                self._s_tool_name, self._s_tool_args_buffer
            )
            ir_events.append(
                IRStreamEvent(
                    type=StreamEventType.CONTENT_BLOCK_DELTA,
                    index=idx,
                    delta_type="input_json",
                    delta_json=sanitized,
                )
            )
            return

        if self._s_block_type == "tool_use":
            self._s_tool_args_seen = True
        idx = self._s_output_to_block.get(event.get("output_index", 0))
        if idx is None:
            return
        ir_events.append(
            IRStreamEvent(
                type=StreamEventType.CONTENT_BLOCK_DELTA,
                index=idx,
                delta_type="input_json",
                delta_json=delta,
            )
        )

    def _handle_args_done(
        self, event: Dict[str, Any], ir_events: List[IRStreamEvent]
    ) -> None:
        if not self._s_block_open:
            return
        if self._s_block_type != "tool_use":
            self._close_block(ir_events)
            return

        raw = event.get("arguments", "")
        if raw == "":
            raw = self._s_tool_args_buffer
        if raw == "" or self._s_tool_args_seen:
            self._close_block(ir_events)
            return
        if self._s_tool_name == "Read":
            raw = _sanitize_read_tool_input(self._s_tool_name, raw)

        idx = self._s_output_to_block.get(event.get("output_index", 0))
        if idx is None:
            idx = self._s_block_index
        if idx != self._s_block_index:
            # Already streamed/closed via deltas.
            return
        ir_events.append(
            IRStreamEvent(
                type=StreamEventType.CONTENT_BLOCK_DELTA,
                index=idx,
                delta_type="input_json",
                delta_json=raw,
            )
        )
        self._close_block(ir_events)

    def _handle_item_done(
        self, event: Dict[str, Any], ir_events: List[IRStreamEvent]
    ) -> None:
        item = event.get("item")
        if not isinstance(item, dict):
            return

        if item.get("type") == "web_search_call" and item.get("status") == "completed":
            self._handle_web_search_done(item, ir_events)
            return

        # Capture the signature on the finished reasoning item — providers
        # often attach encrypted_content only here.
        if item.get("type") == "reasoning":
            sig = str(item.get("encrypted_content") or "").strip()
            if sig:
                self._s_pending_sig = sig

        self._close_block(ir_events)

    def _handle_web_search_done(
        self, item: Dict[str, Any], ir_events: List[IRStreamEvent]
    ) -> None:
        self._close_block(ir_events)
        tool_use_id = "srvtoolu_" + item.get("id", "")
        query = ""
        action = item.get("action")
        if isinstance(action, dict):
            query = action.get("query", "")

        idx1 = self._s_block_index
        ir_events.append(
            IRStreamEvent(
                type=StreamEventType.CONTENT_BLOCK_START,
                index=idx1,
                content_block=IRServerToolUseBlock(
                    id=tool_use_id,
                    name="web_search",
                    input={"query": query},
                ),
            )
        )
        ir_events.append(
            IRStreamEvent(type=StreamEventType.CONTENT_BLOCK_STOP, index=idx1)
        )
        self._s_block_index += 1

        idx2 = self._s_block_index
        ir_events.append(
            IRStreamEvent(
                type=StreamEventType.CONTENT_BLOCK_START,
                index=idx2,
                content_block=IRWebSearchToolResultBlock(
                    tool_use_id=tool_use_id,
                    content=[],
                ),
            )
        )
        ir_events.append(
            IRStreamEvent(type=StreamEventType.CONTENT_BLOCK_STOP, index=idx2)
        )
        self._s_block_index += 1

    def _handle_completed(
        self, event: Dict[str, Any], ir_events: List[IRStreamEvent]
    ) -> None:
        if self._s_stopped:
            return

        self._close_block(ir_events)

        response_data = event.get("response") or {}
        # Recover assistant text that only ever appeared in the terminal
        # payload (only when nothing was streamed). Refusals (muse
        # user_blocked) arrive the same way and must surface as text.
        refused = False
        if not self._s_text_delivered:
            for output_index, item in enumerate(
                response_data.get("output", []) or []
            ):
                if not isinstance(item, dict) or item.get("type") != "message":
                    continue
                for content_index, content in enumerate(
                    item.get("content", []) or []
                ):
                    if not isinstance(content, dict):
                        continue
                    if content.get("type") == "output_text" and content.get("text"):
                        self._emit_text(
                            content["text"],
                            (output_index, content_index),
                            ir_events,
                        )
                    elif content.get("type") == "refusal" and content.get("refusal"):
                        refused = True
                        self._emit_text(
                            content["refusal"],
                            (output_index, content_index),
                            ir_events,
                        )
            self._close_block(ir_events)

        usage_data = event.get("usage") or response_data.get("usage") or {}
        usage = _usage_from_responses(usage_data) if usage_data else None

        stop_reason = StopReason.END_TURN
        status = response_data.get("status", "completed")
        if status == "incomplete":
            details = response_data.get("incomplete_details")
            if (
                isinstance(details, dict)
                and details.get("reason") == "max_output_tokens"
            ):
                stop_reason = StopReason.MAX_TOKENS
        elif status == "completed" and self._s_has_tool_call:
            stop_reason = StopReason.TOOL_USE
        if refused:
            stop_reason = StopReason.CONTENT_FILTER

        ir_events.append(
            IRStreamEvent(
                type=StreamEventType.MESSAGE_DELTA,
                stop_reason=stop_reason,
                usage=usage,
            )
        )
        ir_events.append(IRStreamEvent(type=StreamEventType.DONE))
        self._s_stopped = True

    def _close_block(self, ir_events: List[IRStreamEvent]) -> None:
        """Close the open content block, emitting signature_delta first for
        thinking blocks."""
        if not self._s_block_open:
            return
        idx = self._s_block_index
        if self._s_block_type == "thinking" and self._s_pending_sig:
            ir_events.append(
                IRStreamEvent(
                    type=StreamEventType.CONTENT_BLOCK_DELTA,
                    index=idx,
                    delta_type="signature",
                    delta_text=self._s_pending_sig,
                )
            )
            self._s_pending_sig = ""
        ir_events.append(IRStreamEvent(type=StreamEventType.CONTENT_BLOCK_STOP, index=idx))
        self._s_block_open = False
        self._s_block_index += 1
        self._s_tool_name = ""
        self._s_tool_args_seen = False
        self._s_tool_args_buffer = ""

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _parse_sse_event(self, text: str) -> Optional[Dict[str, Any]]:
        """Parse SSE event text."""
        lines = text.strip().split("\n")
        event_type = None
        data = None

        for line in lines:
            if line.startswith("event: "):
                event_type = line[7:]
            elif line.startswith("data: "):
                try:
                    data = json.loads(line[6:])
                except json.JSONDecodeError:
                    pass

        if data and event_type:
            data["type"] = event_type
        return data

    def _parse_json_safely(self, json_str: str) -> Dict[str, Any]:
        """Safely parse JSON string."""
        try:
            return json.loads(json_str) if json_str else {}
        except json.JSONDecodeError:
            return {}

def _is_valid_json(text: str) -> bool:
    try:
        json.loads(text)
        return True
    except (json.JSONDecodeError, ValueError):
        return False


class OpenAIResponsesEncoder:
    """Encodes IR to OpenAI Responses API format.

    Stream encoding is stateful (sequence numbers, item accumulation) — a
    fresh instance must be used per stream.
    """

    def __init__(self) -> None:
        self._reset_stream_state()

    # ------------------------------------------------------------------
    # Request encoding (IR → Responses request)
    # ------------------------------------------------------------------

    def encode_request(
        self, ir: IRRequest, *, options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Encode IR request to OpenAI Responses format (sub2api's
        AnthropicToResponses semantics)."""
        options = options or {}

        # System → developer message item, one input_text part per block.
        system_parts = ir.system_parts or (
            [ir.system] if ir.system else []
        )
        system_parts = [
            part
            for part in system_parts
            if part and not part.startswith("x-anthropic-billing-header: ")
        ]

        items: List[Dict[str, Any]] = self._encode_messages(
            ir.messages, options=options
        )
        if system_parts:
            items = [
                {
                    "type": "message",
                    "role": "developer",
                    "content": [
                        {"type": "input_text", "text": part}
                        for part in system_parts
                    ],
                }
            ] + items

        payload: Dict[str, Any] = {"model": ir.model, "input": items}

        if ir.generation_config.max_tokens:
            payload["max_output_tokens"] = max(
                ir.generation_config.max_tokens, MIN_MAX_OUTPUT_TOKENS
            )

        # Reasoning models (gpt-5+) reject sampling params with a 400.
        if not _is_reasoning_model(ir.model):
            if ir.generation_config.temperature is not None:
                payload["temperature"] = ir.generation_config.temperature
            if ir.generation_config.top_p is not None:
                payload["top_p"] = ir.generation_config.top_p

        if ir.stream:
            payload["stream"] = True

        if ir.tools:
            payload["tools"] = self._encode_tools(ir.tools)

        payload["include"] = ["reasoning.encrypted_content"]

        store_false = False
        payload["store"] = store_false

        if options.get("source") == "anthropic_messages":
            payload["parallel_tool_calls"] = True

        # Reasoning effort — emitted only when the source carried one
        # (Anthropic sources always resolve an effort; Chat sources only
        # when reasoning_effort was set). Summary is enabled whenever the
        # model reasons.
        if ir.thinking and ir.thinking.effort:
            effort = ir.thinking.effort
            reasoning: Dict[str, Any] = {"effort": effort}
            if effort != "none":
                reasoning["summary"] = "auto"
            payload["reasoning"] = reasoning

        # Text: the Anthropic bridge pins verbosity "medium"; the Chat
        # bridge maps response_format into text.format (and emits nothing
        # otherwise). Both live on the same ``text`` object, so they merge:
        # assigning in exclusive branches dropped the structured-output
        # constraint for every Anthropic source.
        text_config: Dict[str, Any] = {}
        if options.get("source") == "anthropic_messages":
            text_config["verbosity"] = "medium"
        if ir.response_format and ir.response_format.type != "text":
            text_config["format"] = self._encode_response_format(
                ir.response_format
            )
        if text_config:
            payload["text"] = text_config

        if ir.tool_choice:
            payload["tool_choice"] = self._encode_tool_choice(ir.tool_choice)

        return payload

    def _encode_messages(
        self,
        messages: List[IRMessage],
        *,
        options: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Encode IR messages to the Responses input item array."""
        options = options or {}
        items: List[Dict[str, Any]] = []
        pending_images: List[Dict[str, Any]] = []

        for msg in messages:
            if msg.role == Role.TOOL:
                for block in msg.content:
                    if isinstance(block, IRToolResultBlock):
                        text, images = self._tool_result_output(block)
                        items.append(
                            {
                                "type": "function_call_output",
                                "call_id": block.tool_use_id,
                                "output": text,
                            }
                        )
                        pending_images.extend(images)
                continue

            if msg.role == Role.SYSTEM:
                # System/developer messages keep their role on the item
                # (sub2api chatSystemToResponses; ``name`` carries the
                # original role). Plain string content stays a string.
                texts = [
                    b.text for b in msg.content if isinstance(b, IRTextBlock) and b.text
                ]
                if texts:
                    if msg.plain_text and len(texts) == 1:
                        content: Any = texts[0]
                    else:
                        content = [
                            {"type": "input_text", "text": t} for t in texts
                        ]
                    items.append(
                        {
                            "type": "message",
                            "role": msg.name or "developer",
                            "content": content,
                        }
                    )
                continue

            if msg.role == Role.USER:
                parts: List[Dict[str, Any]] = []
                for block in msg.content:
                    if isinstance(block, IRTextBlock):
                        if block.text:
                            parts.append(
                                {"type": "input_text", "text": block.text}
                            )
                    elif isinstance(block, IRImageBlock):
                        uri = _image_block_to_data_uri(block)
                        if uri:
                            parts.append({"type": "input_image", "image_url": uri})
                had_pending_images = bool(pending_images)
                parts.extend(pending_images)
                pending_images = []
                if parts:
                    # Chat plain-string user content stays a string (strict
                    # upstreams reject array content without media).
                    if (
                        options.get("source") == "openai_chat"
                        and msg.plain_text
                        and not had_pending_images
                        and len(parts) == 1
                        and parts[0]["type"] == "input_text"
                    ):
                        items.append(
                            {
                                "type": "message",
                                "role": "user",
                                "content": parts[0]["text"],
                            }
                        )
                    else:
                        items.append(
                            {"type": "message", "role": "user", "content": parts}
                        )
                continue

            if msg.role == Role.ASSISTANT:
                # Reasoning items first (provider ciphertext only), then one
                # assistant text message, then function calls — the order
                # multi-turn caches expect. Unsigned thinking (Chat
                # reasoning_content) wraps in explicit tags on the Chat
                # bridge so the text survives (sub2api chatAssistantToResponses).
                for block in msg.content:
                    if isinstance(block, IRThinkingBlock) and not block.is_redacted:
                        sig = (block.signature or "").strip()
                        if sig and not sig.startswith("gAAAA"):
                            items.append(
                                {
                                    "type": "reasoning",
                                    "encrypted_content": sig,
                                }
                            )
                text = "\n\n".join(
                    b.text
                    for b in msg.content
                    if isinstance(b, IRTextBlock) and b.text
                )
                if options.get("source") == "openai_chat":
                    unsigned = "\n".join(
                        b.thinking
                        for b in msg.content
                        if isinstance(b, IRThinkingBlock)
                        and not b.is_redacted
                        and b.thinking
                        and not (b.signature or "").strip()
                    )
                    if unsigned:
                        wrapped = f"<thinking>{unsigned}</thinking>"
                        text = (
                            wrapped + "\n" + text if text else wrapped
                        )
                if text:
                    items.append(
                        {
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": text}],
                        }
                    )
                for block in msg.content:
                    if isinstance(block, IRToolUseBlock):
                        arguments = (
                            _dump_compact(block.input) if block.input else "{}"
                        )
                        items.append(
                            {
                                "type": "function_call",
                                "call_id": block.id,
                                "name": block.name,
                                "arguments": arguments,
                            }
                        )
                continue

        # Images extracted from tool_results that never found a user
        # message are still worth sending as their own user message.
        if pending_images:
            items.append(
                {"type": "message", "role": "user", "content": pending_images}
            )

        return items

    def _tool_result_output(
        self, block: IRToolResultBlock
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """tool_result content → (function_call_output text, image parts).

        The Responses output field only accepts strings; images ride along
        in the following user message.
        """
        if isinstance(block.content, str):
            return (block.content if block.content else "(empty)"), []
        if isinstance(block.content, list):
            texts: List[str] = []
            images: List[Dict[str, Any]] = []
            for b in block.content:
                if isinstance(b, IRTextBlock) and b.text:
                    texts.append(b.text)
                elif isinstance(b, IRImageBlock):
                    uri = _image_block_to_data_uri(b)
                    if uri:
                        images.append({"type": "input_image", "image_url": uri})
            text = "\n\n".join(texts)
            return (text if text else "(empty)"), images
        return "(empty)", []

    def _encode_tools(self, tools: List[IRToolDeclaration]) -> List[Dict[str, Any]]:
        """Encode tool declarations."""
        result: List[Dict[str, Any]] = []
        for tool in tools:
            if tool.type and tool.type.startswith("web_search"):
                result.append({"type": "web_search"})
                continue
            encoded: Dict[str, Any] = {
                "type": "function",
                "name": tool.name,
            }
            if tool.description:
                encoded["description"] = tool.description
            encoded["parameters"] = _normalize_tool_parameters(tool.parameters)
            encoded["strict"] = False
            result.append(encoded)
        return result

    def _encode_tool_choice(self, choice: IRToolChoice) -> Any:
        """Encode tool choice — string forms for auto/required/none."""
        if choice.type == ToolChoiceType.AUTO:
            return "auto"
        if choice.type == ToolChoiceType.ANY:
            return "required"
        if choice.type == ToolChoiceType.NONE:
            return "none"
        if choice.type == ToolChoiceType.SPECIFIC:
            return {"type": "function", "name": choice.name}
        return "auto"

    def _encode_response_format(self, fmt: IRResponseFormat) -> Dict[str, Any]:
        """Encode response format.

        Responses requires ``name`` on every ``json_schema`` format and
        rejects the request with a bare 400 when it is missing. Anthropic's
        ``output_format`` carries no name, so a converted request must supply
        a stable fallback rather than emitting a nameless schema.
        """
        if fmt.type == "json_schema":
            result: Dict[str, Any] = {
                "type": "json_schema",
                "name": fmt.schema_name or DEFAULT_SCHEMA_NAME,
            }
            if fmt.json_schema:
                result["schema"] = fmt.json_schema
            if fmt.strict:
                result["strict"] = True
            return result
        return {"type": fmt.type}

    # ------------------------------------------------------------------
    # Response encoding (IR → Responses response)
    # ------------------------------------------------------------------

    def encode_response(
        self, ir: IRResponse, *, options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Encode IR response to OpenAI Responses format (sub2api's
        AnthropicToResponsesResponse semantics)."""
        is55 = _is_claude_55(ir.model)

        outputs: List[Dict[str, Any]] = []
        msg_parts: List[Dict[str, Any]] = []

        for block in ir.content:
            if isinstance(block, IRThinkingBlock):
                if is55 and (block.signature or block.redacted_data):
                    item: Dict[str, Any] = {
                        "type": "reasoning",
                        "id": _generate_item_id(),
                        "encrypted_content": _encode_thinking_envelope(block),
                    }
                    if block.thinking:
                        item["summary"] = [
                            {"type": "summary_text", "text": block.thinking}
                        ]
                    outputs.append(item)
                elif block.thinking:
                    outputs.append(
                        {
                            "type": "reasoning",
                            "id": _generate_item_id(),
                            "summary": [
                                {"type": "summary_text", "text": block.thinking}
                            ],
                        }
                    )

            elif isinstance(block, IRTextBlock):
                if is55 and block.text:
                    outputs.append(
                        {
                            "type": "message",
                            "id": _generate_item_id(),
                            "role": "assistant",
                            "status": "completed",
                            "content": [
                                {"type": "output_text", "text": block.text}
                            ],
                        }
                    )
                elif block.text:
                    msg_parts.append({"type": "output_text", "text": block.text})

            elif isinstance(block, IRToolUseBlock):
                arguments = _dump_compact(block.input) if block.input else "{}"
                outputs.append(
                    {
                        "type": "function_call",
                        "id": _generate_item_id(),
                        "call_id": block.id,
                        "name": block.name,
                        "arguments": arguments,
                        "status": "completed",
                    }
                )

        if msg_parts:
            outputs.append(
                {
                    "type": "message",
                    "id": _generate_item_id(),
                    "role": "assistant",
                    "content": msg_parts,
                    "status": "completed",
                }
            )

        if not outputs:
            # Struct-tag semantics: an empty text part omits the text key.
            outputs.append(
                {
                    "type": "message",
                    "id": _generate_item_id(),
                    "role": "assistant",
                    "content": [{"type": "output_text"}],
                    "status": "completed",
                }
            )

        status = "incomplete" if ir.stop_reason == StopReason.MAX_TOKENS else "completed"
        response: Dict[str, Any] = {
            "id": ir.id if ir.id else _generate_response_id(),
            "object": "response",
            "created_at": ir.created or int(time.time()),
            "model": ir.model,
            "output": outputs,
            "status": status,
        }
        if status == "incomplete":
            response["incomplete_details"] = {"reason": "max_output_tokens"}

        usage = ir.usage or IRUsage()
        response["usage"] = _usage_to_responses(usage)

        return response

    # ------------------------------------------------------------------
    # Stream encoding (IR stream events → Responses SSE, stateful)
    # ------------------------------------------------------------------

    def _reset_stream_state(self) -> None:
        """Reset per-stream state (sub2api's AnthropicEventToResponsesState)."""
        self._e_response_id: str = ""
        self._e_model: str = ""
        self._e_created: int = int(time.time())
        self._e_seq: int = 0
        self._e_created_sent: bool = False
        self._e_completed_sent: bool = False
        self._e_output_index: int = 0
        self._e_item_id: str = ""
        self._e_item_type: str = ""  # message | function_call | reasoning
        self._e_content_index: int = 0
        self._e_text_accum: str = ""
        self._e_call_id: str = ""
        self._e_name: str = ""
        self._e_content_parts: List[Dict[str, Any]] = []
        self._e_args: str = ""
        self._e_summary: str = ""
        self._e_thinking_sig: str = ""
        self._e_thinking_data: str = ""
        self._e_preserve_sig: bool = False
        self._e_pending_tool_input: str = ""
        self._e_outputs: List[Dict[str, Any]] = []
        self._e_input_tokens: int = 0
        self._e_output_tokens: int = 0
        self._e_cache_read: int = 0
        self._e_cache_creation: int = 0
        self._e_stop_reason: Optional[StopReason] = None

    def encode_stream_event(
        self, ir_event: IRStreamEvent, *, options: Optional[Dict[str, Any]] = None
    ) -> List[Union[Dict[str, Any], str]]:
        """Encode IR stream event to OpenAI Responses format."""
        options = options or {}
        output_format = options.get("output_format", "dict")

        events: List[Dict[str, Any]] = []

        if ir_event.type == StreamEventType.MESSAGE_START:
            response = ir_event.response
            if response is not None:
                self._e_response_id = response.id or self._e_response_id
                self._e_preserve_sig = self._e_preserve_sig or _is_claude_55(
                    response.model or ""
                )
                if not self._e_model:
                    self._e_model = response.model
                usage = response.usage
                if usage is not None:
                    if usage.input_tokens > 0:
                        self._e_input_tokens = usage.input_tokens
                    if usage.cache_read_tokens > 0:
                        self._e_cache_read = usage.cache_read_tokens
                    if usage.cache_creation_tokens > 0:
                        self._e_cache_creation = usage.cache_creation_tokens
            if self._e_created_sent:
                return []
            self._e_created_sent = True
            events.append(
                {
                    "type": "response.created",
                    "sequence_number": self._e_seq,
                    "response": {
                        "id": self._e_response_id,
                        "object": "response",
                        "created_at": self._e_created,
                        "model": self._e_model,
                        "status": "in_progress",
                        "output": [],
                    },
                }
            )
            self._e_seq += 1

        elif ir_event.type == StreamEventType.CONTENT_BLOCK_START:
            events.extend(self._handle_block_start(ir_event))

        elif ir_event.type == StreamEventType.CONTENT_BLOCK_DELTA:
            events.extend(self._handle_block_delta(ir_event))

        elif ir_event.type == StreamEventType.CONTENT_BLOCK_STOP:
            events.extend(self._handle_block_stop())

        elif ir_event.type == StreamEventType.MESSAGE_DELTA:
            usage = ir_event.usage
            if usage is not None:
                self._e_output_tokens = usage.output_tokens
                if usage.input_tokens > 0:
                    self._e_input_tokens = usage.input_tokens
                if usage.cache_read_tokens > 0:
                    self._e_cache_read = usage.cache_read_tokens
                if usage.cache_creation_tokens > 0:
                    self._e_cache_creation = usage.cache_creation_tokens
            if ir_event.stop_reason:
                self._e_stop_reason = ir_event.stop_reason

        elif ir_event.type == StreamEventType.DONE:
            events.extend(self._handle_message_stop())

        elif ir_event.type == StreamEventType.ERROR:
            events.append(
                {
                    "type": "error",
                    "error": {
                        "type": ir_event.error_type or "error",
                        "message": ir_event.error_message or "Unknown error",
                    },
                }
            )

        return [self._format(e, output_format) for e in events]

    def finalize(self) -> List[Union[Dict[str, Any], str]]:
        """Synthesize termination for a stream that ended without
        message_stop (sub2api's FinalizeAnthropicResponsesStream)."""
        if not self._e_created_sent or self._e_completed_sent:
            return []
        events = self._close_current_item()
        events.append(self._make_completed_event())
        self._e_completed_sent = True
        return [self._format(e, "dict") for e in events]

    # --- stream encode helpers ------------------------------------------

    def _handle_block_start(self, ir_event: IRStreamEvent) -> List[Dict[str, Any]]:
        block = ir_event.content_block
        events: List[Dict[str, Any]] = []

        if isinstance(block, IRThinkingBlock):
            # Close any open item first: interleaved thinking after a text
            # block would otherwise overwrite the item state and lose the
            # accumulated assistant text.
            events.extend(self._close_current_item())
            self._e_item_id = _generate_item_id()
            self._e_item_type = "reasoning"
            self._e_thinking_sig = block.signature or ""
            self._e_thinking_data = block.redacted_data or ""
            self._e_summary = block.thinking
            self._e_content_index = 0
            events.append(
                self._wire_event(
                    "response.output_item.added",
                    {
                        "output_index": self._e_output_index,
                        "item": {
                            "type": "reasoning",
                            "id": self._e_item_id,
                            "summary": [],
                        },
                    },
                )
            )

        elif isinstance(block, IRToolUseBlock):
            events.extend(self._close_current_item())
            self._e_item_id = _generate_item_id()
            self._e_item_type = "function_call"
            self._e_call_id = block.id
            self._e_name = block.name
            self._e_args = ""
            # The canonical Anthropic stream leaves input empty here and
            # streams the arguments as input_json_delta; keep a complete
            # inline input as a seed for relays that never send a delta.
            self._e_pending_tool_input = _seed_tool_arguments(block.input)
            events.append(
                self._wire_event(
                    "response.output_item.added",
                    {
                        "output_index": self._e_output_index,
                        "item": {
                            "type": "function_call",
                            "id": self._e_item_id,
                            "call_id": self._e_call_id,
                            "name": self._e_name,
                            "arguments": "",
                            "status": "in_progress",
                        },
                    },
                )
            )

        elif isinstance(block, IRTextBlock):
            if self._e_item_type != "message":
                events.extend(self._close_current_item())
                self._e_item_id = _generate_item_id()
                self._e_item_type = "message"
                self._e_content_index = 0
                events.append(
                    self._wire_event(
                        "response.output_item.added",
                        {
                            "output_index": self._e_output_index,
                            "item": {
                                "type": "message",
                                "id": self._e_item_id,
                                "role": "assistant",
                                "status": "in_progress",
                                "content": [],
                            },
                        },
                    )
                )
            # content_part.added must precede the output_text.delta events
            # for that part (SDK accumulation requirement).
            events.append(
                self._wire_event(
                    "response.content_part.added",
                    {
                        "output_index": self._e_output_index,
                        "content_index": self._e_content_index,
                        "item_id": self._e_item_id,
                        "part": self._output_text_part(""),
                    },
                )
            )
            self._e_text_accum = ""

        return events

    def _handle_block_delta(self, ir_event: IRStreamEvent) -> List[Dict[str, Any]]:
        events: List[Dict[str, Any]] = []

        if ir_event.delta_type == "text":
            if not ir_event.delta_text:
                return []
            self._e_text_accum += ir_event.delta_text
            event: Dict[str, Any] = {
                "output_index": self._e_output_index,
                "content_index": self._e_content_index,
                "item_id": self._e_item_id,
                "delta": ir_event.delta_text,
            }
            events.append(
                self._wire_event("response.output_text.delta", event)
            )

        elif ir_event.delta_type == "thinking":
            if not ir_event.delta_text:
                return []
            self._e_summary += ir_event.delta_text
            events.append(
                self._wire_event(
                    "response.reasoning_summary_text.delta",
                    {
                        "output_index": self._e_output_index,
                        "summary_index": 0,
                        "item_id": self._e_item_id,
                        "delta": ir_event.delta_text,
                    },
                )
            )

        elif ir_event.delta_type == "input_json":
            if not ir_event.delta_json:
                return []
            # A real delta supersedes the seed — splicing two complete JSON
            # documents together corrupts the arguments.
            self._e_pending_tool_input = ""
            self._e_args += ir_event.delta_json
            events.append(
                self._wire_event(
                    "response.function_call_arguments.delta",
                    {
                        "output_index": self._e_output_index,
                        "item_id": self._e_item_id,
                        "call_id": self._e_call_id,
                        "name": self._e_name,
                        "delta": ir_event.delta_json,
                    },
                )
            )

        elif ir_event.delta_type == "signature":
            # Signatures ride in the opaque envelope, never in visible text.
            if self._e_preserve_sig:
                self._e_thinking_sig += ir_event.delta_text or ""

        return events

    def _handle_block_stop(self) -> List[Dict[str, Any]]:
        events: List[Dict[str, Any]] = []

        if self._e_item_type == "reasoning":
            events.append(
                self._wire_event(
                    "response.reasoning_summary_text.done",
                    {
                        "output_index": self._e_output_index,
                        "summary_index": 0,
                        "item_id": self._e_item_id,
                        "text": "",
                    },
                )
            )
            events.extend(self._close_current_item())

        elif self._e_item_type == "function_call":
            # No delta ever arrived: emit the seed once so the done event
            # repeats exactly what the deltas would have streamed.
            if not self._e_args and self._e_pending_tool_input:
                self._e_args = self._e_pending_tool_input
                events.append(
                    self._wire_event(
                        "response.function_call_arguments.delta",
                        {
                            "output_index": self._e_output_index,
                            "item_id": self._e_item_id,
                            "call_id": self._e_call_id,
                            "name": self._e_name,
                            "delta": self._e_pending_tool_input,
                        },
                    )
                )
            self._e_pending_tool_input = ""
            events.append(
                self._wire_event(
                    "response.function_call_arguments.done",
                    {
                        "output_index": self._e_output_index,
                        "item_id": self._e_item_id,
                        "call_id": self._e_call_id,
                        "name": self._e_name,
                        "arguments": self._e_args,
                    },
                )
            )
            events.extend(self._close_current_item())

        elif self._e_item_type == "message":
            # Text part done: full text on both events; the message item
            # stays open for potential further blocks.
            text = self._e_text_accum
            self._e_text_accum = ""
            content_index = self._e_content_index
            self._e_content_parts.append({"type": "output_text", "text": text})
            self._e_content_index += 1
            events.append(
                self._wire_event(
                    "response.output_text.done",
                    {
                        "output_index": self._e_output_index,
                        "content_index": content_index,
                        "item_id": self._e_item_id,
                        "text": text,
                    },
                )
            )
            events.append(
                self._wire_event(
                    "response.content_part.done",
                    {
                        "output_index": self._e_output_index,
                        "content_index": content_index,
                        "item_id": self._e_item_id,
                        "part": self._output_text_part(text),
                    },
                )
            )

        return events

    def _handle_message_stop(self) -> List[Dict[str, Any]]:
        if self._e_completed_sent:
            return []
        events = self._close_current_item()
        events.append(self._make_completed_event())
        self._e_completed_sent = True
        return events

    def _close_current_item(self) -> List[Dict[str, Any]]:
        """Assemble the full open item, accumulate it, and emit
        output_item.done — SDK-side accumulation reads the content from
        these payloads."""
        if not self._e_item_type:
            return []

        item: Dict[str, Any] = {
            "type": self._e_item_type,
            "id": self._e_item_id,
            "status": "completed",
        }
        if self._e_item_type == "message":
            item["role"] = "assistant"
            item["content"] = [
                {"type": "output_text", "text": p["text"]}
                for p in self._e_content_parts
            ]
        elif self._e_item_type == "function_call":
            item["call_id"] = self._e_call_id
            item["name"] = self._e_name
            item["arguments"] = self._e_args if self._e_args else "{}"
        elif self._e_item_type == "reasoning":
            if self._e_preserve_sig and (self._e_thinking_sig or self._e_thinking_data):
                item["encrypted_content"] = _encode_thinking_envelope(
                    IRThinkingBlock(
                        thinking=self._e_summary,
                        signature=self._e_thinking_sig,
                        is_redacted=bool(self._e_thinking_data),
                        redacted_data=self._e_thinking_data,
                    )
                )
            if self._e_summary:
                item["summary"] = [
                    {"type": "summary_text", "text": self._e_summary}
                ]

        self._e_outputs.append(item)
        output_index = self._e_output_index

        # Reset per-item state.
        self._e_item_type = ""
        self._e_item_id = ""
        self._e_call_id = ""
        self._e_name = ""
        self._e_content_parts = []
        self._e_args = ""
        self._e_pending_tool_input = ""
        self._e_summary = ""
        self._e_thinking_sig = ""
        self._e_thinking_data = ""
        self._e_text_accum = ""
        self._e_output_index += 1
        self._e_content_index = 0

        return [
            self._wire_event(
                "response.output_item.done",
                {"output_index": output_index, "item": item},
            )
        ]

    def _make_completed_event(self) -> Dict[str, Any]:
        status = (
            "incomplete" if self._e_stop_reason == StopReason.MAX_TOKENS else "completed"
        )
        event_type = (
            "response.incomplete" if status == "incomplete" else "response.completed"
        )
        usage = _usage_to_responses(
            IRUsage(
                input_tokens=self._e_input_tokens,
                output_tokens=self._e_output_tokens,
                cache_read_tokens=self._e_cache_read,
                cache_creation_tokens=self._e_cache_creation,
            )
        )
        response: Dict[str, Any] = {
            "id": self._e_response_id,
            "object": "response",
            "created_at": self._e_created,
            "model": self._e_model,
            "status": status,
            # The terminal response object serializes with struct-tag
            # semantics (omitempty): empty content/arguments are omitted,
            # unlike the output_item.done wire shapes.
            "output": [self._struct_shape(item) for item in self._e_outputs],
            "usage": usage,
        }
        if status == "incomplete":
            response["incomplete_details"] = {"reason": "max_output_tokens"}
        return {
            "type": event_type,
            "sequence_number": self._e_seq,
            "response": response,
        }

    @staticmethod
    def _struct_shape(item: Dict[str, Any]) -> Dict[str, Any]:
        """Drop empty-valued keys (Go's omitempty) for the terminal payload."""
        shaped: Dict[str, Any] = {"type": item["type"]}
        for key in ("id", "status", "role", "encrypted_content", "call_id", "name"):
            if item.get(key):
                shaped[key] = item[key]
        if item.get("content"):
            shaped["content"] = item["content"]
        if item.get("summary"):
            shaped["summary"] = item["summary"]
        if item.get("arguments"):
            shaped["arguments"] = item["arguments"]
        return shaped

    def _wire_event(self, event_type: str, fields: Dict[str, Any]) -> Dict[str, Any]:
        """Attach type + sequence number (always present on the wire)."""
        event = {"type": event_type, "sequence_number": self._e_seq}
        self._e_seq += 1
        event.update(fields)
        return event

    def _output_text_part(self, text: str) -> Dict[str, Any]:
        """Wire shape of an output_text content part (annotations/logprobs
        always present — SDK requirement)."""
        return {
            "type": "output_text",
            "text": text,
            "annotations": [],
            "logprobs": [],
        }

    def _format(
        self, event: Dict[str, Any], output_format: str
    ) -> Union[Dict[str, Any], str]:
        """Format event based on output format."""
        if output_format == "sse":
            return f"event: {event['type']}\ndata: {json.dumps(event)}\n\n"
        return event


def _generate_item_id() -> str:
    return "item_" + secrets.token_hex(12)


def _generate_response_id() -> str:
    return "resp_" + secrets.token_hex(12)
