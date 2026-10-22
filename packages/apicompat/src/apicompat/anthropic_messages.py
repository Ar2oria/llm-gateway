"""
Anthropic Messages API Encoder/Decoder

Converts between Anthropic Messages API format and the Intermediate Representation.
"""

import copy
import json
from typing import Any, Dict, List, Optional, Tuple, Union
import re

from .ir import (
    ImageSourceType,
    IRContentBlock,
    IRDocumentBlock,
    IRGenerationConfig,
    IRImageBlock,
    IRMessage,
    IRRequest,
    IRResponse,
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
    Role,
    StopReason,
    StreamEventType,
    ToolChoiceType,
)


_TOP_LEVEL_COMBINATORS = ("anyOf", "oneOf", "allOf")

#: Reasoning-effort clamping: Claude's output_config.effort accepts `max`,
#: but some OpenAI-compatible upstreams (e.g. opencode Console Go) only accept
#: none/minimal/low/medium/high/xhigh and reject `max` with a 400. Clamp the
#: wide end to the largest value the upstream domain knows.
_CLAMPED_EFFORT = {
    "adaptive": "xhigh",
}

#: Placeholder signature for thinking blocks synthesized from plaintext
#: reasoning (OpenAI Chat reasoning_content/reasoning, Responses reasoning
#: summaries). Deprecated: sub2api-shaped output always carries a
#: ``signature`` key (empty string when unsigned) on thinking blocks, which
#: satisfies the same strict clients (``signature: z.string()``) without a
#: magic value. Kept for backwards compatibility with existing tests.

def _strip_top_level_combinators(schema: Any) -> Any:
    """Collapse top-level anyOf/oneOf/allOf in a tool input schema.

    The Anthropic API rejects ``input_schema`` with ``anyOf``/``oneOf``/``allOf``
    at the top level. This merges each branch's ``properties`` into the
    top-level ``properties`` (union), clears top-level ``required``, removes the
    combinator keys, and ensures ``type``/``properties``. Only the top level is
    touched; nested combinators are valid for Anthropic and left intact.
    """
    if not isinstance(schema, dict):
        return schema
    if not any(key in schema for key in _TOP_LEVEL_COMBINATORS):
        return schema

    cleaned = copy.deepcopy(schema)
    merged_properties: Dict[str, Any] = dict(cleaned.get("properties") or {})
    for key in _TOP_LEVEL_COMBINATORS:
        branches = cleaned.pop(key, None)
        if not isinstance(branches, list):
            continue
        for branch in branches:
            if not isinstance(branch, dict):
                continue
            branch_properties = branch.get("properties")
            if isinstance(branch_properties, dict):
                for prop_name, prop_schema in branch_properties.items():
                    merged_properties.setdefault(prop_name, prop_schema)

    cleaned["type"] = "object"
    cleaned["properties"] = merged_properties
    cleaned["required"] = []
    return cleaned


def _normalize_input_schema(schema: Any) -> Optional[Dict[str, Any]]:
    """Normalize a tool parameter schema for Anthropic's ``input_schema``.

    Anthropic requires an object root with ``properties``. Empty/None
    schemas become ``{"type": "object", "properties": {}}``; non-object
    roots are replaced by the empty object schema. Returns None when the
    input is unusable (caller decides the fallback).
    """
    if not isinstance(schema, dict) or not schema:
        return {"type": "object", "properties": {}}
    cleaned = copy.deepcopy(schema)
    root_type = cleaned.get("type")
    if root_type is None or root_type == "null":
        cleaned["type"] = "object"
    elif root_type != "object":
        return {"type": "object", "properties": {}}
    if "properties" not in cleaned:
        cleaned["properties"] = {}
    return cleaned


class AnthropicMessagesDecoder:
    """Decodes Anthropic Messages API format to IR."""

    def decode_request(self, payload: Dict[str, Any]) -> IRRequest:
        """Decode an Anthropic Messages request to IR."""
        ir = IRRequest(
            model=payload.get("model", ""),
            stream=payload.get("stream", False),
        )

        # Decode messages
        messages = payload.get("messages", [])
        ir.messages = self._decode_messages(messages)

        # Decode system prompt. ``system`` may be a string or an array of
        # text blocks; the parts are preserved separately ( Responses encodes
        # them as one input_text part per block, preserving the cache shape).
        # ``x-anthropic-billing-header:`` blocks are dropped (airgate quirk:
        # they must not reach Codex-style upstreams).
        system = payload.get("system")
        if isinstance(system, str):
            if system and not system.startswith("x-anthropic-billing-header: "):
                ir.system = system
                ir.system_parts = [system]
        elif isinstance(system, list):
            parts = [
                block.get("text", "")
                for block in system
                if isinstance(block, dict)
                and block.get("type") == "text"
                and block.get("text")
                and not str(block.get("text", "")).startswith(
                    "x-anthropic-billing-header: "
                )
            ]
            if parts:
                ir.system = "\n\n".join(parts)
                ir.system_parts = parts

        # Decode generation config
        ir.generation_config = self._decode_generation_config(payload)

        # Decode tools
        if "tools" in payload:
            ir.tools = self._decode_tools(payload["tools"])

        # Decode tool choice
        if "tool_choice" in payload:
            ir.tool_choice = self._decode_tool_choice(payload["tool_choice"])

        # Decode thinking config. sub2api semantics: an explicit
        # ``thinking.type: "disabled"`` overrides everything (effort "none");
        # otherwise the effort comes from ``output_config.effort`` (classic
        # ``budget_tokens`` does not map to an effort and is kept in the IR
        # only for encoders that can use it).
        if "thinking" in payload:
            ir.thinking = self._decode_thinking_config(payload["thinking"])
        # Claude Code sends reasoning effort in output_config.effort
        # (minimal/low/medium/high/max, possibly xhigh); carry it into the IR
        # so it can be emitted as OpenAI Responses "reasoning.effort" when
        # converting. Claude's value domain is wider than some upstreams'
        # (opencode Console Go accepts none/minimal/low/medium/high/xhigh and
        # rejects `max` with a 400), so `max` is clamped to `xhigh`.
        output_config = payload.get("output_config")
        if isinstance(output_config, dict) and output_config.get("effort"):
            if ir.thinking is None:
                ir.thinking = IRThinkingConfig()
            if ir.thinking.effort != "none":
                ir.thinking.effort = _CLAMPED_EFFORT.get(
                    str(output_config["effort"]), str(output_config["effort"])
                )

        # sub2api semantics: Anthropic sources always resolve to an effort
        # ("none" when thinking is explicitly disabled, else output_config's
        # value or "medium") so downstream protocol fields (Responses
        # reasoning.effort, Chat reasoning_effort) are always populated.
        if ir.thinking is None:
            ir.thinking = IRThinkingConfig()
        if ir.thinking.effort is None:
            ir.thinking.effort = "medium"

        # User ID from metadata
        metadata = payload.get("metadata", {})
        if "user_id" in metadata:
            ir.user = metadata["user_id"]

        # Store unsupported params
        unsupported_keys = ["service_tier"]
        for key in unsupported_keys:
            if key in payload:
                ir.unsupported_params[key] = payload[key]

        return ir

    def _decode_messages(self, messages: List[Dict[str, Any]]) -> List[IRMessage]:
        """Decode messages to IR format."""
        ir_messages = []

        for msg in messages:
            role = self._map_role(msg.get("role", "user"))
            content = msg.get("content", "")

            ir_message = IRMessage(
                role=role,
                plain_text=isinstance(content, str),
            )
            ir_message.content = self._decode_content(content)

            # Check if message contains only tool_result blocks
            # If so, convert each to a separate TOOL role message
            if role == Role.USER:
                tool_results = [
                    b for b in ir_message.content if isinstance(b, IRToolResultBlock)
                ]
                other_content = [
                    b
                    for b in ir_message.content
                    if not isinstance(b, IRToolResultBlock)
                ]

                if tool_results:
                    # Add each tool result as separate TOOL message
                    for tr in tool_results:
                        tool_msg = IRMessage(role=Role.TOOL)
                        tool_msg.content = [tr]
                        ir_messages.append(tool_msg)

                    # If there's other content, add it as user message
                    if other_content:
                        ir_message.content = other_content
                        ir_messages.append(ir_message)
                    continue

            ir_messages.append(ir_message)

        return ir_messages

    def _decode_content(
        self, content: Union[str, List[Dict[str, Any]]]
    ) -> List[IRContentBlock]:
        """Decode content to IR blocks."""
        if isinstance(content, str):
            return [IRTextBlock(text=content)] if content else []

        blocks: List[IRContentBlock] = []
        for item in content:
            block = self._decode_content_block(item)
            if block:
                blocks.append(block)

        return blocks

    def _decode_content_block(self, block: Dict[str, Any]) -> Optional[IRContentBlock]:
        """Decode a single content block."""
        block_type = block.get("type", "text")

        if block_type == "text":
            return IRTextBlock(text=block.get("text", ""))

        elif block_type == "image":
            source = block.get("source", {})
            source_type = source.get("type", "base64")

            if source_type == "base64":
                return IRImageBlock(
                    source_type=ImageSourceType.BASE64,
                    base64_data=source.get("data"),
                    media_type=source.get("media_type"),
                )
            else:  # url
                return IRImageBlock(
                    source_type=ImageSourceType.URL,
                    url=source.get("url"),
                )

        elif block_type == "document":
            source = block.get("source", {})
            return IRDocumentBlock(
                source_type=source.get("type", "base64"),
                data=source.get("data"),
                url=source.get("url"),
                media_type=source.get("media_type"),
                title=block.get("title"),
                context=block.get("context"),
            )

        elif block_type == "tool_use":
            return IRToolUseBlock(
                id=block.get("id", ""),
                name=block.get("name", ""),
                input=block.get("input", {}),
            )

        elif block_type == "tool_result":
            content = block.get("content", "")
            return IRToolResultBlock(
                tool_use_id=block.get("tool_use_id", ""),
                content=content
                if isinstance(content, str)
                else self._decode_tool_result_content(content),
                is_error=block.get("is_error", False),
            )

        elif block_type == "thinking":
            return IRThinkingBlock(
                thinking=block.get("thinking", ""),
                signature=block.get("signature"),
            )

        elif block_type == "redacted_thinking":
            return IRThinkingBlock(
                is_redacted=True,
                redacted_data=block.get("data"),
            )

        return None

    def _decode_tool_result_content(
        self, content: List[Dict[str, Any]]
    ) -> List[IRContentBlock]:
        """Decode tool result content blocks."""
        blocks = []
        for item in content:
            if item.get("type") == "text":
                blocks.append(IRTextBlock(text=item.get("text", "")))
            elif item.get("type") == "image":
                source = item.get("source", {})
                blocks.append(
                    IRImageBlock(
                        source_type=ImageSourceType.BASE64
                        if source.get("type") == "base64"
                        else ImageSourceType.URL,
                        base64_data=source.get("data"),
                        url=source.get("url"),
                        media_type=source.get("media_type"),
                    )
                )
        return blocks

    def _decode_generation_config(self, payload: Dict[str, Any]) -> IRGenerationConfig:
        """Decode generation configuration."""
        config = IRGenerationConfig()

        if "temperature" in payload:
            config.temperature = payload["temperature"]
        if "top_p" in payload:
            config.top_p = payload["top_p"]
        if "top_k" in payload:
            config.top_k = payload["top_k"]
        if "max_tokens" in payload:
            config.max_tokens = payload["max_tokens"]
        if "stop_sequences" in payload:
            config.stop_sequences = payload["stop_sequences"]

        return config

    def _decode_tools(self, tools: List[Dict[str, Any]]) -> List[IRToolDeclaration]:
        """Decode tool declarations."""
        ir_tools = []
        for tool in tools:
            ir_tools.append(
                IRToolDeclaration(
                    name=tool.get("name", ""),
                    description=tool.get("description"),
                    parameters=tool.get("input_schema", {}),
                    type=tool.get("type"),
                )
            )
        return ir_tools

    def _decode_tool_choice(self, tool_choice: Dict[str, Any]) -> IRToolChoice:
        """Decode tool choice configuration."""
        choice_type = tool_choice.get("type", "auto")

        ir_choice = IRToolChoice()

        if choice_type == "auto":
            ir_choice.type = ToolChoiceType.AUTO
        elif choice_type == "none":
            ir_choice.type = ToolChoiceType.NONE
        elif choice_type == "any":
            ir_choice.type = ToolChoiceType.ANY
        elif choice_type == "tool":
            ir_choice.type = ToolChoiceType.SPECIFIC
            ir_choice.name = tool_choice.get("name", "")

        ir_choice.disable_parallel = tool_choice.get("disable_parallel_tool_use", False)

        return ir_choice

    def _decode_thinking_config(self, thinking: Dict[str, Any]) -> IRThinkingConfig:
        """Decode thinking configuration."""
        config = IRThinkingConfig()
        thinking_type = thinking.get("type")

        if thinking_type == "disabled":
            config.effort = "none"
        # enabled/adaptive/between_tools all mean "thinking on"; the concrete
        # budget is not carried across protocols (sub2api re-derives it).

        return config

    def decode_response(self, payload: Dict[str, Any]) -> IRResponse:
        """Decode an Anthropic Messages response to IR."""
        ir = IRResponse(
            id=payload.get("id", ""),
            model=payload.get("model", ""),
        )

        # Decode content blocks
        content = payload.get("content", [])
        for block in content:
            ir_block = self._decode_content_block(block)
            if ir_block:
                ir.content.append(ir_block)

        # Decode stop reason
        stop_reason = payload.get("stop_reason")
        ir.stop_reason = self._map_stop_reason(stop_reason)
        ir.stop_sequence = payload.get("stop_sequence")

        # Decode usage
        usage = payload.get("usage")
        if usage:
            ir.usage = IRUsage(
                input_tokens=usage.get("input_tokens", 0),
                output_tokens=usage.get("output_tokens", 0),
                cache_creation_tokens=usage.get("cache_creation_input_tokens", 0),
                cache_read_tokens=usage.get("cache_read_input_tokens", 0),
            )

        return ir

    def decode_stream_event(
        self, event: Union[Dict[str, Any], str]
    ) -> List[IRStreamEvent]:
        """Decode a streaming event to IR events."""
        # Handle SSE text format
        if isinstance(event, str):
            event = self._parse_sse_event(event)
            if event is None:
                return []

        ir_events = []
        event_type = event.get("type", "")

        if event_type == "message_start":
            message = event.get("message", {})
            message_usage = message.get("usage") or {}
            ir_events.append(
                IRStreamEvent(
                    type=StreamEventType.MESSAGE_START,
                    response=IRResponse(
                        id=message.get("id", ""),
                        model=message.get("model", ""),
                        usage=IRUsage(
                            input_tokens=message_usage.get("input_tokens", 0),
                            output_tokens=message_usage.get("output_tokens", 0),
                            cache_creation_tokens=message_usage.get(
                                "cache_creation_input_tokens", 0
                            ),
                            cache_read_tokens=message_usage.get(
                                "cache_read_input_tokens", 0
                            ),
                        ),
                    ),
                )
            )

        elif event_type == "content_block_start":
            index = event.get("index", 0)
            block = event.get("content_block", {})
            block_type = block.get("type", "text")

            if block_type == "text":
                ir_events.append(
                    IRStreamEvent(
                        type=StreamEventType.CONTENT_BLOCK_START,
                        index=index,
                        content_block=IRTextBlock(),
                    )
                )
            elif block_type == "tool_use":
                ir_events.append(
                    IRStreamEvent(
                        type=StreamEventType.CONTENT_BLOCK_START,
                        index=index,
                        content_block=IRToolUseBlock(
                            id=block.get("id", ""),
                            name=block.get("name", ""),
                            # Relays may put the complete arguments here and
                            # never send an input_json_delta; keep them as a
                            # seed (sub2api's PendingToolInput).
                            input=block.get("input") or {},
                        ),
                    )
                )
            elif block_type == "thinking":
                ir_events.append(
                    IRStreamEvent(
                        type=StreamEventType.CONTENT_BLOCK_START,
                        index=index,
                        content_block=IRThinkingBlock(),
                    )
                )

        elif event_type == "content_block_delta":
            index = event.get("index", 0)
            delta = event.get("delta", {})
            delta_type = delta.get("type", "")

            if delta_type == "text_delta":
                ir_events.append(
                    IRStreamEvent(
                        type=StreamEventType.CONTENT_BLOCK_DELTA,
                        index=index,
                        delta_type="text",
                        delta_text=delta.get("text", ""),
                    )
                )
            elif delta_type == "input_json_delta":
                ir_events.append(
                    IRStreamEvent(
                        type=StreamEventType.CONTENT_BLOCK_DELTA,
                        index=index,
                        delta_type="input_json",
                        delta_json=delta.get("partial_json", ""),
                    )
                )
            elif delta_type == "thinking_delta":
                ir_events.append(
                    IRStreamEvent(
                        type=StreamEventType.CONTENT_BLOCK_DELTA,
                        index=index,
                        delta_type="thinking",
                        delta_text=delta.get("thinking", ""),
                    )
                )
            elif delta_type == "signature_delta":
                ir_events.append(
                    IRStreamEvent(
                        type=StreamEventType.CONTENT_BLOCK_DELTA,
                        index=index,
                        delta_type="signature",
                        delta_text=delta.get("signature", ""),
                    )
                )

        elif event_type == "content_block_stop":
            ir_events.append(
                IRStreamEvent(
                    type=StreamEventType.CONTENT_BLOCK_STOP,
                    index=event.get("index", 0),
                )
            )

        elif event_type == "message_delta":
            delta = event.get("delta", {})
            usage = event.get("usage", {})

            ir_events.append(
                IRStreamEvent(
                    type=StreamEventType.MESSAGE_DELTA,
                    stop_reason=self._map_stop_reason(delta.get("stop_reason")),
                    stop_sequence=delta.get("stop_sequence"),
                    usage=IRUsage(
                        input_tokens=usage.get("input_tokens", 0),
                        output_tokens=usage.get("output_tokens", 0),
                        cache_creation_tokens=usage.get(
                            "cache_creation_input_tokens", 0
                        ),
                        cache_read_tokens=usage.get("cache_read_input_tokens", 0),
                    )
                    if usage
                    else None,
                )
            )

        elif event_type == "message_stop":
            ir_events.append(IRStreamEvent(type=StreamEventType.DONE))

        elif event_type == "ping":
            ir_events.append(IRStreamEvent(type=StreamEventType.PING))

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

        if data:
            if event_type:
                data["type"] = event_type
            return data
        return None

    def _map_role(self, role: str) -> Role:
        """Map Anthropic role to IR role.

        Anthropic officially only allows user/assistant roles, but clients
        (e.g. Claude Code) sometimes send role=system messages. sub2api maps
        unknown roles (including system) to USER content, so they survive
        cross-protocol conversion instead of being dropped.
        """
        role_map = {
            "user": Role.USER,
            "assistant": Role.ASSISTANT,
        }
        return role_map.get(role, Role.USER)

    def _map_stop_reason(self, reason: Optional[str]) -> Optional[StopReason]:
        """Map Anthropic stop reason to IR stop reason."""
        if not reason:
            return None
        reason_map = {
            "end_turn": StopReason.END_TURN,
            "max_tokens": StopReason.MAX_TOKENS,
            "stop_sequence": StopReason.STOP_SEQUENCE,
            "tool_use": StopReason.TOOL_USE,
            "refusal": StopReason.CONTENT_FILTER,
            "pause_turn": StopReason.END_TURN,
        }
        return reason_map.get(reason, StopReason.END_TURN)


class AnthropicMessagesEncoder:
    """Encodes IR to Anthropic Messages API format."""

    def encode_request(
        self, ir: IRRequest, *, options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Encode IR request to Anthropic Messages format."""
        options = options or {}
        model_lower = ir.model.lower()

        # Anthropic requires max_tokens; default like sub2api rather than
        # failing the whole conversion.
        max_tokens = ir.generation_config.max_tokens or 8192

        # SYSTEM-role IR messages (Responses developer items, Chat system
        # messages) fold into the top-level system prompt — sub2api extracts
        # every system/developer item into systemParts regardless of
        # position. The source protocol's own system field comes first.
        messages, system_texts = self._extract_system_messages(ir.messages)
        system_parts = ([ir.system] if ir.system else []) + system_texts

        payload: Dict[str, Any] = {
            "model": ir.model,
            "messages": self._encode_messages(messages),
            "max_tokens": max_tokens,
        }

        # System prompt
        if system_parts:
            payload["system"] = "\n\n".join(system_parts)

        # Stream
        if ir.stream:
            payload["stream"] = True

        # Generation config
        config = ir.generation_config
        if config.temperature is not None:
            payload["temperature"] = config.temperature
        if config.top_p is not None:
            payload["top_p"] = config.top_p
        if config.stop_sequences:
            payload["stop_sequences"] = config.stop_sequences

        # Tools
        if ir.tools:
            payload["tools"] = self._encode_tools(ir.tools)

        # Tool choice
        if ir.tool_choice:
            payload["tool_choice"] = self._encode_tool_choice(ir.tool_choice)

        # Thinking config: effort → output_config.effort. Non-low efforts
        # enable thinking without deriving budget_tokens; Claude 5-5 uses
        # adaptive thinking instead. xhigh maps back to Claude's "max".
        if ir.thinking and ir.thinking.effort:
            effort = ir.thinking.effort
            effort = _CLAMPED_EFFORT.get(effort, effort)
            payload["output_config"] = {"effort": effort}
            if effort not in ( "low", "none" ):
                payload["thinking"] = {
                    "type": "enabled"
                }
                if model_lower.find("5-5") != -1:
                    payload["thinking"] = {
                        "type": "adaptive"
                    }
            elif model_lower.find("claude") != -1:
                payload["output_config"] = {"effort": "low"}

        # User ID in metadata
        if ir.user:
            payload["metadata"] = {"user_id": ir.user}

        # Store unsupported params in metadata if option enabled
        if options.get("preserve_unsupported", False) and ir.unsupported_params:
            if "metadata" not in payload:
                payload["metadata"] = {}
            payload["metadata"]["_unsupported"] = ir.unsupported_params

        return payload

    # ------------------------------------------------------------------
    # Message normalization (Anthropic invariants)
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_system_messages(
        messages: List[IRMessage],
    ) -> Tuple[List[IRMessage], List[str]]:
        """Split SYSTEM-role messages out of the list, returning the
        remaining messages and the system texts in order."""
        remaining: List[IRMessage] = []
        system_texts: List[str] = []
        for msg in messages:
            if msg.role == Role.SYSTEM:
                for block in msg.content:
                    if isinstance(block, IRTextBlock) and block.text:
                        system_texts.append(block.text)
                continue
            remaining.append(msg)
        return remaining, system_texts

    def _normalize_messages(
        self, messages: List[IRMessage]
    ) -> List[Tuple[str, List[IRContentBlock], bool]]:
        """Rebuild the message sequence so it satisfies Anthropic's invariants.

        Mirrors sub2api's Responses→Anthropic pipeline: fold roles (TOOL →
        user), drop empty/blank-only turns, merge consecutive same-role
        messages, repair tool_use/tool_result pairing, then merge again (the
        repair may re-split a user turn). Returns (role, blocks, plain)
        tuples — ``plain`` marks content that the source protocol carried as
        a plain string and that survived merging untouched.
        """
        entries: List[Tuple[str, List[IRContentBlock], bool]] = []
        for msg in messages:
            if msg.role in (Role.USER, Role.TOOL):
                role = "user"
            elif msg.role == Role.ASSISTANT:
                role = "assistant"
            else:
                continue
            if not self._blocks_have_content(msg.content):
                continue
            entries.append((role, list(msg.content), msg.plain_text))

        merged = self._merge_entries(entries)
        repaired = self._repair_tool_pairing(merged)
        return self._merge_entries(repaired)

    @staticmethod
    def _blocks_have_content(blocks: List[IRContentBlock]) -> bool:
        """Content is empty when there are no blocks or only blank text."""
        if not blocks:
            return False
        for block in blocks:
            if isinstance(block, IRTextBlock):
                if block.text.strip():
                    return True
            else:
                return True
        return False

    @staticmethod
    def _merge_entries(
        entries: List[Tuple[str, List[IRContentBlock], bool]]
    ) -> List[Tuple[str, List[IRContentBlock], bool]]:
        """Merge consecutive same-role entries; merged content is always
        block-shaped (a plain string is wrapped when it merges)."""
        merged: List[Tuple[str, List[IRContentBlock], bool]] = []
        for role, blocks, plain in entries:
            if merged and merged[-1][0] == role:
                prev_role, prev_blocks, _ = merged[-1]
                merged[-1] = (prev_role, prev_blocks + blocks, False)
            else:
                merged.append((role, blocks, plain))
        return merged

    @staticmethod
    def _repair_tool_pairing(
        entries: List[Tuple[str, List[IRContentBlock], bool]]
    ) -> List[Tuple[str, List[IRContentBlock], bool]]:
        """Repair tool_use/tool_result adjacency (sub2api algorithm).

        Every tool_result must sit in the user message immediately after the
        assistant message carrying its tool_use; unanswered tool_use blocks
        (and orphan tool_results) are dropped so the request cannot be
        rejected with "tool_result ... must have a corresponding tool_use".
        """
        results: Dict[str, IRToolResultBlock] = {}
        for role, blocks, _plain in entries:
            if role != "user":
                continue
            for block in blocks:
                if isinstance(block, IRToolResultBlock) and block.tool_use_id:
                    results[block.tool_use_id] = block

        out: List[Tuple[str, List[IRContentBlock], bool]] = []
        for role, blocks, plain in entries:
            if role == "assistant":
                tool_uses = [b for b in blocks if isinstance(b, IRToolUseBlock)]
                others = [b for b in blocks if not isinstance(b, IRToolUseBlock)]
                if not tool_uses:
                    out.append((role, blocks, plain))
                    continue
                kept = [tu for tu in tool_uses if tu.id in results]
                if not kept:
                    # No answered calls: keep any non-tool content, else drop.
                    if others:
                        out.append((role, others, False))
                    continue
                out.append((role, others + kept, False))
                out.append(
                    ("user", [results[tu.id] for tu in kept], False)
                )
            elif role == "user":
                has_result = any(
                    isinstance(b, IRToolResultBlock) for b in blocks
                )
                if not has_result:
                    out.append((role, blocks, plain))
                    continue
                # tool_result blocks are re-emitted next to their call; keep
                # any other content of this user turn in place.
                non_result = [
                    b for b in blocks if not isinstance(b, IRToolResultBlock)
                ]
                if non_result:
                    out.append((role, non_result, False))
            else:
                out.append((role, blocks, plain))
        return out

    def _encode_messages(self, messages: List[IRMessage]) -> List[Dict[str, Any]]:
        """Encode IR messages to Anthropic format."""
        result: List[Dict[str, Any]] = []

        for role, blocks, plain in self._normalize_messages(messages):
            if plain and len(blocks) == 1 and isinstance(blocks[0], IRTextBlock):
                content: Any = blocks[0].text
            else:
                content = [
                    encoded
                    for encoded in (
                        self._encode_content_block(block) for block in blocks
                    )
                    if encoded
                ]
            if role == "assistant" and not content:
                # Anthropic rejects empty assistant content; keep the shape.
                content = [{"type": "text", "text": ""}]
            result.append({"role": role, "content": content})

        return result

    def _encode_content_block(self, block: IRContentBlock) -> Optional[Dict[str, Any]]:
        """Encode a content block to Anthropic format."""
        if isinstance(block, IRTextBlock):
            return {"type": "text", "text": block.text}

        elif isinstance(block, IRImageBlock):
            if block.source_type == ImageSourceType.BASE64:
                return {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": block.media_type or "image/png",
                        "data": block.base64_data,
                    },
                }
            else:
                return {
                    "type": "image",
                    "source": {
                        "type": "url",
                        "url": block.url,
                    },
                }

        elif isinstance(block, IRDocumentBlock):
            source: Dict[str, Any] = {"type": block.source_type}
            if block.data:
                source["data"] = block.data
            if block.url:
                source["url"] = block.url
            if block.media_type:
                source["media_type"] = block.media_type

            result: Dict[str, Any] = {"type": "document", "source": source}
            if block.title:
                result["title"] = block.title
            if block.context:
                result["context"] = block.context
            return result

        elif isinstance(block, IRToolUseBlock):
            return {
                "type": "tool_use",
                "id": block.id,
                "name": block.name,
                "input": block.input,
            }

        elif isinstance(block, IRToolResultBlock):
            result: Dict[str, Any] = {
                "type": "tool_result",
                "tool_use_id": block.tool_use_id,
            }

            if isinstance(block.content, str):
                result["content"] = block.content
            elif isinstance(block.content, list):
                result["content"] = [
                    self._encode_content_block(b) for b in block.content if b
                ]
            else:
                result["content"] = str(block.content)

            if block.is_error:
                result["is_error"] = True

            return result

        elif isinstance(block, IRThinkingBlock):
            if block.is_redacted:
                return {
                    "type": "redacted_thinking",
                    "data": block.redacted_data,
                }
            # The signature key must always be present on thinking blocks
            # (sub2api wire shape): strict clients reject its absence, and an
            # empty string is a valid unsigned value that upstreams ignore.
            return {
                "type": "thinking",
                "thinking": block.thinking,
                "signature": block.signature if block.signature else "",
            }

        elif isinstance(block, IRServerToolUseBlock):
            return {
                "type": "server_tool_use",
                "id": block.id,
                "name": block.name,
                "input": block.input,
            }

        elif isinstance(block, IRWebSearchToolResultBlock):
            return {
                "type": "web_search_tool_result",
                "tool_use_id": block.tool_use_id,
                "content": block.content,
            }

        return None

    def _encode_tools(self, tools: List[IRToolDeclaration]) -> List[Dict[str, Any]]:
        """Encode tool declarations."""
        result = []
        for tool in tools:
            if tool.type and tool.type.startswith("web_search"):
                # Server-side web search tool (from a Responses "web_search").
                result.append(
                    {"type": "web_search_20250305", "name": "web_search"}
                )
                continue
            if tool.type:
                # Unknown server tool type: pass the type through.
                encoded: Dict[str, Any] = {
                    "type": tool.type,
                    "name": tool.name,
                }
                if tool.description:
                    encoded["description"] = tool.description
                schema = _normalize_input_schema(tool.parameters)
                if schema is not None:
                    encoded["input_schema"] = schema
                result.append(encoded)
                continue
            input_schema = _strip_top_level_combinators(
                _normalize_input_schema(tool.parameters)
                or {"type": "object", "properties": {}}
            )
            encoded = {
                "name": tool.name,
                "input_schema": input_schema,
            }
            if tool.description:
                encoded["description"] = tool.description
            result.append(encoded)
        return result

    def _encode_tool_choice(self, choice: IRToolChoice) -> Dict[str, Any]:
        """Encode tool choice."""
        result: Dict[str, Any] = {}

        if choice.type == ToolChoiceType.AUTO:
            result["type"] = "auto"
        elif choice.type == ToolChoiceType.NONE:
            result["type"] = "none"
        elif choice.type == ToolChoiceType.ANY:
            result["type"] = "any"
        elif choice.type == ToolChoiceType.SPECIFIC:
            result["type"] = "tool"
            result["name"] = choice.name

        if choice.disable_parallel:
            result["disable_parallel_tool_use"] = True

        return result

    def encode_response(
        self, ir: IRResponse, *, options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Encode IR response to Anthropic Messages format."""
        # Encode content blocks
        content = []
        has_tool_use = False
        for block in ir.content:
            encoded = self._encode_content_block(block)
            if encoded:
                content.append(encoded)
            # Check if any block is a tool_use block
            if isinstance(block, IRToolUseBlock):
                has_tool_use = True

        # Anthropic rejects an empty content array; an empty response is
        # surfaced as a single empty text block (sub2api behavior).
        if not content:
            content = [{"type": "text", "text": ""}]

        # Determine stop_reason:
        # If response contains tool_use blocks, stop_reason should be "tool_use"
        # regardless of the original stop_reason
        if has_tool_use:
            stop_reason = "tool_use"
        else:
            stop_reason = self._map_stop_reason(ir.stop_reason)

        response: Dict[str, Any] = {
            "id": ir.id,
            "type": "message",
            "role": "assistant",
            "model": ir.model,
            "content": content,
            "stop_reason": stop_reason,
        }

        if ir.stop_sequence:
            response["stop_sequence"] = ir.stop_sequence

        # Cost (preserved from OpenAI Responses via metadata["cost"])
        if ir.metadata.get("cost") is not None:
            response["cost"] = ir.metadata["cost"]

        # Usage — all four buckets always present (sub2api wire shape; the
        # cache fields are how multi-turn clients track prompt caching).
        usage = ir.usage or IRUsage()
        response["usage"] = {
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "cache_creation_input_tokens": usage.cache_creation_tokens,
            "cache_read_input_tokens": usage.cache_read_tokens,
        }

        return response

    def encode_stream_event(
        self, ir_event: IRStreamEvent, *, options: Optional[Dict[str, Any]] = None
    ) -> List[Union[Dict[str, Any], str]]:
        """Encode IR stream event to Anthropic Messages format."""
        options = options or {}
        output_format = options.get("output_format", "dict")

        events = []

        if ir_event.type == StreamEventType.MESSAGE_START:
            response = ir_event.response
            events.append(
                self._format_event(
                    "message_start",
                    {
                        "message": {
                            "id": response.id if response else "",
                            "type": "message",
                            "role": "assistant",
                            "model": response.model if response else "",
                            "content": [],
                            "stop_reason": None,
                            "usage": {
                                "input_tokens": 0,
                                "output_tokens": 0,
                                "cache_creation_input_tokens": 0,
                                "cache_read_input_tokens": 0,
                            },
                        }
                    },
                    output_format,
                )
            )

        elif ir_event.type == StreamEventType.CONTENT_BLOCK_START:
            block = ir_event.content_block
            if isinstance(block, IRTextBlock):
                content_block = {"type": "text", "text": ""}
            elif isinstance(block, IRToolUseBlock):
                content_block = {
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": {},
                }
            elif isinstance(block, IRThinkingBlock):
                # signature key always present on thinking blocks — strict
                # clients (Grok Build) reject its absence.
                content_block = {
                    "type": "thinking",
                    "thinking": "",
                    "signature": "",
                }
            elif isinstance(block, IRServerToolUseBlock):
                content_block = {
                    "type": "server_tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                }
            elif isinstance(block, IRWebSearchToolResultBlock):
                content_block = {
                    "type": "web_search_tool_result",
                    "tool_use_id": block.tool_use_id,
                    "content": block.content,
                }
            else:
                content_block = {"type": "text", "text": ""}

            events.append(
                self._format_event(
                    "content_block_start",
                    {"index": ir_event.index, "content_block": content_block},
                    output_format,
                )
            )

        elif ir_event.type == StreamEventType.CONTENT_BLOCK_DELTA:
            if ir_event.delta_type == "text":
                delta = {"type": "text_delta", "text": ir_event.delta_text}
            elif ir_event.delta_type == "input_json":
                delta = {
                    "type": "input_json_delta",
                    "partial_json": ir_event.delta_json,
                }
            elif ir_event.delta_type == "thinking":
                delta = {"type": "thinking_delta", "thinking": ir_event.delta_text}
            elif ir_event.delta_type == "signature":
                delta = {
                    "type": "signature_delta",
                    "signature": ir_event.delta_text or "",
                }
            else:
                delta = {"type": "text_delta", "text": ir_event.delta_text or ""}

            events.append(
                self._format_event(
                    "content_block_delta",
                    {"index": ir_event.index, "delta": delta},
                    output_format,
                )
            )

        elif ir_event.type == StreamEventType.CONTENT_BLOCK_STOP:
            events.append(
                self._format_event(
                    "content_block_stop",
                    {"index": ir_event.index},
                    output_format,
                )
            )

        elif ir_event.type == StreamEventType.MESSAGE_DELTA:
            delta: Dict[str, Any] = {}
            if ir_event.stop_reason:
                delta["stop_reason"] = self._map_stop_reason(ir_event.stop_reason)
            if ir_event.stop_sequence:
                delta["stop_sequence"] = ir_event.stop_sequence

            # Usage — all four buckets always present (sub2api wire shape).
            usage_obj = ir_event.usage or IRUsage()
            usage: Dict[str, Any] = {
                "input_tokens": usage_obj.input_tokens,
                "output_tokens": usage_obj.output_tokens,
                "cache_creation_input_tokens": usage_obj.cache_creation_tokens,
                "cache_read_input_tokens": usage_obj.cache_read_tokens,
            }

            events.append(
                self._format_event(
                    "message_delta",
                    {"delta": delta, "usage": usage},
                    output_format,
                )
            )

        elif ir_event.type == StreamEventType.DONE:
            events.append(self._format_event("message_stop", {}, output_format))

        elif ir_event.type == StreamEventType.PING:
            events.append(self._format_event("ping", {}, output_format))

        elif ir_event.type == StreamEventType.ERROR:
            events.append(
                self._format_event(
                    "error",
                    {
                        "error": {
                            "type": ir_event.error_type or "error",
                            "message": ir_event.error_message or "Unknown error",
                        }
                    },
                    output_format,
                )
            )

        return events

    def _format_event(
        self, event_type: str, data: Dict[str, Any], output_format: str
    ) -> Union[Dict[str, Any], str]:
        """Format event based on output format."""
        data["type"] = event_type
        if output_format == "sse":
            return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"
        return data

    def _map_role(self, role: Role) -> str:
        """Map IR role to Anthropic role."""
        role_map = {
            Role.USER: "user",
            Role.ASSISTANT: "assistant",
            Role.TOOL: "user",  # Tool results go as user messages
        }
        return role_map.get(role, "user")

    def _map_stop_reason(self, reason: Optional[StopReason]) -> Optional[str]:
        """Map IR stop reason to Anthropic stop reason."""
        if not reason:
            return None
        reason_map = {
            StopReason.END_TURN: "end_turn",
            StopReason.MAX_TOKENS: "max_tokens",
            StopReason.STOP_SEQUENCE: "stop_sequence",
            StopReason.TOOL_USE: "tool_use",
            StopReason.CONTENT_FILTER: "refusal",
            StopReason.ERROR: "end_turn",
        }
        return reason_map.get(reason, "end_turn")
