"""
OpenAI Chat Completions (Classic) Encoder/Decoder

Converts between OpenAI Chat Completions format and the Intermediate Representation.
"""

import json
import re
import secrets
import time
from typing import Any, Dict, List, Optional, Tuple, Union

from .ir import (
    ImageSourceType,
    IRAudioBlock,
    IRContentBlock,
    IRGenerationConfig,
    IRImageBlock,
    IRMessage,
    IRRequest,
    IRResponse,
    IRResponseFormat,
    IRStreamEvent,
    IRTextBlock,
    IRThinkingBlock,
    IRThinkingConfig,
    IRToolChoice,
    IRToolDeclaration,
    IRToolResultBlock,
    IRToolUseBlock,
    IRUsage,
    Role,
    StopReason,
    StreamEventType,
    ToolChoiceType,
)
from .openai_responses import (
    MIN_MAX_OUTPUT_TOKENS,
    _dump_compact,
    _is_reasoning_model,
    _normalize_tool_parameters,
    _sanitize_read_tool_input,
)


class OpenAIChatDecoder:
    """Decodes OpenAI Chat Completions format to IR."""

    def __init__(self):
        self._reset_stream_state()

    def _reset_stream_state(self):
        """Reset per-stream state (sub2api's ChatCompletionsToAnthropicStreamState).

        Terminal events (message_delta + message_stop) are NOT emitted during
        streaming — the finish reason and usage are captured in state and the
        caller emits them via :meth:`finalize` when the stream ends (sub2api
        contract: the service layer always calls Finalize, clean or not).
        """
        # Whether message_start / the terminal sequence have been emitted.
        self._s_started: bool = False
        self._s_stopped: bool = False
        # Current content block lifecycle.
        self._s_block_index: int = 0
        self._s_block_open: bool = False
        self._s_block_type: Optional[str] = None  # text | thinking | tool_use
        self._s_has_tool_call: bool = False
        self._s_refused: bool = False
        # Tool calls keyed by the upstream tool_call index. The block
        # announcement is deferred until the tool's name arrives; argument
        # fragments seen before the announcement are buffered and flushed
        # with it (some upstreams stream id/arguments before the name).
        self._s_tool_block: Dict[int, int] = {}
        self._s_tool_announced: Dict[int, bool] = {}
        self._s_tool_names: Dict[int, str] = {}
        self._s_pending_tool_id: Dict[int, str] = {}
        self._s_pending_tool_args: Dict[int, str] = {}
        self._s_tool_args_seen: bool = False
        # Terminal state.
        self._s_finish_reason: Optional[str] = None
        self._s_usage: Optional[IRUsage] = None
        self._s_response_id: str = ""
        self._s_model: str = ""

    def decode_request(self, payload: Dict[str, Any]) -> IRRequest:
        """Decode an OpenAI Chat request to IR."""
        ir = IRRequest(
            model=payload.get("model", ""),
            stream=payload.get("stream", False),
        )

        # Decode messages
        messages = payload.get("messages", [])
        ir.messages, ir.system = self._decode_messages(messages)

        # Decode generation config
        ir.generation_config = self._decode_generation_config(payload)

        # Decode tools
        if "tools" in payload:
            ir.tools = self._decode_tools(payload["tools"])

        # Decode tool choice
        if "tool_choice" in payload:
            ir.tool_choice = self._decode_tool_choice(payload["tool_choice"])

        # Decode response format
        if "response_format" in payload:
            ir.response_format = self._decode_response_format(
                payload["response_format"]
            )

        # Reasoning effort (sub2api forwards it as Responses reasoning.effort)
        if payload.get("reasoning_effort"):
            ir.thinking = IRThinkingConfig(effort=str(payload["reasoning_effort"]))

        # User
        if "user" in payload:
            ir.user = payload["user"]

        # Store unsupported params
        unsupported_keys = [
            "store",
            "stream_options",
            "service_tier",
            "parallel_tool_calls",
        ]
        for key in unsupported_keys:
            if key in payload:
                ir.unsupported_params[key] = payload[key]

        return ir

    def _decode_messages(
        self, messages: List[Dict[str, Any]]
    ) -> tuple[List[IRMessage], Optional[str]]:
        """Decode messages; system/developer messages stay in the message
        list (position preserved, original role in ``name``) so each encoder
        can apply its target protocol's system semantics (Responses keeps
        the role, Chat merges leading / downgrades mid, Anthropic folds all
        into the top-level system)."""
        ir_messages = []
        system_parts = []

        # Legacy function calling carries no call IDs: assign sequential
        # call_legacy_N ids and pair them with role=function results by name
        # (sub2api legacyFunctionCallIDs).
        used_ids = {
            tc.get("id")
            for msg in messages
            for tc in msg.get("tool_calls", []) or []
            if tc.get("id")
        }
        used_ids.update(
            msg.get("tool_call_id") for msg in messages if msg.get("tool_call_id")
        )
        pending_legacy: Dict[str, List[str]] = {}
        next_legacy = 0

        def assign_legacy(name: str) -> str:
            nonlocal next_legacy
            while True:
                next_legacy += 1
                candidate = f"call_legacy_{next_legacy}"
                if candidate not in used_ids:
                    break
            used_ids.add(candidate)
            pending_legacy.setdefault(name, []).append(candidate)
            return candidate

        def claim_legacy(name: str) -> str:
            queue = pending_legacy.get(name, [])
            if not queue:
                return ""
            pending_legacy[name] = queue[1:]
            return queue[0]

        for msg in messages:
            role = msg.get("role", "user")

            # system/developer → SYSTEM-role messages with role fidelity.
            if role in ("system", "developer"):
                content = msg.get("content", "")
                if isinstance(content, str):
                    texts = [content] if content else []
                elif isinstance(content, list):
                    texts = []
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            if block.get("text"):
                                texts.append(block["text"])
                        elif isinstance(block, str) and block:
                            texts.append(block)
                else:
                    texts = []
                ir_messages.append(
                    IRMessage(
                        role=Role.SYSTEM,
                        name=role,
                        plain_text=isinstance(content, str),
                        content=[IRTextBlock(text=t) for t in texts],
                    )
                )
                continue

            # Decode user/assistant/tool messages
            ir_role = self._map_role(role)
            ir_message = IRMessage(role=ir_role)

            # Decode content
            content = msg.get("content")
            if content is not None:
                ir_message.content = self._decode_content(content, msg)
                ir_message.plain_text = isinstance(content, str)

            # Decode tool calls (for assistant messages)
            if "tool_calls" in msg:
                for tc in msg["tool_calls"]:
                    tool_use = IRToolUseBlock(
                        id=tc.get("id", ""),
                        name=tc.get("function", {}).get("name", ""),
                        input=self._parse_json_safely(
                            tc.get("function", {}).get("arguments", "{}")
                        ),
                    )
                    ir_message.content.append(tool_use)

            # Legacy function_call on the assistant message → tool use with a
            # synthesized call id.
            if role == "assistant" and msg.get("function_call") and not msg.get(
                "tool_calls"
            ):
                fc = msg["function_call"]
                ir_message.content.append(
                    IRToolUseBlock(
                        id=assign_legacy(fc.get("name", "")),
                        name=fc.get("name", ""),
                        input=self._parse_json_safely(
                            fc.get("arguments", "{}")
                        ),
                    )
                )

            # Plaintext reasoning on the assistant message → thinking block
            # (replayed as reasoning_content / <thinking> by the encoders).
            reasoning = msg.get("reasoning_content") or msg.get("reasoning")
            if reasoning and role == "assistant":
                ir_message.content.insert(0, IRThinkingBlock(thinking=reasoning))

            # Handle tool message (tool result)
            if role == "tool":
                tool_result = IRToolResultBlock(
                    tool_use_id=msg.get("tool_call_id", ""),
                    content=msg.get("content", ""),
                )
                ir_message.content = [tool_result]

            # Legacy function result (role=function) → tool result whose id
            # was assigned to the matching legacy call (fallback: name).
            if role == "function":
                call_id = msg.get("tool_call_id", "") or claim_legacy(
                    msg.get("name", "")
                )
                tool_result = IRToolResultBlock(
                    tool_use_id=call_id or msg.get("name", ""),
                    content=msg.get("content", ""),
                )
                ir_message.content = [tool_result]

            # Name
            if "name" in msg:
                ir_message.name = msg["name"]

            ir_messages.append(ir_message)

        system = "\n\n".join(system_parts) if system_parts else None
        return ir_messages, system

    def _decode_content(
        self, content: Union[str, List[Dict[str, Any]]], msg: Dict[str, Any]
    ) -> List[IRContentBlock]:
        """Decode message content to IR blocks."""
        blocks: List[IRContentBlock] = []

        if isinstance(content, str):
            if content:
                blocks.append(IRTextBlock(text=content))
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, str):
                    blocks.append(IRTextBlock(text=item))
                elif isinstance(item, dict):
                    block = self._decode_content_block(item)
                    if block:
                        blocks.append(block)

        return blocks

    def _decode_content_block(self, block: Dict[str, Any]) -> Optional[IRContentBlock]:
        """Decode a single content block."""
        block_type = block.get("type", "text")

        if block_type == "text":
            return IRTextBlock(text=block.get("text", ""))

        elif block_type == "image_url":
            image_url = block.get("image_url", {})
            if isinstance(image_url, str):
                url = image_url
                detail = None
            else:
                url = image_url.get("url", "")
                detail = image_url.get("detail")

            # Check if it's a data URL (base64)
            if url.startswith("data:"):
                media_type, base64_data = self._parse_data_url(url)
                return IRImageBlock(
                    source_type=ImageSourceType.BASE64,
                    base64_data=base64_data,
                    media_type=media_type,
                    detail=detail,
                )
            else:
                return IRImageBlock(
                    source_type=ImageSourceType.URL,
                    url=url,
                    detail=detail,
                )

        elif block_type == "input_audio":
            audio_data = block.get("input_audio", {})
            return IRAudioBlock(
                source_type="base64",
                data=audio_data.get("data"),
                format=audio_data.get("format"),
            )

        return None

    def _decode_generation_config(self, payload: Dict[str, Any]) -> IRGenerationConfig:
        """Decode generation configuration."""
        config = IRGenerationConfig()

        if "temperature" in payload:
            config.temperature = payload["temperature"]
        if "top_p" in payload:
            config.top_p = payload["top_p"]
        if "max_tokens" in payload:
            config.max_tokens = payload["max_tokens"]
        if "max_completion_tokens" in payload:
            config.max_tokens = payload["max_completion_tokens"]
        if "stop" in payload:
            stop = payload["stop"]
            if isinstance(stop, str):
                config.stop_sequences = [stop]
            else:
                config.stop_sequences = stop
        return config

    def _decode_tools(self, tools: List[Dict[str, Any]]) -> List[IRToolDeclaration]:
        """Decode tool declarations."""
        ir_tools = []
        for tool in tools:
            if tool.get("type") == "function":
                func = tool.get("function", {})
                ir_tools.append(
                    IRToolDeclaration(
                        name=func.get("name", ""),
                        description=func.get("description"),
                        parameters=func.get("parameters", {}),
                        strict=func.get("strict", False),
                    )
                )
        return ir_tools

    def _decode_tool_choice(
        self, tool_choice: Union[str, Dict[str, Any]]
    ) -> IRToolChoice:
        """Decode tool choice configuration."""
        if isinstance(tool_choice, str):
            if tool_choice == "auto":
                return IRToolChoice(type=ToolChoiceType.AUTO)
            elif tool_choice == "none":
                return IRToolChoice(type=ToolChoiceType.NONE)
            elif tool_choice == "required":
                return IRToolChoice(type=ToolChoiceType.ANY)
        elif isinstance(tool_choice, dict):
            if tool_choice.get("type") == "function":
                name = tool_choice.get("function", {}).get("name", "")
                return IRToolChoice(type=ToolChoiceType.SPECIFIC, name=name)
        return IRToolChoice(type=ToolChoiceType.AUTO)

    def _decode_response_format(
        self, response_format: Dict[str, Any]
    ) -> IRResponseFormat:
        """Decode response format configuration."""
        fmt_type = response_format.get("type", "text")
        ir_format = IRResponseFormat(type=fmt_type)

        if fmt_type == "json_schema":
            schema = response_format.get("json_schema", {})
            ir_format.json_schema = schema.get("schema")
            ir_format.schema_name = schema.get("name")
            ir_format.strict = schema.get("strict", False)

        return ir_format

    def decode_response(self, payload: Dict[str, Any]) -> IRResponse:
        """Decode an OpenAI Chat response to IR (sub2api C→A semantics)."""
        ir = IRResponse(
            id=payload.get("id", ""),
            model=payload.get("model", ""),
            created=payload.get("created"),
        )

        # Decode first choice (primary response)
        choices = payload.get("choices", [])
        if choices:
            choice = choices[0]
            ir.choice_index = choice.get("index", 0)

            # Decode message content
            message = choice.get("message", {})

            # Decode plaintext reasoning -> thinking block. Providers use
            # different field names: reasoning_content (hy3-free) or
            # reasoning (mimo) — both are plaintext strings.
            reasoning = message.get("reasoning_content") or message.get("reasoning")
            has_reasoning = bool(reasoning)
            if reasoning:
                ir.content.append(IRThinkingBlock(thinking=reasoning))

            content = message.get("content")
            # Refusal (content stays null when the model refuses): surface
            # the text — Anthropic has no refusal block, so it becomes a
            # plain text block. Dropping it yields an empty message.
            refusal = message.get("refusal")
            if refusal:
                content = (content or "") + refusal

            # Decode tool calls
            tool_calls = message.get("tool_calls", []) or []
            for tc in tool_calls:
                func = tc.get("function", {})
                arguments = func.get("arguments") or ""
                if not arguments.strip():
                    arguments = "{}"
                arguments = _sanitize_read_tool_input(
                    func.get("name", ""), arguments
                )
                ir.content.append(
                    IRToolUseBlock(
                        id=tc.get("id", ""),
                        name=func.get("name", ""),
                        input=self._parse_json_safely(arguments),
                    )
                )

            # Text block: emitted when there is text, or when the message
            # carries no tool calls (an all-empty content array is invalid).
            # DeepSeek reasoning-only fallback: no text and no tool calls →
            # surface the reasoning as visible text so the turn isn't empty.
            if not content and has_reasoning and not tool_calls:
                content = reasoning
            if content or not tool_calls:
                ir.content.append(IRTextBlock(text=content or ""))

            # Decode finish reason: length → max_tokens, tool_calls →
            # tool_use, else derived from the blocks (sub2api
            # chatFinishReasonToAnthropicStopReason).
            finish_reason = choice.get("finish_reason")
            if finish_reason == "length":
                ir.stop_reason = StopReason.MAX_TOKENS
            elif finish_reason == "tool_calls":
                ir.stop_reason = StopReason.TOOL_USE
            elif tool_calls:
                ir.stop_reason = StopReason.TOOL_USE
            else:
                ir.stop_reason = StopReason.END_TURN
            # A refusal marks the stop as content-filter semantics
            # (Anthropic's stop_reason "refusal") rather than end_turn.
            if refusal:
                ir.stop_reason = StopReason.CONTENT_FILTER
        else:
            # Empty choices still report a completed turn.
            ir.stop_reason = StopReason.END_TURN

        # Decode usage
        usage = payload.get("usage")
        if usage:
            ir.usage = self._usage_from_openai(usage)
            ir.usage.total_tokens = usage.get("total_tokens")
            completion_details = usage.get("completion_tokens_details", {})
            if completion_details:
                ir.usage.reasoning_tokens = completion_details.get(
                    "reasoning_tokens", 0
                )

        # Preserve the upstream cost (compatible endpoints report it; the
        # Anthropic encoder emits it as the response's cost field).
        if payload.get("cost") is not None:
            ir.metadata["cost"] = payload["cost"]

        return ir

    def _usage_from_openai(self, usage: Dict[str, Any]) -> IRUsage:
        """Map OpenAI chat usage to IR usage.

        OpenAI prompt_tokens INCLUDES cached tokens (split under
        prompt_tokens_details); Anthropic input_tokens excludes cache reads
        AND cache creation — subtract so clients don't double-count.
        ``cache_write_tokens`` and ``cache_creation_tokens`` are alternate
        spellings of the same quantity (sub2api prefers write).
        """
        details = usage.get("prompt_tokens_details", {})
        cached = details.get("cached_tokens", 0) if isinstance(details, dict) else 0
        creation = 0
        if isinstance(details, dict):
            creation = details.get("cache_write_tokens") or details.get(
                "cache_creation_tokens"
            ) or 0
        return IRUsage(
            input_tokens=max(0, usage.get("prompt_tokens", 0) - cached - creation),
            output_tokens=usage.get("completion_tokens", 0),
            cache_read_tokens=cached,
            cache_creation_tokens=creation,
        )

    def decode_stream_event(
        self, event: Union[Dict[str, Any], str]
    ) -> List[IRStreamEvent]:
        """Decode a streaming chunk to IR events (sub2api C→A contract).

        Terminal events are deferred to :meth:`finalize`; [DONE] carries no
        semantics of its own.
        """
        # Handle SSE text format
        if isinstance(event, str):
            event = self._parse_sse_line(event)
            if event is None:
                return []

        if event == "[DONE]":
            return []

        ir_events: List[IRStreamEvent] = []

        if event.get("id"):
            self._s_response_id = event["id"]
        if not self._s_model and event.get("model"):
            self._s_model = event["model"]

        # Usage in a streaming chunk (include_usage) may arrive on the finish
        # chunk or its own chunk — capture for the finalize message_delta.
        usage = event.get("usage")
        if usage:
            self._s_usage = self._usage_from_openai(usage)

        if not self._s_started:
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

        for choice in event.get("choices", []):
            delta = choice.get("delta") or {}

            # Reasoning content → thinking block.
            reasoning = delta.get("reasoning_content") or delta.get("reasoning")
            if reasoning:
                self._s_ensure_block("thinking", ir_events)
                ir_events.append(
                    IRStreamEvent(
                        type=StreamEventType.CONTENT_BLOCK_DELTA,
                        index=self._s_block_index,
                        delta_type="thinking",
                        delta_text=reasoning,
                    )
                )

            # Refusal (content null): route the text through the text-delta
            # machinery so the client sees it, and remember it so the final
            # stop reason maps to Anthropic's refusal semantics.
            content = delta.get("content")
            if delta.get("refusal"):
                self._s_refused = True
                content = (content or "") + delta["refusal"]

            # Text content → text block (closes any open thinking block).
            if content:
                if self._s_block_open and self._s_block_type == "thinking":
                    self._s_close_block(ir_events)
                self._s_ensure_block("text", ir_events)
                ir_events.append(
                    IRStreamEvent(
                        type=StreamEventType.CONTENT_BLOCK_DELTA,
                        index=self._s_block_index,
                        delta_type="text",
                        delta_text=content,
                    )
                )

            # Tool calls → tool_use blocks (deferred announcement).
            for tc in delta.get("tool_calls", []) or []:
                self._s_handle_tool_call(tc, ir_events)

            finish_reason = choice.get("finish_reason")
            if finish_reason:
                self._s_finish_reason = finish_reason

        return ir_events

    def finalize(self) -> List[IRStreamEvent]:
        """Emit the terminal Anthropic events (close open blocks +
        message_delta + message_stop) when the stream ends — sub2api's
        FinalizeChatCompletionsAnthropicStream. Idempotent."""
        if self._s_stopped:
            return []

        ir_events: List[IRStreamEvent] = []
        if not self._s_started:
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

        # Announce tools whose name never arrived so their buffered arguments
        # are not silently dropped.
        if self._s_pending_tool_id:
            for idx in sorted(self._s_pending_tool_id):
                call_id = self._s_pending_tool_id[idx]
                self._s_close_block(ir_events)
                self._s_announce_tool(idx, call_id, "", ir_events)

        self._s_close_block(ir_events)

        stop_reason = self._terminal_stop_reason()
        ir_events.append(
            IRStreamEvent(
                type=StreamEventType.MESSAGE_DELTA,
                stop_reason=stop_reason,
                usage=self._s_usage,
            )
        )
        ir_events.append(IRStreamEvent(type=StreamEventType.DONE))
        self._s_stopped = True
        return ir_events

    def _terminal_stop_reason(self) -> StopReason:
        """Finish reason → Anthropic stop reason (sub2api
        ccFinishReasonToAnthropicStopReason), with refusal semantics for
        streams that carried a refusal delta."""
        reason = self._s_finish_reason
        if reason == "length":
            return StopReason.MAX_TOKENS
        if reason == "tool_calls":
            return StopReason.TOOL_USE
        if self._s_refused:
            return StopReason.CONTENT_FILTER
        if self._s_has_tool_call:
            return StopReason.TOOL_USE
        return StopReason.END_TURN

    # --- stream block helpers (sub2api ensure/close/announce) -----------

    def _s_ensure_block(
        self, block_type: str, ir_events: List[IRStreamEvent]
    ) -> List[IRStreamEvent]:
        """Open a content block of the given type if none is open."""
        if self._s_block_open and self._s_block_type == block_type:
            return []
        self._s_close_block(ir_events)
        self._s_block_open = True
        self._s_block_type = block_type
        if block_type == "thinking":
            block: IRContentBlock = IRThinkingBlock()
        else:
            block = IRTextBlock()
        ir_events.append(
            IRStreamEvent(
                type=StreamEventType.CONTENT_BLOCK_START,
                index=self._s_block_index,
                content_block=block,
            )
        )
        return ir_events

    def _s_close_block(self, ir_events: List[IRStreamEvent]) -> List[IRStreamEvent]:
        """Close the open content block. A tool_use block that streamed no
        argument delta gets a final input_json_delta "{}" — some clients
        assemble tool input exclusively from deltas."""
        if not self._s_block_open:
            return []
        idx = self._s_block_index
        if self._s_block_type == "tool_use" and not self._s_tool_args_seen:
            ir_events.append(
                IRStreamEvent(
                    type=StreamEventType.CONTENT_BLOCK_DELTA,
                    index=idx,
                    delta_type="input_json",
                    delta_json="{}",
                )
            )
        ir_events.append(
            IRStreamEvent(type=StreamEventType.CONTENT_BLOCK_STOP, index=idx)
        )
        self._s_block_open = False
        self._s_block_index += 1
        self._s_block_type = None
        self._s_tool_args_seen = False
        return ir_events

    def _s_handle_tool_call(
        self, tc: Dict[str, Any], ir_events: List[IRStreamEvent]
    ) -> List[IRStreamEvent]:
        """Process one upstream tool_call delta. The block announcement is
        deferred until the tool's name arrives; argument fragments seen
        before the announcement are buffered and flushed with it."""
        idx = tc.get("index", 0)
        func = tc.get("function") or {}
        name = func.get("name", "")
        args = func.get("arguments", "")

        if idx not in self._s_tool_announced:
            # New tool call: it ends whatever block is currently streaming.
            self._s_close_block(ir_events)
            self._s_has_tool_call = True

            call_id = tc.get("id", "")
            if name:
                self._s_announce_tool(idx, call_id, name, ir_events)
            else:
                self._s_tool_announced[idx] = False
                self._s_pending_tool_id[idx] = call_id
        elif not self._s_tool_announced[idx] and name:
            # Deferred announcement: the name has arrived.
            call_id = self._s_pending_tool_id.get(idx, "")
            if tc.get("id"):
                call_id = tc["id"]
            self._s_close_block(ir_events)
            self._s_announce_tool(idx, call_id, name, ir_events)

        if args:
            if self._s_tool_announced.get(idx):
                block_idx = self._s_tool_block[idx]
                if self._s_block_open and block_idx == self._s_block_index:
                    self._s_tool_args_seen = True
                ir_events.append(
                    IRStreamEvent(
                        type=StreamEventType.CONTENT_BLOCK_DELTA,
                        index=block_idx,
                        delta_type="input_json",
                        delta_json=args,
                    )
                )
            else:
                self._s_pending_tool_args[idx] = (
                    self._s_pending_tool_args.get(idx, "") + args
                )

        return ir_events

    def _s_announce_tool(
        self,
        idx: int,
        call_id: str,
        name: str,
        ir_events: List[IRStreamEvent],
    ) -> List[IRStreamEvent]:
        """Assign the next block index to the tool, emit its
        content_block_start, and flush any buffered argument fragments."""
        block_idx = self._s_block_index
        self._s_tool_block[idx] = block_idx
        self._s_tool_announced[idx] = True
        self._s_tool_names[idx] = name
        self._s_tool_args_seen = False
        self._s_block_open = True
        self._s_block_type = "tool_use"
        self._s_pending_tool_id.pop(idx, None)

        ir_events.append(
            IRStreamEvent(
                type=StreamEventType.CONTENT_BLOCK_START,
                index=block_idx,
                content_block=IRToolUseBlock(id=call_id, name=name),
            )
        )
        pending = self._s_pending_tool_args.pop(idx, "")
        if pending:
            self._s_tool_args_seen = True
            ir_events.append(
                IRStreamEvent(
                    type=StreamEventType.CONTENT_BLOCK_DELTA,
                    index=block_idx,
                    delta_type="input_json",
                    delta_json=pending,
                )
            )
        return ir_events

    def _parse_sse_line(self, line: str) -> Optional[Union[Dict[str, Any], str]]:
        """Parse an SSE data line."""
        line = line.strip()
        if not line or line.startswith(":"):
            return None
        if line.startswith("data: "):
            data = line[6:]
            if data == "[DONE]":
                return "[DONE]"
            try:
                return json.loads(data)
            except json.JSONDecodeError:
                return None
        return None

    def _map_role(self, role: str) -> Role:
        """Map OpenAI role to IR role."""
        role_map = {
            "system": Role.SYSTEM,
            "developer": Role.SYSTEM,
            "user": Role.USER,
            "assistant": Role.ASSISTANT,
            "tool": Role.TOOL,
            "function": Role.TOOL,
        }
        return role_map.get(role, Role.USER)

    def _parse_data_url(self, data_url: str) -> tuple[str, str]:
        """Parse a data URL into media type and base64 data."""
        match = re.match(r"data:([^;]+);base64,(.+)", data_url)
        if match:
            return match.group(1), match.group(2)
        return "application/octet-stream", data_url

    def _parse_json_safely(self, json_str: str) -> Dict[str, Any]:
        """Safely parse JSON string."""
        try:
            return json.loads(json_str) if json_str else {}
        except json.JSONDecodeError:
            return {}


class OpenAIChatEncoder:
    """Encodes IR to OpenAI Chat Completions format.

    Wire behavior mirrors sub2api's bridges: A→C requests follow
    AnthropicToChatCompletionsRequest (max floor, parallel_tool_calls,
    reasoning-model sampling drop, tool_choice gating, reasoning_content on
    tool-call assistant messages only); R→C responses/streams follow
    ResponsesToChatCompletions / ResponsesEventToChatChunks (id passthrough,
    finish-chunk shape, usage gated on the ``include_usage`` option).

    Stream encoding is stateful (chunk id/model/created cached from
    MESSAGE_START) — a fresh instance must be used per stream.
    """

    def __init__(self) -> None:
        self._e_chunk_id: str = ""
        self._e_model: str = ""
        self._e_created: Optional[int] = None

    # ------------------------------------------------------------------
    # Request encoding (IR → Chat request; sub2api A→C semantics)
    # ------------------------------------------------------------------

    def encode_request(
        self, ir: IRRequest, *, options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Encode IR request to OpenAI Chat format."""
        options = options or {}
        messages = self._encode_messages(
            ir.messages, system=ir.system, options=options
        )
        tools = self._encode_tools(ir.tools, options=options)
        payload: Dict[str, Any] = {
            "model": ir.model,
            "messages": messages,
        }

        # Stream
        if ir.stream:
            payload["stream"] = True
            payload["stream_options"] = {
                "include_usage": True
            }

        # Reasoning effort (from Anthropic output_config.effort / thinking;
        # sub2api always resolves one for Anthropic sources).
        if ir.thinking and ir.thinking.effort:
            payload["reasoning_effort"] = ir.thinking.effort

        # Sampling params: reasoning models (gpt-5+) reject them with a 400.
        if not _is_reasoning_model(ir.model):
            if ir.generation_config.temperature is not None:
                payload["temperature"] = ir.generation_config.temperature
            if ir.generation_config.top_p is not None:
                payload["top_p"] = ir.generation_config.top_p

        # max_tokens → max_completion_tokens, floored at 128 (very small
        # values cause upstream API errors).
        if ir.generation_config.max_tokens:
            payload["max_completion_tokens"] = max(
                ir.generation_config.max_tokens, MIN_MAX_OUTPUT_TOKENS
            )

        if tools:
            payload["tools"] = tools
            # tool_choice is only forwarded when tools survived the
            # conversion, and a named choice only when it points at a
            # declared tool (upstreams 400 otherwise).
            if ir.tool_choice:
                declared = {
                    t["function"]["name"] for t in tools if t.get("function")
                }
                encoded_choice = self._encode_tool_choice(ir.tool_choice, declared)
                if encoded_choice is not None:
                    payload["tool_choice"] = encoded_choice

        # parallel_tool_calls is forced on for Anthropic sources; other
        # sources pass it through (absent when unset — sub2api semantics).
        if options.get("source") == "anthropic_messages":
            payload["parallel_tool_calls"] = True

        # Response format
        if ir.response_format:
            payload["response_format"] = self._encode_response_format(
                ir.response_format
            )

        return payload

    def _encode_messages(
        self,
        messages: List[IRMessage],
        *,
        system: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Encode IR messages to Chat format (sub2api semantics).

        A→C follows anthropicToChatMessages (reasoning_content rides only on
        tool-call assistant messages); R→C follows
        buildChatMessagesFromItems (reasoning replay across the turn,
        consecutive tool calls merged into one assistant message).
        """
        options = options or {}
        is_responses_source = options.get("source") == "openai_responses"
        result: List[Dict[str, Any]] = []

        # Leading SYSTEM messages (plus the source protocol's own system
        # field) merge into one system message — Chat accepts system content
        # only at the start; later ones downgrade to user.
        leading: List[str] = [system] if system else []
        idx = 0
        while idx < len(messages) and messages[idx].role == Role.SYSTEM:
            for block in messages[idx].content:
                if isinstance(block, IRTextBlock) and block.text.strip():
                    leading.append(block.text)
            idx += 1
        if leading:
            result.append({"role": "system", "content": "\n\n".join(leading)})

        pending_reasoning = ""
        last_turn_reasoning = ""
        pending_images: List[Dict[str, Any]] = []

        for msg in messages[idx:]:
            if msg.role == Role.SYSTEM:
                # Mid-conversation system/developer → user turn.
                texts = [
                    b.text
                    for b in msg.content
                    if isinstance(b, IRTextBlock) and b.text.strip()
                ]
                if texts:
                    result.append(
                        {"role": "user", "content": "\n\n".join(texts)}
                    )
                continue

            if msg.role == Role.TOOL:
                for block in msg.content:
                    if not isinstance(block, IRToolResultBlock):
                        continue
                    text, images = self._tool_result_content(block)
                    result.append(
                        {
                            "role": "tool",
                            "tool_call_id": block.tool_use_id,
                            "content": text,
                        }
                    )
                    pending_images.extend(images)
                # A tool reply ends the pending thinking span; the turn's
                # reasoning survives for later calls in the same turn.
                pending_reasoning = ""
                continue

            if msg.role == Role.USER:
                text_parts: List[str] = []
                image_parts: List[Dict[str, Any]] = []
                for block in msg.content:
                    if isinstance(block, IRTextBlock) and block.text:
                        text_parts.append(block.text)
                        image_parts.append({"type": "text", "text": block.text})
                    elif isinstance(block, IRImageBlock):
                        image_parts.append(self._encode_image_block(block))
                has_image = any(p["type"] == "image_url" for p in image_parts)
                parts = (
                    image_parts + pending_images
                    if has_image or pending_images
                    else []
                )
                pending_images = []
                if parts:
                    result.append({"role": "user", "content": parts})
                elif text_parts:
                    # Text-only content folds into a single string (strict
                    # chat upstreams reject array content).
                    result.append(
                        {"role": "user", "content": "\n\n".join(text_parts)}
                    )
                # A user turn ends the reasoning replay span.
                pending_reasoning = ""
                last_turn_reasoning = ""
                continue

            if msg.role == Role.ASSISTANT:
                thinking_texts = [
                    b.thinking
                    for b in msg.content
                    if isinstance(b, IRThinkingBlock)
                    and not b.is_redacted
                    and b.thinking
                ]
                tool_calls = [
                    self._encode_tool_call(b)
                    for b in msg.content
                    if isinstance(b, IRToolUseBlock)
                ]
                text = "\n\n".join(
                    b.text
                    for b in msg.content
                    if isinstance(b, IRTextBlock) and b.text
                )

                if is_responses_source:
                    # A thinking-only message is the Responses reasoning item:
                    # hold its text for the assistant message it belongs to.
                    if thinking_texts and not text and not tool_calls:
                        pending_reasoning = "\n".join(thinking_texts)
                        if pending_reasoning:
                            last_turn_reasoning = pending_reasoning
                        continue
                    reasoning = pending_reasoning or last_turn_reasoning
                    if tool_calls and result and result[-1].get("role") == "assistant":
                        # Consecutive tool calls merge into one assistant
                        # message (sub2api appendAssistantToolCall).
                        prev = result[-1]
                        prev.setdefault("tool_calls", []).extend(tool_calls)
                        if not prev.get("reasoning_content") and reasoning:
                            prev["reasoning_content"] = reasoning
                    elif text or tool_calls:
                        message: Dict[str, Any] = {"role": "assistant"}
                        if text:
                            message["content"] = text
                        if tool_calls:
                            message["tool_calls"] = tool_calls
                        if reasoning:
                            message["reasoning_content"] = reasoning
                        result.append(message)
                    pending_reasoning = ""
                    continue

                # Anthropic source: reasoning_content rides along with tool
                # calls only, never on a plain assistant text turn.
                if not text and not tool_calls:
                    continue
                message = {"role": "assistant"}
                if text:
                    message["content"] = text
                if tool_calls:
                    if thinking_texts:
                        message["reasoning_content"] = "\n".join(thinking_texts)
                    message["tool_calls"] = tool_calls
                result.append(message)
                continue

        # Images lifted from tool_results that never found a user message
        # still travel as their own user turn.
        if pending_images:
            result.append({"role": "user", "content": pending_images})

        return self._repair_chat_messages(result)

    @staticmethod
    def _repair_chat_messages(
        messages: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Repair tool_call/tool-reply adjacency (sub2api normalizeChatMessages).

        Tool replies are re-emitted right after the assistant message that
        announces them; unanswered tool_calls are dropped (and an assistant
        message left without calls survives only when it has content);
        orphan replies with no announcing assistant are dropped.
        """
        replies: Dict[str, Dict[str, Any]] = {}
        for m in messages:
            if m.get("role") == "tool" and m.get("tool_call_id"):
                replies[m["tool_call_id"]] = m

        out: List[Dict[str, Any]] = []
        for m in messages:
            if m.get("role") == "tool":
                # A bare tool message with no id is a direct passthrough;
                # anything else is re-emitted next to its call.
                if not m.get("tool_call_id"):
                    out.append(m)
                continue
            tool_calls = m.get("tool_calls")
            if tool_calls:
                kept = [
                    tc
                    for tc in tool_calls
                    if tc.get("id") and tc.get("id") in replies
                ]
                if not kept:
                    if m.get("content"):
                        repaired = dict(m)
                        repaired.pop("tool_calls", None)
                        out.append(repaired)
                    continue
                repaired = dict(m)
                repaired["tool_calls"] = kept
                out.append(repaired)
                for tc in kept:
                    out.append(replies[tc["id"]])
                continue
            out.append(m)
        return out

    def _tool_result_content(
        self, block: IRToolResultBlock
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """tool_result content → (tool-message text, image parts lifted into
        a follow-up user message)."""
        if isinstance(block.content, str):
            return (block.content if block.content else "(empty)"), []
        if isinstance(block.content, list):
            texts: List[str] = []
            images: List[Dict[str, Any]] = []
            for b in block.content:
                if isinstance(b, IRTextBlock) and b.text:
                    texts.append(b.text)
                elif isinstance(b, IRImageBlock):
                    images.append(self._encode_image_block(b))
            text = "\n\n".join(texts)
            return (text if text else "(empty)"), images
        return "(empty)", []

    def _encode_image_block(self, block: IRImageBlock) -> Dict[str, Any]:
        """Encode an image block."""
        if block.source_type == ImageSourceType.BASE64:
            media_type = block.media_type or "image/png"
            url = f"data:{media_type};base64,{block.base64_data}"
        else:
            url = block.url or ""

        image_url: Dict[str, Any] = {"url": url}
        if block.detail:
            image_url["detail"] = block.detail

        return {"type": "image_url", "image_url": image_url}

    def _encode_tool_call(self, block: IRToolUseBlock) -> Dict[str, Any]:
        """Encode a tool call."""
        return {
            "id": block.id,
            "type": "function",
            "function": {
                "name": block.name,
                "arguments": _dump_compact(block.input) if block.input else "{}",
            },
        }

    def _encode_tools(
        self,
        tools: List[IRToolDeclaration],
        *,
        options: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Encode tool declarations. Server-side tools (web_search) have no
        Chat equivalent and are dropped (sub2api anthropicToolsToChatTools)."""
        options = options or {}
        result = []
        for tool in tools:
            if tool.type and tool.type.startswith("web_search"):
                continue
            function: Dict[str, Any] = {"name": tool.name}
            if options.get("source") == "anthropic_messages":
                function["strict"] = False
            encoded: Dict[str, Any] = {
                "type": "function",
                "function": function,
            }
            if tool.description:
                encoded["function"]["description"] = tool.description
            encoded["function"]["parameters"] = _normalize_tool_parameters(
                tool.parameters
            )
            result.append(encoded)
        return result

    def _encode_tool_choice(
        self, choice: IRToolChoice, declared: set
    ) -> Optional[Union[str, Dict[str, Any]]]:
        """Encode tool choice; None drops the choice (undeclared tool or
        unknown type — chat upstreams reject both)."""
        if choice.type == ToolChoiceType.AUTO:
            return "auto"
        if choice.type == ToolChoiceType.ANY:
            return "required"
        if choice.type == ToolChoiceType.NONE:
            return "none"
        if choice.type == ToolChoiceType.SPECIFIC:
            if not choice.name or choice.name not in declared:
                return None
            return {
                "type": "function",
                "function": {"name": choice.name},
            }
        return None

    def _encode_response_format(self, fmt: IRResponseFormat) -> Dict[str, Any]:
        """Encode response format."""
        if fmt.type == "json_schema":
            result: Dict[str, Any] = {
                "type": "json_schema",
                "json_schema": {},
            }
            if fmt.schema_name:
                result["json_schema"]["name"] = fmt.schema_name
            if fmt.json_schema:
                result["json_schema"]["schema"] = fmt.json_schema
            if fmt.strict:
                result["json_schema"]["strict"] = True
            return result
        return {"type": fmt.type}

    # ------------------------------------------------------------------
    # Response encoding (IR → Chat response; sub2api R→C semantics)
    # ------------------------------------------------------------------

    def encode_response(
        self, ir: IRResponse, *, options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Encode IR response to OpenAI Chat format."""
        options = options or {}
        message: Dict[str, Any] = {"role": "assistant"}

        text_parts = []
        refusal = ""
        reasoning = ""
        tool_calls = []
        for block in ir.content:
            if isinstance(block, IRTextBlock):
                if block.refusal:
                    refusal += block.text
                else:
                    text_parts.append(block.text)
            elif isinstance(block, IRToolUseBlock):
                tool_calls.append(self._encode_tool_call(block))
            elif isinstance(block, IRThinkingBlock) and block.thinking:
                reasoning += block.thinking

        if text_parts:
            message["content"] = "".join(text_parts)
        if refusal:
            message["refusal"] = refusal
        if reasoning:
            message["reasoning_content"] = reasoning
        if tool_calls:
            message["tool_calls"] = tool_calls

        finish_reason = self._finish_reason(ir, tool_calls)

        model = options.get("model") or ir.model
        response: Dict[str, Any] = {
            "id": ir.id if ir.id else f"chatcmpl-{secrets.token_hex(12)}",
            "object": "chat.completion",
            "created": ir.created or int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": ir.choice_index,
                    "message": message,
                    "finish_reason": finish_reason,
                }
            ],
        }

        if ir.usage:
            response["usage"] = self._usage_to_chat(ir.usage)

        return response

    def _finish_reason(
        self, ir: IRResponse, tool_calls: List[Dict[str, Any]]
    ) -> str:
        """IR stop reason → Chat finish reason (sub2api
        responsesStatusToChatFinishReason, with the content-filter
        incomplete reason preserved via metadata)."""
        if ir.metadata.get("incomplete_reason") == "content_filter":
            return "content_filter"
        if ir.stop_reason == StopReason.MAX_TOKENS:
            return "length"
        if tool_calls:
            return "tool_calls"
        return "stop"

    @staticmethod
    def _usage_to_chat(usage: IRUsage) -> Dict[str, Any]:
        """IR (Anthropic-bucket) usage → Chat usage semantics: prompt_tokens
        re-includes cache reads and cache creation (sub2api
        chatUsageFromResponsesUsage)."""
        total_input = (
            usage.input_tokens + usage.cache_read_tokens + usage.cache_creation_tokens
        )
        out: Dict[str, Any] = {
            "prompt_tokens": total_input,
            "completion_tokens": usage.output_tokens,
            "total_tokens": total_input + usage.output_tokens,
        }
        details: Dict[str, Any] = {}
        if usage.cache_read_tokens:
            details["cached_tokens"] = usage.cache_read_tokens
        if usage.cache_creation_tokens:
            details["cache_creation_tokens"] = usage.cache_creation_tokens
        if details:
            out["prompt_tokens_details"] = details
        if usage.reasoning_tokens:
            out["completion_tokens_details"] = {
                "reasoning_tokens": usage.reasoning_tokens
            }
        return out

    # ------------------------------------------------------------------
    # Stream encoding (IR stream events → Chat chunks, stateful)
    # ------------------------------------------------------------------

    def encode_stream_event(
        self, ir_event: IRStreamEvent, *, options: Optional[Dict[str, Any]] = None
    ) -> List[Union[Dict[str, Any], str]]:
        """Encode IR stream event to OpenAI Chat format."""
        options = options or {}
        output_format = options.get("output_format", "dict")  # dict or sse

        events = []

        if ir_event.type == StreamEventType.MESSAGE_START:
            response = ir_event.response
            if response is not None:
                self._e_chunk_id = response.id or self._e_chunk_id
                self._e_model = response.model or self._e_model
                self._e_created = response.created or self._e_created
            events.append(
                self._format_event(
                    self._delta_chunk({"role": "assistant"}), output_format
                )
            )

        elif ir_event.type == StreamEventType.CONTENT_BLOCK_DELTA:
            if ir_event.delta_type == "text" and ir_event.delta_text:
                events.append(
                    self._format_event(
                        self._delta_chunk({"content": ir_event.delta_text}),
                        output_format,
                    )
                )
            elif ir_event.delta_type == "thinking" and ir_event.delta_text:
                events.append(
                    self._format_event(
                        self._delta_chunk(
                            {"reasoning_content": ir_event.delta_text}
                        ),
                        output_format,
                    )
                )
            elif ir_event.delta_type == "input_json" and ir_event.delta_json:
                events.append(
                    self._format_event(
                        self._delta_chunk(
                            {
                                "tool_calls": [
                                    {
                                        "index": ir_event.index,
                                        "function": {
                                            "arguments": ir_event.delta_json
                                        },
                                    }
                                ]
                            }
                        ),
                        output_format,
                    )
                )

        elif ir_event.type == StreamEventType.CONTENT_BLOCK_START:
            if isinstance(ir_event.content_block, IRToolUseBlock):
                events.append(
                    self._format_event(
                        self._delta_chunk(
                            {
                                "tool_calls": [
                                    {
                                        "index": ir_event.index,
                                        "id": ir_event.content_block.id,
                                        "type": "function",
                                        "function": {
                                            "name": ir_event.content_block.name,
                                            "arguments": "",
                                        },
                                    }
                                ]
                            }
                        ),
                        output_format,
                    )
                )

        elif ir_event.type == StreamEventType.MESSAGE_DELTA:
            if ir_event.stop_reason:
                finish_reason = self._map_stop_reason(ir_event.stop_reason)
                events.append(
                    self._format_event(
                        self._finish_chunk(finish_reason), output_format
                    )
                )
            if ir_event.usage and options.get("include_usage"):
                # Usage chunk gated on the caller's opt-in (sub2api's
                # IncludeUsage stream option); the gateway opts in so stats
                # keep token counts.
                chunk = self._chunk_base()
                chunk["choices"] = []
                chunk["usage"] = self._usage_to_chat(ir_event.usage)
                events.append(self._format_event(chunk, output_format))

        elif ir_event.type == StreamEventType.DONE:
            if output_format == "sse":
                events.append("data: [DONE]\n\n")
            else:
                events.append("[DONE]")

        return events

    def _chunk_base(self) -> Dict[str, Any]:
        """Chunk skeleton reusing the id/model captured on MESSAGE_START."""
        return {
            "id": self._e_chunk_id,
            "object": "chat.completion.chunk",
            "created": self._e_created if self._e_created is not None else int(time.time()),
            "model": self._e_model,
        }

    def _delta_chunk(self, delta: Dict[str, Any]) -> Dict[str, Any]:
        chunk = self._chunk_base()
        chunk["choices"] = [{"index": 0, "delta": delta, "finish_reason": None}]
        return chunk

    def _finish_chunk(self, finish_reason: str) -> Dict[str, Any]:
        chunk = self._chunk_base()
        chunk["choices"] = [
            {
                "index": 0,
                "delta": {"content": ""},
                "finish_reason": finish_reason,
            }
        ]
        return chunk

    def _format_event(
        self, chunk: Dict[str, Any], output_format: str
    ) -> Union[Dict[str, Any], str]:
        """Format event based on output format."""
        if output_format == "sse":
            return f"data: {json.dumps(chunk)}\n\n"
        return chunk

    def _map_role(self, role: Role) -> str:
        """Map IR role to OpenAI role."""
        role_map = {
            Role.SYSTEM: "system",
            Role.USER: "user",
            Role.ASSISTANT: "assistant",
            Role.TOOL: "tool",
        }
        return role_map.get(role, "user")

    def _map_stop_reason(self, reason: Optional[StopReason]) -> Optional[str]:
        """Map IR stop reason to OpenAI finish reason."""
        if not reason:
            return None
        reason_map = {
            StopReason.END_TURN: "stop",
            StopReason.MAX_TOKENS: "length",
            StopReason.STOP_SEQUENCE: "stop",
            StopReason.TOOL_USE: "tool_calls",
            # A refusal maps to a plain stop on Chat targets (sub2api
            # semantics; the content_filter finish comes from the
            # incomplete_reason metadata only).
            StopReason.CONTENT_FILTER: "stop",
            StopReason.ERROR: "stop",
        }
        return reason_map.get(reason, "stop")
