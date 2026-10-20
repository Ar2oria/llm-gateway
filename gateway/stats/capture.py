"""Extract token usage from upstream responses (non-streaming JSON and SSE).
Pure scanning — never modifies the forwarded bytes."""

from __future__ import annotations

import json
from dataclasses import dataclass


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    output_tokens: int = 0

    @property
    def total(self) -> int:
        return (self.input_tokens + self.cache_read_tokens
                + self.cache_write_tokens + self.output_tokens)


def _int(value: object) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, (int, float)):
        return int(value)
    return 0


def _usage_from_anthropic(usage: dict) -> Usage | None:
    input_tokens = _int(usage.get("input_tokens"))
    output_tokens = _int(usage.get("output_tokens"))
    cache_read = _int(usage.get("cache_read_input_tokens"))
    cache_write = _int(usage.get("cache_creation_input_tokens"))
    # Responses API semantics: input_tokens INCLUDES cached tokens, with the
    # split under input_tokens_details (Anthropic input_tokens excludes them).
    details = usage.get("input_tokens_details")
    if isinstance(details, dict) and not cache_read:
        cached = _int(details.get("cached_tokens"))
        if cached:
            cache_read = cached
            input_tokens = max(0, input_tokens - cached)
    if any((input_tokens, output_tokens, cache_read, cache_write)):
        return Usage(input_tokens=input_tokens, cache_read_tokens=cache_read,
                     cache_write_tokens=cache_write, output_tokens=output_tokens)
    return None


def _usage_from_openai(usage: dict) -> Usage | None:
    prompt = _int(usage.get("prompt_tokens"))
    completion = _int(usage.get("completion_tokens"))
    cached = _int(usage.get("cached_tokens"))
    details = usage.get("prompt_tokens_details")
    if isinstance(details, dict):
        cached = _int(details.get("cached_tokens"))
    cached = cached or _int(usage.get("prompt_cache_hit_tokens"))  # DeepSeek-style
    # Cache creation (write) is reported by some relays under several names:
    # prompt_tokens_details.cached_creation_tokens, top-level cached_creation_tokens,
    # or buzzai-style claude_cache_creation_{5_m,1_h}_tokens.
    creation = 0
    if isinstance(details, dict):
        creation = _int(details.get("cached_creation_tokens"))
    creation = creation or _int(usage.get("cached_creation_tokens"))
    if not creation:
        creation = (_int(usage.get("claude_cache_creation_5_m_tokens"))
                    or _int(usage.get("claude_cache_creation_1_h_tokens")))
    if any((prompt, completion, cached, creation)):
        return Usage(
            input_tokens=max(0, prompt - cached - creation),
            cache_read_tokens=cached,
            cache_write_tokens=creation,
            output_tokens=completion,
        )
    return None


def _extract_from_usage(usage: dict) -> Usage | None:
    # Prefer OpenAI semantics when the provider says so (usage_semantic) or when
    # OpenAI keys are present — some relays emit BOTH key families and the
    # OpenAI numbers (prompt_tokens/completion_tokens) are authoritative there.
    if usage.get("usage_semantic") == "openai" or "prompt_tokens" in usage:
        return _usage_from_openai(usage) or _usage_from_anthropic(usage)
    return _usage_from_anthropic(usage) or _usage_from_openai(usage)


def extract_usage(body: dict) -> Usage:
    """Extract usage from a parsed JSON response body (format-agnostic)."""
    usage = body.get("usage")
    if not isinstance(usage, dict):
        # Some providers nest usage under choices[0].
        choices = body.get("choices")
        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
            usage = choices[0].get("usage")
    if not isinstance(usage, dict):
        return Usage()
    found = _extract_from_usage(usage)
    return found or Usage()


def _is_content_event(data: dict) -> bool:
    """True when an SSE event carries actual model output (not protocol noise).

    Anthropic: content_block_delta with text_delta / thinking_delta /
    input_json_delta payloads (message_start and content_block_start arrive
    before generation).
    OpenAI: a choices[].delta with non-empty content or reasoning_content
    (role-only opening deltas arrive before generation).
    Responses: any "response.*.delta" event (output_text, reasoning text,
    function-call arguments) — response.created/completed are protocol events.
    For reasoning models the encrypted thinking is not streamed as deltas,
    so the first output_item (reasoning/message/function_call) is counted as
    content start to avoid TTFT being delayed until the final tool delta.
    """
    etype = data.get("type")
    if etype == "response.output_item.added":
        return True
    if isinstance(etype, str) and etype.startswith("response.") and etype.endswith(".delta"):
        return True
    if etype == "content_block_delta":
        delta = data.get("delta")
        if isinstance(delta, dict):
            if any(delta.get(k) for k in ("text", "thinking", "partial_json")):
                return True
        return False
    if etype in ("message_start", "content_block_start", "content_block_stop",
                 "message_delta", "message_stop", "ping"):
        return False
    choices = data.get("choices")
    if isinstance(choices, list):
        for choice in choices:
            if isinstance(choice, dict):
                delta = choice.get("delta")
                if isinstance(delta, dict):
                    if delta.get("content") or delta.get("reasoning_content"):
                        return True
    return False


class SseUsageScanner:
    """Scan raw SSE chunks for usage, forwarding bytes unchanged.

    OpenAI: the final chunk carries a top-level `usage` object (cumulative).
    Anthropic: `message_start` carries input/cache counts in message.usage and
    the last `message_delta` carries the cumulative output_tokens.

    has_content becomes True once the first *content* event is seen — the
    point where the model actually starts emitting output (content deltas,
    thinking deltas, tool-input deltas), as opposed to protocol-opening
    events (message_start, role-only deltas) that arrive before generation.
    Used for TTFT measurement.
    """

    def __init__(self) -> None:
        self._pending = b""
        self.usage = Usage()
        self.has_content = False

    def feed(self, chunk: bytes) -> None:
        self._pending += chunk
        lines = self._pending.split(b"\n")
        self._pending = lines.pop()
        for line in lines:
            self._scan_line(line)

    def flush(self) -> None:
        """Scan the residual line left at stream end (no trailing newline)."""
        if self._pending:
            self._scan_line(self._pending)
            self._pending = b""

    def _scan_line(self, line: bytes) -> None:
        if not line.startswith(b"data:"):
            return
        payload = line[5:].strip()
        if not payload or payload == b"[DONE]":
            return
        try:
            data = json.loads(payload)
        except ValueError:
            return
        if not isinstance(data, dict):
            return
        if not self.has_content and _is_content_event(data):
            self.has_content = True
        if data.get("type") == "message_start":
            message = data.get("message")
            if isinstance(message, dict):
                self._merge(extract_usage(message))
        elif data.get("type") == "message_delta":
            self._merge(extract_usage(data))
        elif data.get("type") == "response.completed":
            # Responses API: usage is nested under response.usage
            response = data.get("response")
            if isinstance(response, dict):
                self._merge(extract_usage(response))
        else:
            self._merge(extract_usage(data))

    def _merge(self, candidate: Usage) -> None:
        if candidate.total == 0:
            return
        self.usage = Usage(
            input_tokens=candidate.input_tokens or self.usage.input_tokens,
            cache_read_tokens=candidate.cache_read_tokens or self.usage.cache_read_tokens,
            cache_write_tokens=candidate.cache_write_tokens or self.usage.cache_write_tokens,
            output_tokens=candidate.output_tokens or self.usage.output_tokens,
        )
