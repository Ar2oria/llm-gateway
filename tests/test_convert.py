"""Tests for gateway.convert — protocol matching + body/SSE conversion."""

import asyncio
import gzip
import json

import pytest

from gateway.convert import (
	ConversionFailed,
	protocol_convert,
	convert_response_body,
	convert_sse_chunks,
	get_convert_stats,
)


# ---------------------------------------------------------------------------
# convert_request_body
# ---------------------------------------------------------------------------

def test_openai_chat_to_anthropic_request_body():
    stats = get_convert_stats("/v1/chat/completions", "anthropic_messages")
    data = protocol_convert(stats, {
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
        ],
        "max_tokens": 50,
        "stream": True,
    })
    assert data["system"] == "sys"
    assert data["messages"] == [{"role": "user", "content": "hi"}]
    assert data["max_tokens"] == 50
    assert data["stream"] is True
    assert data["model"] == "gpt-4o"


def test_anthropic_to_openai_request_body():
    stats = get_convert_stats("/v1/messages", "openai_chat")
    data = protocol_convert(stats, {
        "model": "claude-3",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 100,
        "system": "be nice",
    })
    assert data["messages"][0]["role"] == "system"
    assert data["messages"][0]["content"] == "be nice"
    assert data["messages"][1]["content"] == "hi"
    # max_tokens is floored at 128 (sub2api DefaultMinMaxOutputTokens —
    # very small values cause upstream API errors).
    assert data["max_completion_tokens"] == 128


def test_anthropic_system_role_message_survives_as_user_content():
    """Anthropic role=system messages (sent by Claude Code) must survive
    conversion — sub2api maps unknown roles to user content, so the text is
    never silently dropped."""
    stats = get_convert_stats("/v1/messages", "openai_chat")
    data = protocol_convert(stats, {
        "model": "claude-3",
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "system", "content": "current system prompt here"},
        ],
        "max_tokens": 100,
    })
    assert "current system prompt here" in json.dumps(data, ensure_ascii=False)
    assert "hi" in json.dumps(data, ensure_ascii=False)


def test_anthropic_system_role_message_to_responses_uses_input_text():
    """Anthropic role=system messages (sent by Claude Code) become Responses
    USER message items with input_text content (sub2api maps unknown roles
    to user) — the text survives instead of being dropped."""
    stats = get_convert_stats("/v1/messages", "openai_responses")
    out = protocol_convert(stats, {
        "model": "claude-3",
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "system", "content": [{"type": "text", "text": "system prompt"}]},
        ],
        "max_tokens": 100,
    })
    user_items = [i for i in out["input"]
                  if isinstance(i, dict) and i.get("role") == "user"]
    assert user_items, "system role message should survive as user content"
    texts = [b.get("text") for i in user_items for b in i["content"]
             if b.get("type") == "input_text"]
    assert "system prompt" in texts, f"system text lost: {out['input']}"
    assert "hi" in texts


def test_anthropic_top_k_dropped_in_openai_chat_conversion():
    """Anthropic top_k has no Chat Completions equivalent and is dropped
    (sub2api ChatCompletionsRequest carries no top_k). Endpoints that want
    it can add it back via server.extra_body."""
    stats = get_convert_stats("/v1/messages", "openai_chat")
    out = protocol_convert(stats, {
        "model": "m", "max_tokens": 100, "top_k": 40,
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert "top_k" not in out


def test_anthropic_thinking_config_maps_to_reasoning_effort():
    """Anthropic thinking config resolves to an effort like sub2api: an
    explicit disable wins ("none"), else output_config.effort, else the
    "medium" default. budget_tokens alone does not select an effort."""
    stats = get_convert_stats("/v1/messages", "openai_chat")
    # adaptive (Claude Code) -> medium default
    out = protocol_convert(stats, {
        "model": "m", "max_tokens": 100,
        "thinking": {"type": "adaptive"},
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert out.get("reasoning_effort") == "medium"
    # enabled + budget -> medium default (budget does not pick the effort)
    out = protocol_convert(stats, {
        "model": "m", "max_tokens": 100,
        "thinking": {"type": "enabled", "budget_tokens": 512},
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert out.get("reasoning_effort") == "medium"
    # disabled -> "none"
    out = protocol_convert(stats, {
        "model": "m", "max_tokens": 100,
        "thinking": {"type": "disabled"},
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert out.get("reasoning_effort") == "none"
    # output_config.effort wins over thinking-derived effort
    out = protocol_convert(stats, {
        "model": "m", "max_tokens": 100,
        "thinking": {"type": "adaptive"},
        "output_config": {"effort": "xhigh"},
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert out.get("reasoning_effort") == "xhigh"


def test_anthropic_effort_max_forwarded_unchanged():
    """Claude Code's `output_config.effort: max` is forwarded as-is.

    This intentionally diverges from sub2api, which clamps `max` down to
    `xhigh` for upstreams whose effort domain stops there. The upstreams this
    gateway talks to accept `max`, so passing it through preserves Claude
    Code's selected thinking depth instead of silently weakening it.
    """
    stats = get_convert_stats("/v1/messages", "openai_responses")
    out = protocol_convert(stats, {
        "model": "m", "max_tokens": 100,
        "output_config": {"effort": "max"},
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert out.get("reasoning") == {"effort": "max", "summary": "auto"}
    # Chat direction forwards it unchanged too.
    stats = get_convert_stats("/v1/messages", "openai_chat")
    out = protocol_convert(stats, {
        "model": "m", "max_tokens": 100,
        "output_config": {"effort": "max"},
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert out.get("reasoning_effort") == "max"


def test_anthropic_adaptive_effort_clamped_to_xhigh_for_openai():
    """`adaptive` is Claude-only and has no upstream equivalent, so it still
    clamps to the largest value the OpenAI effort domain knows."""
    stats = get_convert_stats("/v1/messages", "openai_responses")
    out = protocol_convert(stats, {
        "model": "m", "max_tokens": 100,
        "output_config": {"effort": "adaptive"},
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert out.get("reasoning") == {"effort": "xhigh", "summary": "auto"}
    stats = get_convert_stats("/v1/messages", "openai_chat")
    out = protocol_convert(stats, {
        "model": "m", "max_tokens": 100,
        "output_config": {"effort": "adaptive"},
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert out.get("reasoning_effort") == "xhigh"


def test_convert_request_missing_max_tokens_defaults_for_anthropic():
    """Anthropic requires max_tokens; a missing value defaults to 8192
    (sub2api behavior) instead of failing the conversion."""
    stats = get_convert_stats("/v1/chat/completions", "anthropic_messages")
    data = protocol_convert(stats, {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert data["max_tokens"] == 8192


def test_no_conversion_when_path_matches_pinned_protocol():
    """A request whose path already matches the pinned protocol is returned
    unchanged (no conversion)."""
    stats = get_convert_stats("/v1/messages", "anthropic_messages")
    data = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
    assert protocol_convert(stats, data) is data


# ---------------------------------------------------------------------------
# convert_response_body
# ---------------------------------------------------------------------------

def test_anthropic_to_openai_response_body():
    raw = json.dumps({
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-3",
        "content": [{"type": "text", "text": "hi back"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }).encode()
    out = convert_response_body(raw, get_convert_stats("/v1/chat/completions", "anthropic_messages"))
    data = json.loads(out)
    assert data["object"] == "chat.completion"
    assert data["choices"][0]["message"]["content"] == "hi back"
    assert data["choices"][0]["finish_reason"] == "stop"
    assert data["usage"]["prompt_tokens"] == 10
    assert data["usage"]["completion_tokens"] == 5


def test_openai_to_anthropic_response_body():
    raw = json.dumps({
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "model": "gpt-4o",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
    }).encode()
    out = convert_response_body(raw, get_convert_stats("/v1/messages", "openai_chat"))
    data = json.loads(out)
    assert data["type"] == "message"
    assert data["content"] == [{"type": "text", "text": "hi"}]
    assert data["stop_reason"] == "end_turn"


def test_openai_chat_reasoning_content_becomes_anthropic_thinking():
    """OpenAI Chat message.reasoning_content (plaintext reasoning) must
    become an Anthropic thinking block in C->A response conversion, ordered
    BEFORE the text block (Anthropic's canonical content order)."""
    raw = json.dumps({
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "model": "m",
        "choices": [{"index": 0, "message": {
            "role": "assistant",
            "content": "answer",
            "reasoning_content": "think step by step",
        }, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
    }).encode()
    out = convert_response_body(raw, get_convert_stats("/v1/messages", "openai_chat"))
    data = json.loads(out)
    assert [c["type"] for c in data["content"]] == ["thinking", "text"]
    assert data["content"][0]["thinking"] == "think step by step"
    assert data["content"][1]["text"] == "answer"


def test_openai_chat_reasoning_field_becomes_anthropic_thinking():
    """Some providers (mimo) use message.reasoning instead of
    reasoning_content for plaintext thinking; both must map to a thinking
    block in C->A conversion, ordered before the text block."""
    raw = json.dumps({
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "model": "m",
        "choices": [{"index": 0, "message": {
            "role": "assistant", "content": "answer", "reasoning": "think hard",
        }, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
    }).encode()
    out = convert_response_body(raw, get_convert_stats("/v1/messages", "openai_chat"))
    data = json.loads(out)
    assert [c["type"] for c in data["content"]] == ["thinking", "text"]
    assert data["content"][0]["thinking"] == "think hard"


def test_converted_thinking_block_signature_is_string():
    """Converted thinking blocks (from plaintext reasoning) must carry a
    string signature — the key is always present and an empty string
    satisfies strict clients (Cherry Studio's @ai-sdk/anthropic zod schema
    ``signature: z.string()``) without inventing a magic value (sub2api
    wire shape)."""
    raw = json.dumps({
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "model": "m",
        "choices": [{"index": 0, "message": {
            "role": "assistant",
            "content": "answer",
            "reasoning_content": "think step by step",
        }, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
    }).encode()
    out = convert_response_body(raw, get_convert_stats("/v1/messages", "openai_chat"))
    data = json.loads(out)
    signature = data["content"][0]["signature"]
    assert isinstance(signature, str)


def test_real_thinking_signature_passes_through_unchanged():
    """A thinking block carrying a genuine (non-None) signature must keep it
    verbatim — the placeholder is only for signatures that are missing.
    No gateway conversion path produces a non-None signature today, so this
    locks in the encoder's null-guard directly."""
    from apicompat.anthropic_messages import AnthropicMessagesEncoder
    from apicompat.ir import IRResponse, IRThinkingBlock

    encoder = AnthropicMessagesEncoder()
    response = IRResponse(id="msg_1", model="m", content=[
        IRThinkingBlock(thinking="signed thought", signature="real-sig-123"),
    ])
    block = encoder.encode_response(response)["content"][0]
    assert block["signature"] == "real-sig-123"


def test_openai_chat_cost_preserved_in_anthropic_conversion():
    """OpenAI Chat response cost must survive C->A conversion."""
    raw = json.dumps({
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "model": "m",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        "cost": "0.0005",
    }).encode()
    out = convert_response_body(raw, get_convert_stats("/v1/messages", "openai_chat"))
    assert json.loads(out).get("cost") == "0.0005"


def test_responses_cost_preserved_in_anthropic_conversion():
    raw = json.dumps({
        "id": "resp_1",
        "object": "response",
        "model": "m",
        "status": "completed",
        "output": [{"type": "message", "role": "assistant",
                    "content": [{"type": "output_text", "text": "hi"}]}],
        "usage": {"input_tokens": 2, "output_tokens": 1, "total_tokens": 3},
        "cost": "0.00123",
    }).encode()
    out = convert_response_body(raw, get_convert_stats("/v1/messages", "openai_responses"))
    data = json.loads(out)
    assert data["type"] == "message"
    assert data["cost"] == "0.00123"
    assert data["content"] == [{"type": "text", "text": "hi"}]


def test_responses_reasoning_content_becomes_anthropic_thinking():
    raw = json.dumps({
        "id": "resp_1",
        "object": "response",
        "model": "m",
        "status": "completed",
        "output": [
            {"type": "reasoning", "id": "rs_1", "status": "completed",
             "content": [{"type": "reasoning_text", "text": "think step by step"}]},
            {"type": "message", "role": "assistant",
             "content": [{"type": "output_text", "text": "answer"}]},
        ],
        "usage": {"input_tokens": 2, "output_tokens": 3, "total_tokens": 5},
    }).encode()
    out = convert_response_body(raw, get_convert_stats("/v1/messages", "openai_responses"))
    data = json.loads(out)
    assert [c["type"] for c in data["content"]] == ["thinking", "text"]
    assert data["content"][0]["thinking"] == "think step by step"
    assert data["content"][1]["text"] == "answer"


def test_responses_encrypted_reasoning_becomes_signed_thinking_block():
    """Encrypted reasoning must surface as a thinking block carrying
    encrypted_content as the signature, so multi-turn clients can send it
    back and preserve the provider's prompt cache (sub2api behavior)."""
    raw = json.dumps({
        "id": "resp_1",
        "object": "response",
        "model": "m",
        "status": "completed",
        "output": [
            {"type": "reasoning", "id": "rs_1", "status": "completed",
             "encrypted_content": "Q-PaDgFmAlMp6aZyWqft", "summary": []},
            {"type": "message", "role": "assistant",
             "content": [{"type": "output_text", "text": "answer"}]},
        ],
        "usage": {"input_tokens": 2, "output_tokens": 3, "total_tokens": 5},
    }).encode()
    out = convert_response_body(raw, get_convert_stats("/v1/messages", "openai_responses"))
    data = json.loads(out)
    assert [c["type"] for c in data["content"]] == ["thinking", "text"]
    assert data["content"][0]["signature"] == "Q-PaDgFmAlMp6aZyWqft"


def test_convert_response_invalid_json_raises():
    with pytest.raises(ConversionFailed):
        convert_response_body(b"not json", get_convert_stats("/v1/chat/completions", "anthropic_messages"))


def test_responses_refusal_content_preserved_in_anthropic():
    """Responses refusal content must convert to an Anthropic text block —
    dropping it yields an empty message. muse (user_blocked) returns a
    refusal-only output; the refusal text is the only signal the client gets."""
    raw = json.dumps({
        "id": "resp_1", "object": "response", "status": "completed", "model": "m",
        "output": [{"id": "o1", "type": "message", "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "refusal",
                                 "refusal": "I'm sorry, but I can't help with that request."}]}],
        "usage": {"input_tokens": 2, "output_tokens": 3, "total_tokens": 5},
    }).encode()
    out = json.loads(convert_response_body(raw, get_convert_stats("/v1/messages", "openai_responses")))
    assert out["content"] == [{"type": "text",
                               "text": "I'm sorry, but I can't help with that request."}]
    # sub2api maps a completed refusal to end_turn (the refusal TEXT is
    # the client's signal; Anthropic has no refusal stop for converted
    # Responses upstreams).
    assert out["stop_reason"] == "end_turn"


def test_responses_error_body_becomes_anthropic_error():
    """An upstream error body (e.g. 403 {"error": {...}} with no output) must
    convert to the Anthropic error envelope, preserving the message —
    otherwise the client sees an empty assistant message and the real cause
    (user_blocked) is lost."""
    raw = json.dumps({
        "model": "muse-spark-1.2-contributor",
        "error": {"param": None, "type": "permission_error",
                  "message": "[user_blocked] Your access has been restricted"},
    }).encode()
    out = json.loads(convert_response_body(raw, get_convert_stats("/v1/messages", "openai_responses")))
    assert out["type"] == "error"
    assert out["error"]["type"] == "permission_error"
    assert "user_blocked" in out["error"]["message"]


def test_openai_chat_refusal_content_preserved_in_anthropic():
    """Chat Completions refusal (message.refusal, content null) must convert
    to a text block — dropping it yields an empty message with no signal of
    what happened. The stop reason maps to Anthropic's refusal semantics."""
    raw = json.dumps({
        "id": "chatcmpl-1", "object": "chat.completion", "model": "m",
        "choices": [{"index": 0, "message": {"role": "assistant",
                                              "content": None,
                                              "refusal": "I can't help with that."},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
    }).encode()
    out = json.loads(convert_response_body(raw, get_convert_stats("/v1/messages", "openai_chat")))
    assert out["content"] == [{"type": "text", "text": "I can't help with that."}]
    assert out["stop_reason"] == "refusal"


def test_responses_usage_input_excludes_cached_in_anthropic():
    """Responses input_tokens includes cached tokens; Anthropic input_tokens
    excludes cache reads. The converted response must subtract cached_tokens
    from input_tokens — otherwise clients double-count and perceive a ~50%
    cache hit rate on an almost-fully-cached request."""
    raw = json.dumps({
        "id": "resp_1", "object": "response", "status": "completed", "model": "m",
        "output": [{"id": "o1", "type": "message", "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": "hi"}]}],
        "usage": {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150,
                  "input_tokens_details": {"cached_tokens": 60}},
    }).encode()
    out = json.loads(convert_response_body(raw, get_convert_stats("/v1/messages", "openai_responses")))
    assert out["usage"]["input_tokens"] == 40
    assert out["usage"]["cache_read_input_tokens"] == 60
    assert out["usage"]["output_tokens"] == 50


def test_convert_response_body_accepts_gzip():
    raw = json.dumps({
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-3",
        "content": [{"type": "text", "text": "hi back"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }).encode()
    out = convert_response_body(gzip.compress(raw),
                                get_convert_stats("/v1/chat/completions", "anthropic_messages"))
    data = json.loads(out)
    assert data["choices"][0]["message"]["content"] == "hi back"


# ---------------------------------------------------------------------------
# convert_sse_chunks
# ---------------------------------------------------------------------------

async def _drain(chunks, stats) -> bytes:
    async def gen():
        for c in chunks:
            yield c
    out = b""
    async for piece in convert_sse_chunks(gen(), stats):
        out += piece
    return out


def _lines(data: bytes) -> list[str]:
    return [l for l in data.decode().splitlines() if l.strip()]


def test_sse_openai_chat_to_anthropic():
    chunks = [
        b'data: {"id":"c1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}\n\n',
        b'data: {"id":"c1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"hi"},"finish_reason":null}]}\n\n',
        b'data: {"id":"c1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n',
        b"data: [DONE]\n\n",
    ]
    out = asyncio.run(_drain(chunks, get_convert_stats("/v1/messages", "openai_chat")))
    lines = _lines(out)
    # Anthropic stream: event:/data: pairs, message_start first
    assert lines[0] == "event: message_start"
    assert '"type": "message_start"' in lines[1]
    assert any('"text_delta"' in l for l in lines)
    assert any('"stop_reason": "end_turn"' in l for l in lines)
    # OpenAI chat streams terminate with [DONE]; it must become message_stop
    # so the Anthropic client sees the stream finish.
    assert "event: message_stop" in lines


def test_sse_anthropic_to_openai_chat():
    chunks = [
        b'event: message_start\ndata: {"type":"message_start","message":{"id":"m1","model":"claude-3"}}\n\n',
        b'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hi"}}\n\n',
        b'event: message_stop\ndata: {"type":"message_stop"}\n\n',
    ]
    out = asyncio.run(_drain(chunks, get_convert_stats("/v1/chat/completions", "anthropic_messages")))
    lines = _lines(out)
    # OpenAI chat stream: bare data: lines; [DONE] terminator emitted.
    assert all(not l.startswith("event:") for l in lines)
    assert 'data: {"id": "m1"' in lines[0]
    assert '"delta": {"content": "hi"}' in lines[1]
    assert lines[-1] == "data: [DONE]"


def test_sse_split_chunks_boundary():
    chunk = b'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hi"}}\n\n'
    out = asyncio.run(_drain([chunk[:15], chunk[15:]],
                             get_convert_stats("/v1/chat/completions", "anthropic_messages")))
    # Split mid-line: both pieces reassembled into a single converted chunk.
    assert out.count(b'"delta"') == 1
    assert b'"content": "hi"' in out


def test_sse_openai_chat_reasoning_delta_and_finish_dedup():
    """Streaming OpenAI chat with plaintext reasoning deltas + duplicate
    finish_reason (mimo-style) must produce a valid Anthropic stream:
    thinking block, text block, and a single message_delta."""
    chunks = [
        'data: {"id":"g1","object":"chat.completion.chunk","model":"m","choices":[{"index":0,"finish_reason":null,"delta":{"role":"assistant","content":"","reasoning":"think one"}}]}',
        'data: {"id":"g1","object":"chat.completion.chunk","model":"m","choices":[{"index":0,"finish_reason":null,"delta":{"role":"assistant","content":"","reasoning":" think two"}}]}',
        'data: {"id":"g1","object":"chat.completion.chunk","model":"m","choices":[{"index":0,"finish_reason":null,"delta":{"content":"RAW"}}]}',
        'data: {"id":"g1","object":"chat.completion.chunk","model":"m","choices":[{"index":0,"finish_reason":"stop","delta":{"role":"assistant","content":"","reasoning":null}}]}',
        'data: {"id":"g1","object":"chat.completion.chunk","model":"m","choices":[{"index":0,"finish_reason":"stop","delta":{"role":"assistant","content":""}}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}',
        "data: [DONE]",
    ]
    raw = "\n\n".join(c + "\n\n" for c in chunks)
    out = asyncio.run(_drain([raw.encode()], get_convert_stats("/v1/messages", "openai_chat")))
    text = out.decode()
    # duplicate finish chunks merge into a SINGLE message_delta carrying
    # both the stop_reason and the usage from the second chunk
    mdeltas = [json.loads(l.split(": ", 1)[1]) for l in text.splitlines()
               if l.startswith("data: ") and '"message_delta"' in l]
    assert len(mdeltas) == 1, f"expected one merged message_delta, got {mdeltas}"
    assert mdeltas[0]["delta"].get("stop_reason") == "end_turn"
    assert mdeltas[0]["usage"].get("output_tokens") == 1
    # thinking block with deltas + text
    assert '"type": "thinking"' in text
    assert "thinking_delta" in text
    assert '"think one"' in text and '" think two"' in text
    assert '"text": "RAW"' in text
    # Every content block (thinking AND text) must be opened with its own
    # content_block_start — text deltas must not be emitted under an index
    # that was never started (breaks strict clients like Cherry Studio).
    starts = [json.loads(l.split(": ", 1)[1]) for l in text.splitlines()
              if l.startswith("data: ") and '"content_block_start"' in l]
    assert [(s["index"], s["content_block"]["type"]) for s in starts] == [(0, "thinking"), (1, "text")]
    deltas = [json.loads(l.split(": ", 1)[1]) for l in text.splitlines()
              if l.startswith("data: ") and '"content_block_delta"' in l]
    delta_idxs = sorted({d["index"] for d in deltas})
    assert delta_idxs == [0, 1], f"deltas must stay within started blocks, got indices {delta_idxs}"
    # Every started block must be closed (thinking 0 then text 1).
    stops = [json.loads(l.split(": ", 1)[1])["index"] for l in text.splitlines()
             if l.startswith("data: ") and '"content_block_stop"' in l]
    assert stops == [0, 1], f"every started block needs a matching stop, got {stops}"


def test_sse_openai_chat_reasoning_content_delta_maps_to_thinking():
    """hy3-free streams plaintext reasoning as delta.reasoning_content (mimo
    uses delta.reasoning). The streaming decoder must map BOTH to an
    Anthropic thinking block — dropping reasoning_content loses the entire
    thinking process in Anthropic clients (Cherry Studio)."""
    chunks = [
        'data: {"id":"g1","object":"chat.completion.chunk","model":"m","choices":[{"index":0,"finish_reason":null,"delta":{"role":"assistant"}}]}',
        'data: {"id":"g1","object":"chat.completion.chunk","model":"m","choices":[{"index":0,"finish_reason":null,"delta":{"reasoning_content":"用户问"}}]}',
        'data: {"id":"g1","object":"chat.completion.chunk","model":"m","choices":[{"index":0,"finish_reason":null,"delta":{"reasoning_content":"：9.11和9.9哪个大"}}]}',
        'data: {"id":"g1","object":"chat.completion.chunk","model":"m","choices":[{"index":0,"finish_reason":null,"delta":{"content":"9.9更大。"}}]}',
        'data: {"id":"g1","object":"chat.completion.chunk","model":"m","choices":[{"index":0,"finish_reason":"stop","delta":{}}],"usage":{"prompt_tokens":5,"completion_tokens":8,"total_tokens":13}}',
        "data: [DONE]",
    ]
    raw = "\n\n".join(c + "\n\n" for c in chunks)
    out = asyncio.run(_drain([raw.encode()], get_convert_stats("/v1/messages", "openai_chat")))
    text = out.decode()
    # The reasoning text reaches the client as thinking deltas (SSE escapes
    # non-ASCII, so compare on the parsed events rather than raw text).
    deltas = [json.loads(l.split(": ", 1)[1]) for l in text.splitlines()
              if l.startswith("data: ") and '"content_block_delta"' in l]
    thinking_texts = [d["delta"]["thinking"] for d in deltas
                      if d["delta"].get("type") == "thinking_delta"]
    assert "用户问" in thinking_texts
    assert "：9.11和9.9哪个大" in thinking_texts
    # thinking block opened before the text block, both closed
    starts = [json.loads(l.split(": ", 1)[1]) for l in text.splitlines()
              if l.startswith("data: ") and '"content_block_start"' in l]
    assert [(s["index"], s["content_block"]["type"]) for s in starts] == [(0, "thinking"), (1, "text")]
    assert any(d["delta"].get("text") == "9.9更大。" for d in deltas
               if d["delta"].get("type") == "text_delta")
    stops = [json.loads(l.split(": ", 1)[1])["index"] for l in text.splitlines()
             if l.startswith("data: ") and '"content_block_stop"' in l]
    assert stops == [0, 1]


def test_sse_openai_chat_finish_chunk_usage_maps_to_message_delta():
    """OpenAI chat streams that carry usage on the finish_reason chunk (mimo
    via opencode sends usage together with choices) must map it into the
    Anthropic message_delta — an empty usage {} fails strict client schema
    validation (Cherry Studio requires usage.output_tokens)."""
    chunks = [
        'data: {"id":"g1","object":"chat.completion.chunk","model":"m","choices":[{"index":0,"finish_reason":null,"delta":{"role":"assistant","content":"hi"}}]}',
        'data: {"id":"g1","object":"chat.completion.chunk","model":"m","choices":[{"index":0,"finish_reason":"stop","delta":{}}],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15,"prompt_tokens_details":{"cached_tokens":4}}}',
        "data: [DONE]",
    ]
    raw = "\n\n".join(c + "\n\n" for c in chunks)
    out = asyncio.run(_drain([raw.encode()], get_convert_stats("/v1/messages", "openai_chat")))
    text = out.decode()
    assert '"output_tokens": 5' in text
    assert '"input_tokens": 6' in text
    assert '"cache_read_input_tokens": 4' in text
    assert '"usage": {}' not in text


def test_sse_openai_chat_split_finish_usage_chunks():
    """mimo (via opencode) sends the finish_reason and usage on TWO separate
    finish chunks (first without usage, second with). They must merge into a
    SINGLE Anthropic message_delta carrying both stop_reason and usage —
    extra message_deltas without stop_reason break strict clients (Cherry
    Studio: "Response ended with finish reason 'other'")."""
    chunks = [
        'data: {"id":"g1","object":"chat.completion.chunk","model":"m","choices":[{"index":0,"finish_reason":null,"delta":{"role":"assistant","content":"hi"}}]}',
        'data: {"id":"g1","object":"chat.completion.chunk","model":"m","choices":[{"index":0,"finish_reason":"stop","delta":{}}]}',
        'data: {"id":"g1","object":"chat.completion.chunk","model":"m","choices":[{"index":0,"finish_reason":"stop","delta":{}}],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15,"prompt_tokens_details":{"cached_tokens":4}}}',
        "data: [DONE]",
    ]
    raw = "\n\n".join(c + "\n\n" for c in chunks)
    out = asyncio.run(_drain([raw.encode()], get_convert_stats("/v1/messages", "openai_chat")))
    text = out.decode()
    deltas = [json.loads(l.split(": ", 1)[1]) for l in text.splitlines()
              if l.startswith("data: ") and '"message_delta"' in l]
    assert len(deltas) == 1, f"expected a single message_delta, got {deltas}"
    d = deltas[0]
    assert d["delta"].get("stop_reason") == "end_turn"
    assert d["usage"] == {"output_tokens": 5, "input_tokens": 6,
                          "cache_read_input_tokens": 4,
                          "cache_creation_input_tokens": 0}, f"got {d['usage']}"


def test_openai_chat_usage_input_excludes_cached_in_anthropic():
    """OpenAI chat prompt_tokens includes cached tokens; Anthropic
    input_tokens excludes cache reads. The converted response must subtract
    cached_tokens — otherwise clients double-count and perceive a ~50%
    cache hit rate."""
    raw = json.dumps({
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "model": "m",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120,
                  "prompt_tokens_details": {"cached_tokens": 60}},
    }).encode()
    out = json.loads(convert_response_body(raw, get_convert_stats("/v1/messages", "openai_chat")))
    assert out["usage"]["input_tokens"] == 40
    assert out["usage"]["cache_read_input_tokens"] == 60
    assert out["usage"]["output_tokens"] == 20


def test_sse_openai_chat_usage_input_excludes_cached():
    """Streaming OpenAI chat usage must also subtract cached_tokens from
    input_tokens (mimo reports prompt_tokens including cached)."""
    chunks = [
        'data: {"id":"g1","object":"chat.completion.chunk","model":"m","choices":[{"index":0,"finish_reason":null,"delta":{"role":"assistant","content":"hi"}}]}',
        'data: {"id":"g1","object":"chat.completion.chunk","model":"m","choices":[{"index":0,"finish_reason":"stop","delta":{}}],"usage":{"prompt_tokens":100,"completion_tokens":20,"total_tokens":120,"prompt_tokens_details":{"cached_tokens":60}}}',
        "data: [DONE]",
    ]
    raw = "\n\n".join(c + "\n\n" for c in chunks)
    out = asyncio.run(_drain([raw.encode()], get_convert_stats("/v1/messages", "openai_chat")))
    deltas = [json.loads(l.split(": ", 1)[1]) for l in out.decode().splitlines()
              if l.startswith("data: ") and '"message_delta"' in l]
    assert len(deltas) == 1
    assert deltas[0]["usage"]["input_tokens"] == 40
    assert deltas[0]["usage"]["cache_read_input_tokens"] == 60
    assert deltas[0]["usage"]["output_tokens"] == 20


def test_sse_responses_completed_carries_full_usage_for_stats():
    """The converted Anthropic stream's message_delta must carry the full
    usage (input + cache_read + output) so the stats scanner can accumulate
    counts that Anthropic's message_start never carries. Responses input_tokens
    includes cached tokens, so cached must be subtracted from input to match
    Anthropic semantics (input excludes cache reads)."""
    chunks = [
        b'event: response.created\ndata: {"type":"response.created","response":{"id":"r1","model":"m","created_at":1}}\n\n',
        b'event: response.output_text.delta\ndata: {"type":"response.output_text.delta","output_index":0,"delta":"hi"}\n\n',
        b'event: response.completed\ndata: {"type":"response.completed","response":{"status":"completed","usage":{"input_tokens":100,"output_tokens":50,"total_tokens":150,"input_tokens_details":{"cached_tokens":60}}}}\n\n',
    ]
    out = asyncio.run(_drain(chunks, get_convert_stats("/v1/messages", "openai_responses")))
    text = out.decode()
    assert '"input_tokens": 40' in text
    assert '"cache_read_input_tokens": 60' in text
    assert '"output_tokens": 50' in text
    # stats scanner accumulates full usage
    from gateway.stats.capture import SseUsageScanner
    scanner = SseUsageScanner()
    scanner.feed(out)
    scanner.flush()
    assert scanner.usage.input_tokens == 40
    assert scanner.usage.cache_read_tokens == 60
    assert scanner.usage.output_tokens == 50


def test_sse_responses_completed_terminates_anthropic_stream():
    """Upstream emits response.completed (not response.done) as the final
    event; the converted Anthropic stream must still get message_delta +
    message_stop so clients can finish cleanly."""
    chunks = [
        b'event: response.created\ndata: {"type":"response.created","response":{"id":"r1","model":"m","created_at":1}}\n\n',
        b'event: response.output_text.delta\ndata: {"type":"response.output_text.delta","output_index":0,"delta":"hi"}\n\n',
        b'event: response.completed\ndata: {"type":"response.completed","response":{"status":"completed","usage":{"input_tokens":4,"output_tokens":3}}}\n\n',
    ]
    out = asyncio.run(_drain(chunks, get_convert_stats("/v1/messages", "openai_responses")))
    lines = _lines(out)
    assert 'event: message_delta' in lines
    assert '"stop_reason": "end_turn"' in out.decode()
    assert 'event: message_stop' in lines


def test_sse_responses_refusal_in_completed_emits_text_block():
    """muse (user_blocked) streams 200 + response.completed whose output
    carries a refusal message that was never streamed via output_item events.
    The refusal text must be emitted (start/delta/stop) — otherwise the client
    sees an empty stream and the model's refusal is silently lost."""
    chunks = [
        b'event: response.created\ndata: {"type":"response.created","response":{"id":"r1","model":"m"}}\n\n',
        b'event: response.in_progress\ndata: {"type":"response.in_progress","response":{"id":"r1","status":"in_progress"}}\n\n',
        b'event: response.completed\ndata: {"type":"response.completed","response":{"status":"completed","output":[{"id":"msg_refusal","type":"message","status":"completed","role":"assistant","content":[{"type":"refusal","refusal":"I\'m sorry, but I can\'t help with that request."}]}],"usage":{"input_tokens":2,"output_tokens":3}}}\n\n',
    ]
    out = asyncio.run(_drain(chunks, get_convert_stats("/v1/messages", "openai_responses")))
    text = out.decode()
    # The refusal text reaches the client as a text delta
    assert "I'm sorry, but I can't help with that request." in text
    # A complete content block (start + delta + stop) was emitted for it
    starts = [json.loads(l.split(": ", 1)[1]) for l in text.splitlines()
              if l.startswith("data: ") and '"content_block_start"' in l]
    stops = [json.loads(l.split(": ", 1)[1])["index"] for l in text.splitlines()
             if l.startswith("data: ") and '"content_block_stop"' in l]
    assert len(starts) == 1, f"expected one emitted text block, got {starts}"
    assert stops == [starts[0]["index"]], f"block must be closed, stops={stops}"
    # refusal is a content-filter style stop
    assert '"stop_reason": "refusal"' in text
    assert "message_stop" in text


def test_sse_openai_chat_refusal_delta_emits_text():
    """A streamed refusal (delta.refusal, content null) must surface as a
    text delta and map the stop reason to Anthropic's refusal — otherwise
    the client sees an empty stream."""
    chunks = [
        'data: {"id":"g1","object":"chat.completion.chunk","model":"m","choices":[{"index":0,"delta":{"role":"assistant"}}]}',
        "data: {\"id\":\"g1\",\"object\":\"chat.completion.chunk\",\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"refusal\":\"I can't help with that.\"}}]}",
        'data: {"id":"g1","object":"chat.completion.chunk","model":"m","choices":[{"index":0,"finish_reason":"stop","delta":{}}],"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}}',
        "data: [DONE]",
    ]
    raw = "\n\n".join(c + "\n\n" for c in chunks)
    out = asyncio.run(_drain([raw.encode()], get_convert_stats("/v1/messages", "openai_chat")))
    text = out.decode()
    assert "I can't help with that." in text
    assert '"stop_reason": "refusal"' in text
    assert "message_stop" in text


def test_sse_responses_reasoning_item_pairs_start_with_stop():
    """A Responses stream whose first output is a reasoning item (opencode's
    muse-spark emits encrypted_content reasoning) must emit a content_block_start
    for it — otherwise the subsequent content_block_stop is orphaned and
    Claude Code's stream parser chokes."""
    chunks = [
        'event: response.created\ndata: {"type":"response.created","sequence_number":0,"response":{"id":"r1","status":"in_progress","model":"m"}}',
        'event: response.in_progress\ndata: {"type":"response.in_progress","response":{"id":"r1","status":"in_progress"}}',
        'event: response.output_item.added\ndata: {"type":"response.output_item.added","sequence_number":2,"output_index":0,"item":{"id":"rs_1","type":"reasoning","status":"in_progress","summary":[]}}',
        'event: response.output_item.done\ndata: {"type":"response.output_item.done","sequence_number":3,"output_index":0,"item":{"id":"rs_1","type":"reasoning","status":"completed","encrypted_content":"ABC"}}',
        'event: response.output_item.added\ndata: {"type":"response.output_item.added","sequence_number":4,"output_index":1,"item":{"id":"msg_1","type":"message","status":"in_progress","role":"assistant","content":[]}}',
        'event: response.content_part.added\ndata: {"type":"response.content_part.added","sequence_number":5,"output_index":1,"content_index":0,"item_id":"msg_1","part":{"type":"output_text","text":""}}',
        'event: response.output_text.delta\ndata: {"type":"response.output_text.delta","sequence_number":6,"output_index":1,"content_index":0,"item_id":"msg_1","delta":"Hi"}',
        'event: response.output_text.delta\ndata: {"type":"response.output_text.delta","sequence_number":7,"output_index":1,"content_index":0,"item_id":"msg_1","delta":"!"}',
        'event: response.content_part.done\ndata: {"type":"response.content_part.done","sequence_number":8,"output_index":1,"content_index":0,"item_id":"msg_1","part":{"type":"output_text","text":"Hi!"}}',
        'event: response.output_item.done\ndata: {"type":"response.output_item.done","sequence_number":9,"output_index":1,"item":{"id":"msg_1","type":"message","status":"completed","role":"assistant","content":[{"type":"output_text","text":"Hi!"}]}}',
        'event: response.completed\ndata: {"type":"response.completed","response":{"id":"r1","status":"completed","usage":{"input_tokens":3,"output_tokens":4}}}',
    ]
    raw = "\n\n".join(c + "\n\n" for c in chunks)
    text = asyncio.run(_drain([raw.encode()],
                              get_convert_stats("/v1/messages", "openai_responses"))).decode()
    # Every content_block_stop must be preceded by a start with the same index.
    starts, stops = {}, []
    for l in text.splitlines():
        if not l.startswith("data: "):
            continue
        try:
            ev = json.loads(l[6:])
        except json.JSONDecodeError:
            continue
        if ev.get("type") == "content_block_start":
            starts[ev["index"]] = ev["content_block"]["type"]
        elif ev.get("type") == "content_block_stop":
            stops.append(ev["index"])
    assert stops == [0, 1], f"orphaned/duplicated stops: {stops}"


def test_sse_responses_reasoning_text_delta_maps_to_thinking():
    """deepseek-v4-flash (via opencode) streams plaintext reasoning as
    response.reasoning_text.delta events. These must map to Anthropic
    thinking_delta on the reasoning item's block — otherwise the entire
    thinking process is silently dropped and the client sees an empty
    thinking block."""
    chunks = [
        'event: response.created\ndata: {"type":"response.created","sequence_number":0,"response":{"id":"r1","status":"in_progress","model":"m"}}',
        'event: response.output_item.added\ndata: {"type":"response.output_item.added","sequence_number":2,"output_index":0,"item":{"id":"rs_1","type":"reasoning","status":"in_progress","content":[],"summary":[]}}',
        'event: response.reasoning_text.delta\ndata: {"type":"response.reasoning_text.delta","sequence_number":4,"output_index":0,"content_index":0,"item_id":"rs_1","delta":"We need"}',
        'event: response.reasoning_text.delta\ndata: {"type":"response.reasoning_text.delta","sequence_number":5,"output_index":0,"content_index":0,"item_id":"rs_1","delta":" to compare 9.11 and 9.9"}',
        'event: response.reasoning_text.done\ndata: {"type":"response.reasoning_text.done","sequence_number":6,"output_index":0,"content_index":0,"item_id":"rs_1","text":"We need to compare 9.11 and 9.9"}',
        'event: response.output_item.done\ndata: {"type":"response.output_item.done","sequence_number":7,"output_index":0,"item":{"id":"rs_1","type":"reasoning","status":"completed","content":[{"type":"reasoning_text","text":"We need to compare 9.11 and 9.9"}]}}',
        'event: response.output_item.added\ndata: {"type":"response.output_item.added","sequence_number":8,"output_index":1,"item":{"id":"msg_1","type":"message","status":"in_progress","role":"assistant","content":[]}}',
        'event: response.output_text.delta\ndata: {"type":"response.output_text.delta","sequence_number":9,"output_index":1,"content_index":0,"item_id":"msg_1","delta":"9.9更大。"}',
        'event: response.output_item.done\ndata: {"type":"response.output_item.done","sequence_number":10,"output_index":1,"item":{"id":"msg_1","type":"message","status":"completed","role":"assistant","content":[{"type":"output_text","text":"9.9更大。"}]}}',
        'event: response.completed\ndata: {"type":"response.completed","response":{"id":"r1","status":"completed","usage":{"input_tokens":3,"output_tokens":4}}}',
    ]
    raw = "\n\n".join(c + "\n\n" for c in chunks)
    text = asyncio.run(_drain([raw.encode()],
                              get_convert_stats("/v1/messages", "openai_responses"))).decode()
    deltas = [json.loads(l.split(": ", 1)[1]) for l in text.splitlines()
              if l.startswith("data: ") and '"content_block_delta"' in l]
    # Reasoning deltas map to thinking_delta on the reasoning item's block
    thinking_texts = [d["delta"]["thinking"] for d in deltas
                      if d["delta"].get("type") == "thinking_delta"]
    assert "We need" in thinking_texts, f"reasoning delta lost: {deltas}"
    assert " to compare 9.11 and 9.9" in thinking_texts
    # Text delta still maps to the message block
    assert any(d["delta"].get("text") == "9.9更大。" for d in deltas
               if d["delta"].get("type") == "text_delta")
    # thinking block opened before text, both closed
    starts = [json.loads(l.split(": ", 1)[1]) for l in text.splitlines()
              if l.startswith("data: ") and '"content_block_start"' in l]
    assert [(s["index"], s["content_block"]["type"]) for s in starts] == [(0, "thinking"), (1, "text")]
    stops = [json.loads(l.split(": ", 1)[1])["index"] for l in text.splitlines()
             if l.startswith("data: ") and '"content_block_stop"' in l]
    assert stops == [0, 1]


def test_sse_responses_reasoning_summary_delta_maps_to_thinking():
    """Official OpenAI Responses streams carry reasoning summaries as
    response.reasoning_summary_text.delta; map them to thinking_delta too
    (same treatment as reasoning_text.delta)."""
    chunks = [
        'event: response.created\ndata: {"type":"response.created","sequence_number":0,"response":{"id":"r1","status":"in_progress","model":"m"}}',
        'event: response.output_item.added\ndata: {"type":"response.output_item.added","sequence_number":2,"output_index":0,"item":{"id":"rs_1","type":"reasoning","status":"in_progress","summary":[]}}',
        'event: response.reasoning_summary_text.delta\ndata: {"type":"response.reasoning_summary_text.delta","sequence_number":4,"output_index":0,"content_index":0,"item_id":"rs_1","delta":"Thinking about"}',
        'event: response.output_item.done\ndata: {"type":"response.output_item.done","sequence_number":7,"output_index":0,"item":{"id":"rs_1","type":"reasoning","status":"completed","summary":[{"type":"summary_text","text":"Thinking about"}]}}',
        'event: response.completed\ndata: {"type":"response.completed","response":{"id":"r1","status":"completed","usage":{"input_tokens":3,"output_tokens":4}}}',
    ]
    raw = "\n\n".join(c + "\n\n" for c in chunks)
    text = asyncio.run(_drain([raw.encode()],
                              get_convert_stats("/v1/messages", "openai_responses"))).decode()
    assert "Thinking about" in text
    assert "thinking_delta" in text
