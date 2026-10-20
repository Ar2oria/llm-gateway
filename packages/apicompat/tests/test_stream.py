"""Tests for the byte-level stream API (convert_stream / astream / StreamCodec)."""

import asyncio

import pytest

from apicompat import (
    ConversionError,
    Protocol,
    StreamCodec,
    astream,
    convert_stream,
    encode_stream_terminator,
)

# A minimal OpenAI Responses stream: created → text delta → completed.
RESPONSES_STREAM = [
    b'event: response.created\ndata: {"type":"response.created","response":'
    b'{"id":"resp_abc","model":"gpt-5","status":"in_progress"}}\n\n',
    b'event: response.output_item.added\ndata: {"type":"response.output_item.added",'
    b'"output_index":0,"item":{"type":"message","role":"assistant","content":[],'
    b'"status":"in_progress"}}\n\n',
    b'event: response.output_text.delta\ndata: {"type":"response.output_text.delta",'
    b'"output_index":0,"content_index":0,"delta":"hello"}\n\n',
    b'event: response.completed\ndata: {"type":"response.completed","response":'
    b'{"id":"resp_abc","model":"gpt-5","status":"completed","output":[],'
    b'"usage":{"input_tokens":3,"output_tokens":1}}}\n\n',
]


def _join(chunks):
    return b"".join(chunks)


def test_convert_stream_produces_anthropic_events():
    out = _join(convert_stream("openai_responses", "anthropic_messages", RESPONSES_STREAM))
    assert b"event: message_start" in out
    assert b"event: content_block_delta" in out
    assert b'"text": "hello"' in out
    assert b"event: message_stop" in out


def test_convert_stream_handles_chunks_split_mid_line():
    """Upstream framing never aligns with SSE events in practice."""
    along = b"".join(RESPONSES_STREAM)
    # Split every 7 bytes — guaranteed to cut events mid-line.
    chunks = [along[i:i + 7] for i in range(0, len(along), 7)]
    out = _join(convert_stream("openai_responses", "anthropic_messages", chunks))
    expected = _join(convert_stream("openai_responses", "anthropic_messages", RESPONSES_STREAM))
    assert out == expected


def test_convert_stream_accepts_str_chunks():
    along = b"".join(RESPONSES_STREAM).decode()
    out = _join(convert_stream("openai_responses", "anthropic_messages", [along]))
    assert b"event: message_stop" in out


def test_convert_stream_synthesizes_termination_on_truncated_upstream():
    """No response.completed: the client must still get message_stop."""
    truncated = RESPONSES_STREAM[:3]
    out = _join(convert_stream("openai_responses", "anthropic_messages", truncated))
    assert b"event: message_stop" in out
    assert b'"stop_reason": "end_turn"' in out


def test_convert_stream_chat_to_chat_passes_done_marker():
    chunks = [
        b'data: {"id":"c1","object":"chat.completion.chunk","choices":'
        b'[{"index":0,"delta":{"content":"hi"},"finish_reason":null}]}\n\n',
        b"data: [DONE]\n\n",
    ]
    out = _join(convert_stream("openai_chat", "openai_chat", chunks))
    assert out.endswith(b"data: [DONE]\n\n")


def test_convert_stream_accepts_protocol_enum():
    out = _join(convert_stream(
        Protocol.OPENAI_RESPONSES, Protocol.ANTHROPIC_MESSAGES, RESPONSES_STREAM
    ))
    assert b"event: message_stop" in out


def test_convert_stream_rejects_unknown_protocol():
    with pytest.raises(ConversionError, match="unknown protocol"):
        list(convert_stream("not_a_protocol", "openai_chat", []))


def test_stream_terminator_per_protocol():
    assert b"message_stop" in encode_stream_terminator("anthropic_messages")
    assert b"response.completed" in encode_stream_terminator("openai_responses")
    assert b"[DONE]" in encode_stream_terminator("openai_chat")


def test_stream_codec_feed_and_finalize_separately():
    codec = StreamCodec("openai_responses", "anthropic_messages")
    out = []
    for chunk in RESPONSES_STREAM:
        out.extend(codec.feed(chunk))
    out.extend(codec.finalize())
    assert b"".join(out) == _join(
        convert_stream("openai_responses", "anthropic_messages", RESPONSES_STREAM)
    )


def test_stream_codec_finalize_is_idempotent():
    codec = StreamCodec("openai_responses", "anthropic_messages")
    for chunk in RESPONSES_STREAM:
        codec.feed(chunk)
    codec.finalize()
    assert codec.finalize() == []


def test_stream_codec_ignores_malformed_events():
    codec = StreamCodec("openai_responses", "anthropic_messages")
    assert codec.feed(b"data: {not json}\n\n") == []
    # A later valid event still converts.
    out = codec.feed(RESPONSES_STREAM[0])
    assert b"message_start" in b"".join(out)


def test_astream_matches_sync_output():
    async def run():
        async def source():
            for chunk in RESPONSES_STREAM:
                yield chunk
        return _join([c async for c in astream(
            "openai_responses", "anthropic_messages", source()
        )])

    assert asyncio.run(run()) == _join(
        convert_stream("openai_responses", "anthropic_messages", RESPONSES_STREAM)
    )


def test_chunk_size_controls_output_fragmentation():
    fine = list(convert_stream("openai_responses", "anthropic_messages",
                               RESPONSES_STREAM, chunk_size=16))
    coarse = list(convert_stream("openai_responses", "anthropic_messages",
                                 RESPONSES_STREAM, chunk_size=0))
    assert _join(fine) == _join(coarse)
    assert len(fine) > len(coarse)
