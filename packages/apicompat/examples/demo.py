"""Runnable tour of the apicompat API.

    python examples/demo.py          # from packages/apicompat
    uv run --package apicompat python packages/apicompat/examples/demo.py

Everything here runs offline — no API keys, no network.
"""

from __future__ import annotations

import asyncio
import json

from apicompat import (
    StreamCodec,
    astream,
    convert_request,
    convert_response,
    convert_stream,
    encode_stream_terminator,
)


def show(title: str, payload: object) -> None:
    print(f"\n--- {title} ---")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


# ---------------------------------------------------------------------------
# 1. Request: an Anthropic client talking to an OpenAI Responses upstream
# ---------------------------------------------------------------------------

anthropic_request = {
    "model": "claude-sonnet-5-5",
    "max_tokens": 512,
    "system": "You are concise.",
    "messages": [{"role": "user", "content": "What is 2+2?"}],
}

responses_request = convert_request(
    "anthropic_messages", "openai_responses", anthropic_request
)
show("Anthropic request -> Responses request", responses_request)


# ---------------------------------------------------------------------------
# 2. Non-streaming response: convert back for the Anthropic client
# ---------------------------------------------------------------------------

responses_response = {
    "id": "resp_1",
    "object": "response",
    "status": "completed",
    "model": "gpt-5",
    "output": [
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "4"}],
        }
    ],
    "usage": {"input_tokens": 12, "output_tokens": 1},
}

anthropic_response = convert_response(
    "openai_responses",
    "anthropic_messages",
    responses_response,
    options={"model": "claude-sonnet-5-5"},  # echo the client's model name
)
show("Responses response -> Anthropic response", anthropic_response)


# ---------------------------------------------------------------------------
# 3. Streaming: bytes in, bytes out
# ---------------------------------------------------------------------------

upstream_sse = [
    b'event: response.created\ndata: {"type":"response.created","response":'
    b'{"id":"resp_1","model":"gpt-5"}}\n\n',
    b'event: response.output_item.added\ndata: {"type":"response.output_item.added",'
    b'"output_index":0,"item":{"type":"message","role":"assistant","content":[],'
    b'"status":"in_progress"}}\n\n',
    b'event: response.output_text.delta\ndata: {"type":"response.output_text.delta",'
    b'"output_index":0,"content_index":0,"delta":"4"}\n\n',
    b'event: response.completed\ndata: {"type":"response.completed","response":'
    b'{"id":"resp_1","model":"gpt-5","status":"completed","output":[],'
    b'"usage":{"input_tokens":12,"output_tokens":1}}}\n\n',
]

print("\n--- Responses SSE -> Anthropic SSE ---")
for chunk in convert_stream("openai_responses", "anthropic_messages", upstream_sse):
    print(chunk.decode().rstrip("\n"))


# ---------------------------------------------------------------------------
# 4. Async variant
# ---------------------------------------------------------------------------

async def demo_async() -> None:
    async def source():
        for chunk in upstream_sse:
            yield chunk

    print("\n--- astream (async) ---")
    async for chunk in astream("openai_responses", "anthropic_messages", source()):
        print(chunk.decode().rstrip("\n"))


asyncio.run(demo_async())


# ---------------------------------------------------------------------------
# 5. Manual control: upstream died, still emit a clean terminator
# ---------------------------------------------------------------------------

codec = StreamCodec("openai_responses", "anthropic_messages")
for chunk in upstream_sse[:2]:            # pretend the connection dropped here
    for out in codec.feed(chunk):
        pass
terminator = encode_stream_terminator("anthropic_messages")
print("\n--- terminator after a dropped upstream ---")
print(terminator.decode().rstrip("\n"))
