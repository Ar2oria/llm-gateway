"""Parity tests against sub2api's apicompat conversion behavior.

The golden files under ``tests/golden/sub2api/`` were produced by running
sub2api's own converters (``internal/pkg/apicompat``) over the fixtures in
``fixtures.json`` via a temporary Go harness. Each golden embeds the input
payload and sub2api's converted output.

These tests replay the same inputs through this project's IR-based
converter and assert the outputs are identical (after masking
nondeterministic ids/timestamps). When the two reference projects
(new-api / sub2api) disagree, sub2api wins — these goldens are the spec.

Regenerating the goldens:
    cd ~/workspace/sub2api/backend
    GOLDEN_DIR=/tmp/sub2api-golden \
    GOLDEN_FIXTURES=<repo>/tests/golden/sub2api/fixtures.json \
    go test ./internal/pkg/apicompat/ -run TestZZGoldenDump
    cp /tmp/sub2api-golden/*.json <repo>/tests/golden/sub2api/
"""

import json
import re
from pathlib import Path

import pytest

from apicompat import Protocol, convert_request, convert_response
from apicompat.anthropic_messages import (
    AnthropicMessagesDecoder,
    AnthropicMessagesEncoder,
)
from apicompat.openai_chat import (
    OpenAIChatDecoder,
    OpenAIChatEncoder,
)
from apicompat.openai_responses import (
    OpenAIResponsesDecoder,
    OpenAIResponsesEncoder,
)

GOLDEN_DIR = Path(__file__).parent / "golden"

# Intentionally no longer maintaining Anthropic thinking.budget_tokens when
# converting Responses reasoning effort back to Anthropic.
_SKIPPED_CASES = {
    "responses_to_anthropic_request__r2ar_reasoning_high":
        "thinking.budget_tokens is intentionally no longer maintained",
    "responses_to_anthropic_request__r2ar_reasoning_xhigh":
        "thinking.budget_tokens is intentionally no longer maintained",
    # sub2api clamps `max` down to `xhigh`; this gateway forwards `max` so
    # Claude Code's selected thinking depth reaches upstreams that accept it.
    # The divergence is deliberate (commit b83d80a), not a regression.
    "anthropic_to_responses_request__a2r_output_config_max":
        "`max` effort is intentionally forwarded instead of clamped to `xhigh`",
}

# Randomly generated ids (Go: 12 random bytes hex) and wall-clock stamps —
# identical in shape, never identical in value.
_ITEM_ID_RE = re.compile(r"^item_[0-9a-f]{24}$")
_RESP_ID_RE = re.compile(r"^resp_[0-9a-f]{24}$")
_CHATCMPL_ID_RE = re.compile(r"^chatcmpl-[0-9a-f]{24}$")


def _normalize(value):
    """Mask nondeterministic values (generated ids, timestamps)."""
    if isinstance(value, dict):
        out = {}
        for key, sub in value.items():
            if key in ("created_at", "created") and isinstance(sub, int):
                out[key] = 0
            else:
                out[key] = _normalize(sub)
        return out
    if isinstance(value, list):
        return [_normalize(v) for v in value]
    if isinstance(value, str):
        if _ITEM_ID_RE.match(value):
            return "item_X"
        if _RESP_ID_RE.match(value):
            return "resp_X"
        if _CHATCMPL_ID_RE.match(value):
            return "chatcmpl_X"
    return value


def _run_conversion(case):
    """Run one fixture through this project's converter."""
    direction = case["direction"]
    # sub2api's response converters take the client-facing model as a
    # parameter (it wins over the upstream response's model).
    model_options = {"model": case["client_model"]} if case.get("client_model") else None

    if direction == "anthropic_to_responses_request":
        return convert_request(
            Protocol.ANTHROPIC_MESSAGES, Protocol.OPENAI_RESPONSES, case["input"]
        )

    if direction == "responses_to_anthropic_response":
        return convert_response(
            Protocol.OPENAI_RESPONSES,
            Protocol.ANTHROPIC_MESSAGES,
            case["input"],
            options=model_options,
        )

    if direction == "responses_to_anthropic_request":
        return convert_request(
            Protocol.OPENAI_RESPONSES, Protocol.ANTHROPIC_MESSAGES, case["input"]
        )

    if direction == "anthropic_to_responses_response":
        return convert_response(
            Protocol.ANTHROPIC_MESSAGES, Protocol.OPENAI_RESPONSES, case["input"]
        )

    if direction == "responses_stream_to_anthropic":
        decoder = OpenAIResponsesDecoder()
        encoder = AnthropicMessagesEncoder()
        events = []
        for raw in case.get("inputs", []):
            for ir_event in decoder.decode_stream_event(raw):
                events.extend(encoder.encode_stream_event(ir_event))
        if case.get("finalize"):
            for ir_event in decoder.finalize():
                events.extend(encoder.encode_stream_event(ir_event))
        return events

    if direction == "anthropic_stream_to_responses":
        decoder = AnthropicMessagesDecoder()
        encoder = OpenAIResponsesEncoder()
        events = []
        for raw in case.get("inputs", []):
            for ir_event in decoder.decode_stream_event(raw):
                events.extend(encoder.encode_stream_event(ir_event))
        if case.get("finalize"):
            events.extend(encoder.finalize())
        return events

    if direction == "anthropic_to_chat_request":
        return convert_request(
            Protocol.ANTHROPIC_MESSAGES, Protocol.OPENAI_CHAT, case["input"]
        )

    if direction == "chat_to_anthropic_response":
        return convert_response(
            Protocol.OPENAI_CHAT,
            Protocol.ANTHROPIC_MESSAGES,
            case["input"],
            options=model_options,
        )

    if direction == "chat_to_anthropic_stream":
        decoder = OpenAIChatDecoder()
        encoder = AnthropicMessagesEncoder()
        events = []
        for raw in case.get("inputs", []):
            for ir_event in decoder.decode_stream_event(raw):
                events.extend(encoder.encode_stream_event(ir_event))
        if case.get("finalize"):
            for ir_event in decoder.finalize():
                events.extend(encoder.encode_stream_event(ir_event))
        return events

    if direction == "chat_to_responses_request":
        return convert_request(
            Protocol.OPENAI_CHAT, Protocol.OPENAI_RESPONSES, case["input"]
        )

    if direction == "responses_to_chat_response":
        return convert_response(
            Protocol.OPENAI_RESPONSES,
            Protocol.OPENAI_CHAT,
            case["input"],
            options=model_options,
        )

    if direction == "responses_to_chat_request":
        return convert_request(
            Protocol.OPENAI_RESPONSES, Protocol.OPENAI_CHAT, case["input"]
        )

    if direction == "responses_to_chat_stream":
        decoder = OpenAIResponsesDecoder()
        encoder = OpenAIChatEncoder()
        chunks = []
        for raw in case.get("inputs", []):
            for ir_event in decoder.decode_stream_event(raw):
                chunks.extend(encoder.encode_stream_event(ir_event))
        if case.get("finalize"):
            for ir_event in decoder.finalize():
                chunks.extend(encoder.encode_stream_event(ir_event))
        # The [DONE] marker is transport framing our encoder appends (the
        # Chat SSE terminator); sub2api's converter contract leaves it to
        # the service layer. Strip it before comparing.
        return [c for c in chunks if c != "[DONE]"]

    raise ValueError(f"unknown direction {direction}")


def _load_cases():
    cases = []
    for path in sorted(GOLDEN_DIR.glob("*.json")):
        if path.name == "fixtures.json":
            continue
        reason = _SKIPPED_CASES.get(path.stem)
        marks = [pytest.mark.skip(reason=reason)] if reason else []
        cases.append(pytest.param(path, id=path.stem, marks=marks))
    return cases


@pytest.mark.parametrize("golden_path", _load_cases())
def test_parity_with_sub2api(golden_path):
    golden = json.loads(golden_path.read_text())
    expected = _normalize(golden["output"])
    actual = _normalize(_run_conversion(golden))
    assert actual == expected, _diff_report(golden["name"], expected, actual)


def _diff_report(name, expected, actual):
    """Build a compact, path-annotated diff for the first mismatch."""
    lines = [f"parity mismatch in {name}"]
    for path, exp, act in _walk(expected, actual):
        lines.append(f"  at {path or '<root>'}:")
        lines.append(f"    sub2api: {json.dumps(exp, ensure_ascii=False)}")
        lines.append(f"    ours:    {json.dumps(act, ensure_ascii=False)}")
        break
    return "\n".join(lines)


def _walk(expected, actual, path=""):
    """Yield (path, expected, actual) for the first few differences."""
    if type(expected) is not type(actual):
        yield path, expected, actual
        return
    if isinstance(expected, dict):
        for key in sorted(set(expected) | set(actual)):
            if key not in expected:
                yield f"{path}.{key}", "<absent>", actual[key]
            elif key not in actual:
                yield f"{path}.{key}", expected[key], "<absent>"
            else:
                yield from _walk(expected[key], actual[key], f"{path}.{key}")
    elif isinstance(expected, list):
        if len(expected) != len(actual):
            yield f"{path}[len]", expected, actual
            return
        for i, (exp, act) in enumerate(zip(expected, actual)):
            yield from _walk(exp, act, f"{path}[{i}]")
    else:
        if expected != actual:
            yield path, expected, actual
