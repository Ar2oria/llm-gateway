"""Structured-output constraints must survive every conversion direction.

A request that pins a JSON schema expects JSON back. Dropping the constraint
during conversion leaves the upstream free to answer in prose, which breaks
any caller that parses the reply — Claude Code's `/goal` Stop hook reports it
as ``JSON hook error: JSON validation failed``.

Each source protocol spells the constraint differently:

- Anthropic: top-level ``output_format`` (current) or
  ``output_config.format`` (newer spelling), schema inline.
- OpenAI Chat: ``response_format.json_schema``.
- OpenAI Responses: ``text.format``.

The IR carries all three as ``IRResponseFormat``, so every direction must
round-trip. ``text`` also carries ``verbosity`` on the Responses side, which
must merge with the format rather than replace it.
"""

from __future__ import annotations

import pytest

from apicompat import Protocol, convert_request

SCHEMA = {
    "type": "object",
    "properties": {
        "ok": {"type": "boolean"},
        "reason": {"type": "string"},
        "impossible": {"type": "boolean"},
    },
    "required": ["ok", "reason"],
    "additionalProperties": False,
}


def anthropic_request(**extra) -> dict:
    request = {
        "model": "judge",
        "max_tokens": 300,
        "messages": [{"role": "user", "content": "condition met?"}],
    }
    request.update(extra)
    return request


def chat_request(**extra) -> dict:
    request = {"model": "judge", "messages": [{"role": "user", "content": "hi"}]}
    request.update(extra)
    return request


def responses_request(**extra) -> dict:
    request = {
        "model": "judge",
        "input": [{"role": "user", "content": [{"type": "input_text", "text": "hi"}]}],
    }
    request.update(extra)
    return request


def chat_format() -> dict:
    return {
        "type": "json_schema",
        "json_schema": {"name": "verdict", "strict": True, "schema": SCHEMA},
    }


def responses_format() -> dict:
    return {"type": "json_schema", "name": "verdict", "strict": True, "schema": SCHEMA}


# --- Anthropic source ---------------------------------------------------


@pytest.mark.parametrize(
    "block",
    [
        pytest.param({"output_format": {"type": "json_schema", "schema": SCHEMA}}, id="output_format"),
        pytest.param({"output_config": {"format": {"type": "json_schema", "schema": SCHEMA}}}, id="output_config.format"),
    ],
)
def test_anthropic_to_responses_keeps_format(block):
    out = convert_request(
        Protocol.ANTHROPIC_MESSAGES, Protocol.OPENAI_RESPONSES, anthropic_request(**block)
    )
    assert out["text"]["format"]["type"] == "json_schema"
    assert out["text"]["format"]["schema"] == SCHEMA


@pytest.mark.parametrize(
    "block",
    [
        pytest.param({"output_format": {"type": "json_schema", "schema": SCHEMA}}, id="output_format"),
        pytest.param({"output_config": {"format": {"type": "json_schema", "schema": SCHEMA}}}, id="output_config.format"),
    ],
)
def test_anthropic_to_chat_keeps_format(block):
    out = convert_request(
        Protocol.ANTHROPIC_MESSAGES, Protocol.OPENAI_CHAT, anthropic_request(**block)
    )
    assert out["response_format"]["type"] == "json_schema"
    assert out["response_format"]["json_schema"]["schema"] == SCHEMA


def test_anthropic_to_responses_keeps_verbosity_alongside_format():
    # The Anthropic bridge pins verbosity "medium"; adding a constraint must
    # not drop it, and it must not swallow the constraint either.
    out = convert_request(
        Protocol.ANTHROPIC_MESSAGES,
        Protocol.OPENAI_RESPONSES,
        anthropic_request(output_format={"type": "json_schema", "schema": SCHEMA}),
    )
    assert out["text"]["verbosity"] == "medium"
    assert out["text"]["format"]["schema"] == SCHEMA


def test_anthropic_without_format_emits_verbosity_only():
    out = convert_request(
        Protocol.ANTHROPIC_MESSAGES, Protocol.OPENAI_RESPONSES, anthropic_request()
    )
    assert out["text"] == {"verbosity": "medium"}


def test_anthropic_output_format_text_is_not_a_constraint():
    out = convert_request(
        Protocol.ANTHROPIC_MESSAGES,
        Protocol.OPENAI_RESPONSES,
        anthropic_request(output_format={"type": "text"}),
    )
    assert "format" not in out["text"]


# --- Chat source --------------------------------------------------------


def test_chat_to_anthropic_keeps_format():
    out = convert_request(
        Protocol.OPENAI_CHAT,
        Protocol.ANTHROPIC_MESSAGES,
        chat_request(response_format=chat_format()),
    )
    assert out["output_format"]["type"] == "json_schema"
    assert out["output_format"]["schema"] == SCHEMA
    assert out["output_format"]["name"] == "verdict"
    assert out["output_format"]["strict"] is True


def test_chat_to_anthropic_keeps_output_config_effort():
    # A preceding output_config assignment must not discard the constraint.
    out = convert_request(
        Protocol.OPENAI_CHAT,
        Protocol.ANTHROPIC_MESSAGES,
        chat_request(
            response_format=chat_format(),
            reasoning_effort="high",
        ),
    )
    assert out["output_config"] == {"effort": "high"}
    assert out["output_format"]["schema"] == SCHEMA


# --- Responses source ---------------------------------------------------


def test_responses_to_anthropic_keeps_format():
    out = convert_request(
        Protocol.OPENAI_RESPONSES,
        Protocol.ANTHROPIC_MESSAGES,
        responses_request(text={"format": responses_format()}),
    )
    assert out["output_format"]["schema"] == SCHEMA
    assert out["output_format"]["name"] == "verdict"


def test_responses_to_chat_keeps_format():
    out = convert_request(
        Protocol.OPENAI_RESPONSES,
        Protocol.OPENAI_CHAT,
        responses_request(text={"format": responses_format()}),
    )
    assert out["response_format"]["json_schema"]["schema"] == SCHEMA


# --- Anthropic -> Anthropic passthrough shape ---------------------------


def test_chat_json_object_to_anthropic_is_not_emitted_as_json_schema():
    # A bare json_object has no schema; Anthropic only models json_schema, so
    # it must not be emitted as an empty constraint.
    out = convert_request(
        Protocol.OPENAI_CHAT,
        Protocol.ANTHROPIC_MESSAGES,
        chat_request(response_format={"type": "json_object"}),
    )
    assert "output_format" not in out
