"""Direct unit tests for the request-pipeline helpers extracted from app.py.

These pin the helper contracts at the unit level; the HTTP-level behavior
tests (full proxy round-trips) live in test_app.py.
"""

import asyncio

import pytest

from gateway.app import _rewrite_headers, _reasoning_effort, _rewrite_body
from gateway.config import EndpointConfig
from gateway.convert import get_convert_stats


def make_endpoint(**kwargs) -> EndpointConfig:
    kwargs.setdefault("model", "m")
    kwargs.setdefault("base_url", "https://up.example")
    return EndpointConfig(**kwargs)


class _StubPromptStore:
    """Stands in for ``app.state.prompt_templates`` in _rewrite_body.

    ``bodies`` maps template name → resolved body; a name that is absent
    resolves to None, which is what a template-less or blank template yields
    in production (and triggers the endpoint.system_prompt fallback).
    """

    def __init__(self, bodies: dict[str, str] | None = None) -> None:
        self.bodies = bodies or {}

    async def aget_prompt(self, name: str) -> str | None:
        return self.bodies.get(name)


class _StubApp:
    """Minimal stand-in exposing only what _rewrite_body reads from the app."""

    def __init__(self, bodies: dict[str, str] | None = None) -> None:
        self.state = type("state", (), {})()
        self.state.prompt_templates = _StubPromptStore(bodies)


# --- _build_upstream_headers ---

def test_hop_by_hop_headers_stripped():
    h = _rewrite_headers(
        {"Host": "127.0.0.1:8088", "Connection": "keep-alive",
         "Content-Length": "3", "Transfer-Encoding": "chunked",
         "Accept": "application/json"},
        make_endpoint(), "apikey",
    )
    assert h == {"accept": "application/json", "authorization": "", "x-api-key": ""}


def test_custom_non_auth_headers_override_client_and_keys_lowercased():
    h = _rewrite_headers(
        {"Authorization": "Bearer client-key", "X-Custom": "abc"},
        make_endpoint(custom_headers={"X-Custom": "ep-value", "User-Agent": "ep-agent"}),
        "apikey",
    )
    assert h == {"authorization": "Bearer client-key", "x-custom": "ep-value",
                 "user-agent": "ep-agent", "x-api-key": "client-key"}


def test_sentinel_authorization_activates_endpoint_auth():
    h = _rewrite_headers(
        {"Authorization": "Bearer apikey"},
        make_endpoint(custom_headers={"Authorization": "Bearer real"}),
        "apikey",
    )
    assert h == {"authorization": "Bearer real", "x-api-key": "real"}


def test_sentinel_x_api_key_activates_endpoint_auth():
    h = _rewrite_headers(
        {"X-Api-Key": "apikey"},
        make_endpoint(custom_headers={"Authorization": "Bearer real"}),
        "apikey",
    )
    assert h == {"authorization": "Bearer real", "x-api-key": "real"}


def test_client_real_auth_wins_over_endpoint_auth_default():
    h = _rewrite_headers(
        {"Authorization": "Bearer wrong-key"},
        make_endpoint(custom_headers={"Authorization": "Bearer real"}),
        "apikey",
    )
    assert h == {"authorization": "Bearer wrong-key", "x-api-key": "wrong-key"}


def test_no_client_auth_is_not_backfilled_from_endpoint_auth():
    h = _rewrite_headers(
        {"Accept": "*/*"},
        make_endpoint(custom_headers={"Authorization": "Bearer real"}),
        "apikey",
    )
    assert h == {"accept": "*/*", "authorization": "", "x-api-key": ""}


def test_sentinel_without_endpoint_auth_sends_no_auth():
    h = _rewrite_headers(
        {"Authorization": "Bearer apikey"},
        make_endpoint(custom_headers={"User-Agent": "ua"}),
        "apikey",
    )
    assert h == {"authorization": "", "x-api-key": "", "user-agent": "ua"}


def test_x_api_key_only_derives_authorization():
    h = _rewrite_headers(
        {"X-Api-Key": "client-key"},
        make_endpoint(),
        "apikey",
    )
    assert h == {"x-api-key": "client-key", "authorization": "bearer client-key"}


def test_custom_headers_fixing_both_auth_headers_skip_derivation():
    h = _rewrite_headers(
        {"Authorization": "Bearer apikey"},
        make_endpoint(custom_headers={"Authorization": "Bearer a", "X-Api-Key": "b"}),
        "apikey",
    )
    assert h == {"authorization": "Bearer a", "x-api-key": "b"}


def test_accept_encoding_strips_unsupported_encodings():
    """br/zstd must not be forwarded: the gateway can only decompress
    gzip/deflate for conversion, and an upstream that picks br makes the
    buffered conversion path fail with a 502 (non-JSON body)."""
    h = _rewrite_headers(
        {"Accept-Encoding": "gzip, deflate, br, zstd"},
        make_endpoint(), "apikey",
    )
    assert h["accept-encoding"] == "gzip, deflate"


def test_accept_encoding_only_unsupported_becomes_identity():
    h = _rewrite_headers(
        {"Accept-Encoding": "br, zstd"},
        make_endpoint(), "apikey",
    )
    assert h["accept-encoding"] == "identity"


def test_accept_encoding_wildcard_not_forwarded():
    """'*' lets the upstream pick any coding (e.g. br) — replace it with
    identity so the response is always decodable."""
    h = _rewrite_headers(
        {"Accept-Encoding": "*"},
        make_endpoint(), "apikey",
    )
    assert h["accept-encoding"] == "identity"


def test_accept_encoding_from_custom_headers_sanitized():
    """custom_headers may carry accept-encoding too — it must be filtered.

    Filtering only the client's value let an endpoint configured with
    ``Accept-Encoding: gzip, br`` reach the upstream with br intact, which the
    gateway cannot decode for conversion/capture.
    """
    h = _rewrite_headers(
        {"Accept-Encoding": "gzip"},
        make_endpoint(custom_headers={"Accept-Encoding": "gzip, br"}),
        "apikey",
    )
    assert h["accept-encoding"] == "gzip"


def test_accept_encoding_only_br_from_custom_headers_becomes_identity():
    h = _rewrite_headers(
        {"Accept-Encoding": "gzip"},
        make_endpoint(custom_headers={"Accept-Encoding": "br"}),
        "apikey",
    )
    assert h["accept-encoding"] == "identity"


# --- _rewrite_body ---


@pytest.mark.parametrize("body, expected", [
    ({"reasoning": {"effort": "xhigh"}}, "xhigh"),
    ({"output_config": {"effort": "medium"}}, "medium"),
    ({"thinking": {"type": "enabled", "budget_tokens": 4096}}, "budget:4096"),
    ({"thinking": {"type": "disabled"}}, "none"),
    ({"thinking": {"type": "adaptive"}}, "adaptive"),
    ({"messages": []}, None),
])
def test_reasoning_effort_extracts_final_upstream_shape(body, expected):
    assert _reasoning_effort(body) == expected


# --- _rewrite_body ---

def rewrite(data, endpoint, path="/v1/chat/completions", target=None, extra_body=None,
            system_prompt=None, templates=None):
    """Run _rewrite_body and return the mutated body.

    ``system_prompt`` used to be a direct argument; it is now resolved inside
    _rewrite_body from the injected template (via app.state) falling back to
    ``endpoint.system_prompt``. To keep these tests expressing "an override
    prompt of X", install it as the endpoint's template body.
    """
    if system_prompt is not None:
        endpoint.prompt_template = "test-template"
        templates = dict(templates or {})
        templates["test-template"] = system_prompt
    asyncio.run(_rewrite_body(_StubApp(templates), endpoint, data,
                              get_convert_stats(path, target), extra_body))
    return data


def test_rewrite_body_untouched_when_nothing_configured():
    data = {"model": "orig", "messages": []}
    out = rewrite(data, make_endpoint())
    assert out == {"model": "orig", "messages": []}


def test_rewrite_body_replaces_upstream_model_only():
    out = rewrite(
        {"model": "orig", "messages": [{"role": "user", "content": "hi"}]},
        make_endpoint(upstream_model="real"),
    )
    assert out["model"] == "real"
    assert out["messages"] == [{"role": "user", "content": "hi"}]


def anthropic_system_block(text):
    """The shape the Anthropic system slot takes after an injected prompt."""
    return [{"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}]


def test_rewrite_body_replaces_first_openai_system_message():
    # System/developer entries inside the first four slots are dropped, then
    # the injected prompt takes slot 0. system at index 5 is conversation and
    # survives.
    out = rewrite({"model": "m", "messages": [
        {"role": "user", "content": "hi"},
        {"role": "system", "content": "old1"},
        {"role": "developer", "content": "old2"},
        {"role": "user", "content": "u3"},
        {"role": "user", "content": "u4"},
        {"role": "system", "content": "keep"},
    ]}, make_endpoint(system_prompt="NEW"))
    assert out["messages"][0] == {"role": "system", "content": "NEW"}
    assert [m["role"] for m in out["messages"]] == [
        "system", "user", "user", "user", "system",
    ]
    assert out["messages"][-1] == {"role": "system", "content": "keep"}


def test_rewrite_body_prepends_openai_system_message_when_absent():
    out = rewrite(
        {"model": "m", "messages": [{"role": "user", "content": "hi"}]},
        make_endpoint(system_prompt="NEW"),
    )
    assert out["messages"][0] == {"role": "system", "content": "NEW"}
    assert out["messages"][1] == {"role": "user", "content": "hi"}


def test_rewrite_body_sets_anthropic_top_level_system():
    out = rewrite({"model": "m", "messages": []},
                  make_endpoint(system_prompt="NEW"),
                  path="/v1/messages")
    assert out["system"] == anthropic_system_block("NEW")


def test_rewrite_body_sets_responses_instructions():
    out = rewrite({"model": "m", "instructions": "old", "input": "hi"},
                  make_endpoint(system_prompt="NEW"),
                  path="/v1/responses")
    assert out["instructions"] == "NEW"
    assert "messages" not in out


def _msg(role, text="x"):
    return {"type": "message", "role": role,
            "content": [{"type": "input_text", "text": text}]}


def test_rewrite_body_clears_stale_instructions_on_the_responses_path():
    out = rewrite({"model": "m", "instructions": "stale", "input": []},
                  make_endpoint(system_prompt="NEW"),
                  path="/v1/responses")
    assert out["instructions"] == "NEW"


def test_rewrite_body_drops_system_and_developer_items_in_the_first_four():
    out = rewrite({
        "model": "m",
        "input": [
            _msg("developer", "d0"),
            _msg("system", "s1"),
            _msg("user", "u2"),
            _msg("developer", "d3"),
        ],
    }, make_endpoint(system_prompt="NEW"), path="/v1/responses")
    assert [i["role"] for i in out["input"]] == ["user"]
    assert out["instructions"] == "NEW"


def test_rewrite_body_leaves_system_items_from_index_four_on():
    # Clients head-load their system items; anything past the fourth slot is
    # real conversation and must survive untouched.
    out = rewrite({
        "model": "m",
        "input": [
            _msg("user", "u0"),
            _msg("user", "u1"),
            _msg("user", "u2"),
            _msg("user", "u3"),
            _msg("developer", "d4"),
            _msg("system", "s5"),
        ],
    }, make_endpoint(system_prompt="NEW"), path="/v1/responses")
    assert [i["role"] for i in out["input"]] == [
        "user", "user", "user", "user", "developer", "system",
    ]


def test_rewrite_body_only_counts_message_items_toward_the_four_slot_window():
    # The window is array position, not "the fourth message": tool items keep
    # their slots, so a developer item behind four tool calls is past it.
    tool = {"type": "function_call_output", "call_id": "c1", "output": "out"}
    out = rewrite({
        "model": "m",
        "input": [tool, tool, tool, tool, _msg("developer", "d4")],
    }, make_endpoint(system_prompt="NEW"), path="/v1/responses")
    assert out["input"][-1]["role"] == "developer"


def test_rewrite_body_responses_leaves_input_alone_without_a_prompt():
    payload = {"model": "m", "instructions": "client", "input": [_msg("developer", "d0")]}
    out = rewrite(dict(payload),
                  make_endpoint(), path="/v1/responses", system_prompt=None)
    assert out == payload


@pytest.mark.parametrize("path,payload", [
    ("/v1/messages", {"model": "m", "system": "client",
                      "messages": [{"role": "user", "content": "hi"}]}),
    ("/v1/responses", {"model": "m", "instructions": "client",
                       "input": [_msg("developer", "d0")]}),
    ("/v1/chat/completions", {"model": "m", "messages": [
        {"role": "system", "content": "client"},
        {"role": "user", "content": "hi"}]}),
])
def test_rewrite_body_without_any_prompt_preserves_client_system(path, payload):
    """Regression: no template and no endpoint.system_prompt must be a no-op.

    An earlier refactor called _rewrite_system_prompt unconditionally, which
    deletes the client's instructions/system item first (to make room for an
    injected prompt) and then writes the None/"" it was given — silently
    wiping the caller's system prompt on every such request.
    """
    out = rewrite(dict(payload), make_endpoint(), path=path, system_prompt=None)
    assert out == payload


def test_rewrite_body_applies_upstream_model_and_system_prompt_together():
    out = rewrite({"model": "orig", "messages": []},
                  make_endpoint(upstream_model="real", system_prompt="NEW"),
                  path="/v1/messages")
    assert out["model"] == "real"
    assert out["system"] == anthropic_system_block("NEW")


def test_rewrite_body_extra_body_null_deletes_field():
    # extra_body applies to converted requests (anthropic -> chat): the
    # converted body carries stop; a null value deletes it.
    out = rewrite({"model": "m", "messages": [{"role": "user", "content": "hi"}], "stop": "XYZ"},
                   make_endpoint(), path="/v1/messages",
                   target="openai_chat", extra_body={"openai_chat": {"stop": None}})
    assert "stop" not in out


def test_rewrite_body_extra_body_merges_non_null_and_deletes():
    out = rewrite({"model": "m", "messages": [], "stop": "XYZ", "max_tokens": 10},
                   make_endpoint(), path="/v1/messages",
                   target="openai_chat",
                   extra_body={"openai_chat": {"stop": None, "max_tokens": 42}})
    assert "stop" not in out
    assert out["max_tokens"] == 42


def test_rewrite_body_extra_body_skipped_on_passthrough():
    # extra_body only merges into converted requests — on direct passthrough
    # (client format matches the pinned protocol / no pin) the body is
    # forwarded as-is.
    out = rewrite({"model": "m", "messages": [], "stop": "XYZ"},
                   make_endpoint(), extra_body={"openai_chat": {"stop": None}})
    assert out["stop"] == "XYZ"


def test_rewrite_body_extra_body_keyed_by_target_protocol():
    # Converting chat -> anthropic: extra_body keyed by the pinned protocol
    out = rewrite({"model": "m", "messages": [], "stop": "XYZ"},
                   make_endpoint(), path="/v1/chat/completions",
                   target="anthropic_messages", extra_body={"anthropic_messages": {"stop": None}})
    assert "stop" not in out


# --- _rewrite_body: injected template body (request-time override) ---

def test_rewrite_body_uses_explicit_system_prompt_override():
    out = rewrite({"model": "m", "system": "client"},
                  make_endpoint(system_prompt="hand"),
                  path="/v1/messages", system_prompt="TEMPLATE")
    assert out["system"] == anthropic_system_block("TEMPLATE")


def test_rewrite_body_falls_back_to_endpoint_prompt_when_override_none():
    out = rewrite({"model": "m", "system": "client"},
                  make_endpoint(system_prompt="hand"),
                  path="/v1/messages", system_prompt=None)
    assert out["system"] == anthropic_system_block("hand")


def test_rewrite_body_writes_nothing_when_no_prompt_anywhere():
    out = rewrite({"model": "m", "messages": [{"role": "user", "content": "hi"}]},
                  make_endpoint(), system_prompt=None)
    assert out == {"model": "m", "messages": [{"role": "user", "content": "hi"}]}


@pytest.mark.parametrize("path", ["/v1/messages", "/v1/responses", "/v1/chat/completions"])
def test_rewrite_body_override_lands_in_each_protocols_system_slot(path):
    out = rewrite({"model": "m", "messages": [{"role": "user", "content": "hi"}]},
                  make_endpoint(system_prompt="hand"), path=path,
                  system_prompt="TEMPLATE")
    if path == "/v1/messages":
        assert out["system"] == anthropic_system_block("TEMPLATE")
    elif path == "/v1/responses":
        assert out["instructions"] == "TEMPLATE"
    else:
        assert out["messages"][0] == {"role": "system", "content": "TEMPLATE"}


def test_rewrite_body_empty_override_keeps_no_prompt():
    # An empty string means "no effective prompt": the truthiness check wins
    # over the handwritten fallback, so a blank body never writes anything.
    out = rewrite({"model": "m", "messages": [{"role": "user", "content": "hi"}]},
                  make_endpoint(system_prompt="hand"), system_prompt="")
    assert out == {"model": "m", "messages": [{"role": "user", "content": "hi"}]}


# --- structured-output calls bypass the injected prompt -------------------

SCHEMA = {"type": "object", "properties": {"ok": {"type": "boolean"}},
          "required": ["ok"], "additionalProperties": False}


def test_rewrite_body_skips_template_for_anthropic_structured_output():
    # Claude Code's hook evaluator (`/goal`), memory recall, and the
    # title/rename helpers all send an Anthropic ``output_format``; their
    # system prompt is the machine-facing instruction that tells the model to
    # answer in JSON. Replacing it with a persona template strips the only
    # instruction the caller depends on.
    out = rewrite(
        {"model": "m", "max_tokens": 100,
         "system": "judge the transcript",
         "messages": [{"role": "user", "content": "hi"}],
         "output_format": {"type": "json_schema", "schema": SCHEMA}},
        make_endpoint(system_prompt="PERSONA"),
        path="/v1/messages", system_prompt="PERSONA",
    )
    assert out["system"] == "judge the transcript"


def test_rewrite_body_skips_template_for_anthropic_output_config_format():
    # The newer spelling must be recognized too.
    out = rewrite(
        {"model": "m", "max_tokens": 100,
         "system": "judge the transcript",
         "messages": [{"role": "user", "content": "hi"}],
         "output_config": {"format": {"type": "json_schema", "schema": SCHEMA}}},
        make_endpoint(), path="/v1/messages", system_prompt="PERSONA",
    )
    assert out["system"] == "judge the transcript"


def test_rewrite_body_skips_template_for_responses_text_format():
    # Codex drives this shape: ``text: {format: {...}}`` over the Responses
    # wire API, with the system prompt in ``instructions``.
    out = rewrite(
        {"model": "m", "instructions": "judge the transcript", "input": "hi",
         "text": {"format": {"type": "json_schema", "name": "v", "schema": SCHEMA}}},
        make_endpoint(), path="/v1/responses", system_prompt="PERSONA",
    )
    assert out["instructions"] == "judge the transcript"


def test_rewrite_body_skips_template_for_chat_response_format():
    out = rewrite(
        {"model": "m",
         "messages": [{"role": "system", "content": "judge the transcript"},
                      {"role": "user", "content": "hi"}],
         "response_format": {"type": "json_schema",
                             "json_schema": {"name": "v", "schema": SCHEMA}}},
        make_endpoint(), path="/v1/chat/completions", system_prompt="PERSONA",
    )
    roles = [m["role"] for m in out["messages"]]
    assert roles == ["system", "user"]
    assert out["messages"][0]["content"] == "judge the transcript"


def test_rewrite_body_text_format_does_not_bypass():
    # An explicit ``{"type": "text"}`` is the absence of a constraint; a
    # normal conversation must still receive the template.
    out = rewrite(
        {"model": "m", "instructions": "old", "input": "hi",
         "text": {"format": {"type": "text"}}},
        make_endpoint(), path="/v1/responses", system_prompt="PERSONA",
    )
    assert out["instructions"] == "PERSONA"


def test_rewrite_body_still_applies_upstream_model_when_bypassing():
    # Only the prompt is skipped; model rewriting and extra_body still run.
    out = rewrite(
        {"model": "m", "max_tokens": 100, "system": "judge",
         "messages": [{"role": "user", "content": "hi"}],
         "output_format": {"type": "json_schema", "schema": SCHEMA}},
        make_endpoint(upstream_model="real"),
        path="/v1/messages", extra_body={"anthropic_messages": {"temperature": 0}},
        system_prompt="PERSONA",
    )
    assert out["model"] == "real"
    assert out["system"] == "judge"
