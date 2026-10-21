"""Direct unit tests for the request-pipeline helpers extracted from app.py.

These pin the helper contracts at the unit level; the HTTP-level behavior
tests (full proxy round-trips) live in test_app.py.
"""

import pytest

from gateway.app import _build_upstream_headers, _reasoning_effort, _rewrite_body
from gateway.config import EndpointConfig
from gateway.convert import get_convert_stats


def make_endpoint(**kwargs) -> EndpointConfig:
    kwargs.setdefault("model", "m")
    kwargs.setdefault("base_url", "https://up.example")
    return EndpointConfig(**kwargs)


# --- _build_upstream_headers ---

def test_hop_by_hop_headers_stripped():
    h = _build_upstream_headers(
        {"Host": "127.0.0.1:8088", "Connection": "keep-alive",
         "Content-Length": "3", "Transfer-Encoding": "chunked",
         "Accept": "application/json"},
        make_endpoint(), "apikey",
    )
    assert h == {"accept": "application/json", "authorization": "", "x-api-key": ""}


def test_custom_non_auth_headers_override_client_and_keys_lowercased():
    h = _build_upstream_headers(
        {"Authorization": "Bearer client-key", "X-Custom": "abc"},
        make_endpoint(custom_headers={"X-Custom": "ep-value", "User-Agent": "ep-agent"}),
        "apikey",
    )
    assert h == {"authorization": "Bearer client-key", "x-custom": "ep-value",
                 "user-agent": "ep-agent", "x-api-key": "client-key"}


def test_sentinel_authorization_activates_endpoint_auth():
    h = _build_upstream_headers(
        {"Authorization": "Bearer apikey"},
        make_endpoint(custom_headers={"Authorization": "Bearer real"}),
        "apikey",
    )
    assert h == {"authorization": "Bearer real", "x-api-key": "real"}


def test_sentinel_x_api_key_activates_endpoint_auth():
    h = _build_upstream_headers(
        {"X-Api-Key": "apikey"},
        make_endpoint(custom_headers={"Authorization": "Bearer real"}),
        "apikey",
    )
    assert h == {"authorization": "Bearer real", "x-api-key": "real"}


def test_client_real_auth_wins_over_endpoint_auth_default():
    h = _build_upstream_headers(
        {"Authorization": "Bearer wrong-key"},
        make_endpoint(custom_headers={"Authorization": "Bearer real"}),
        "apikey",
    )
    assert h == {"authorization": "Bearer wrong-key", "x-api-key": "wrong-key"}


def test_no_client_auth_is_not_backfilled_from_endpoint_auth():
    h = _build_upstream_headers(
        {"Accept": "*/*"},
        make_endpoint(custom_headers={"Authorization": "Bearer real"}),
        "apikey",
    )
    assert h == {"accept": "*/*", "authorization": "", "x-api-key": ""}


def test_sentinel_without_endpoint_auth_sends_no_auth():
    h = _build_upstream_headers(
        {"Authorization": "Bearer apikey"},
        make_endpoint(custom_headers={"User-Agent": "ua"}),
        "apikey",
    )
    assert h == {"authorization": "", "x-api-key": "", "user-agent": "ua"}


def test_x_api_key_only_derives_authorization():
    h = _build_upstream_headers(
        {"X-Api-Key": "client-key"},
        make_endpoint(),
        "apikey",
    )
    assert h == {"x-api-key": "client-key", "authorization": "bearer client-key"}


def test_custom_headers_fixing_both_auth_headers_skip_derivation():
    h = _build_upstream_headers(
        {"Authorization": "Bearer apikey"},
        make_endpoint(custom_headers={"Authorization": "Bearer a", "X-Api-Key": "b"}),
        "apikey",
    )
    assert h == {"authorization": "Bearer a", "x-api-key": "b"}


def test_accept_encoding_strips_unsupported_encodings():
    """br/zstd must not be forwarded: the gateway can only decompress
    gzip/deflate for conversion, and an upstream that picks br makes the
    buffered conversion path fail with a 502 (non-JSON body)."""
    h = _build_upstream_headers(
        {"Accept-Encoding": "gzip, deflate, br, zstd"},
        make_endpoint(), "apikey",
    )
    assert h["accept-encoding"] == "gzip, deflate"


def test_accept_encoding_only_unsupported_becomes_identity():
    h = _build_upstream_headers(
        {"Accept-Encoding": "br, zstd"},
        make_endpoint(), "apikey",
    )
    assert h["accept-encoding"] == "identity"


def test_accept_encoding_wildcard_not_forwarded():
    """'*' lets the upstream pick any coding (e.g. br) — replace it with
    identity so the response is always decodable."""
    h = _build_upstream_headers(
        {"Accept-Encoding": "*"},
        make_endpoint(), "apikey",
    )
    assert h["accept-encoding"] == "identity"


def test_accept_encoding_from_custom_headers_sanitized():
    h = _build_upstream_headers(
        {"Accept-Encoding": "gzip"},
        make_endpoint(custom_headers={"Accept-Encoding": "gzip, br"}),
        "apikey",
    )
    assert h["accept-encoding"] == "gzip"


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
            system_prompt=None):
    _rewrite_body(data, endpoint, get_convert_stats(path, target), extra_body,
                  system_prompt=system_prompt)
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


def test_rewrite_body_replaces_first_openai_system_message():
    out = rewrite({"model": "m", "messages": [
        {"role": "user", "content": "hi"},
        {"role": "system", "content": "old1"},
        {"role": "system", "content": "old2"},
    ]}, make_endpoint(system_prompt="NEW"))
    assert out["messages"][1] == {"role": "system", "content": "NEW"}
    assert out["messages"][2] == {"role": "system", "content": "old2"}


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
    assert out["system"] == "NEW"


def test_rewrite_body_sets_responses_instructions():
    out = rewrite({"model": "m", "instructions": "old", "input": "hi"},
                  make_endpoint(system_prompt="NEW"),
                  path="/v1/responses")
    assert out["instructions"] == "NEW"
    assert "messages" not in out


def test_rewrite_body_applies_upstream_model_and_system_prompt_together():
    out = rewrite({"model": "orig", "messages": []},
                  make_endpoint(upstream_model="real", system_prompt="NEW"),
                  path="/v1/messages")
    assert out["model"] == "real"
    assert out["system"] == "NEW"


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
    assert out["system"] == "TEMPLATE"


def test_rewrite_body_falls_back_to_endpoint_prompt_when_override_none():
    out = rewrite({"model": "m", "system": "client"},
                  make_endpoint(system_prompt="hand"),
                  path="/v1/messages", system_prompt=None)
    assert out["system"] == "hand"


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
        assert out["system"] == "TEMPLATE"
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
