import json

import pytest

from gateway.config import ConfigError, EndpointConfig, GatewayConfig, config_to_dict, load_config, validate_config


def test_load_config_valid(tmp_path):
    p = tmp_path / "config.json"
    p.write_text(json.dumps({
        "server": {"host": "0.0.0.0", "port": 9000, "api_key": "secret"},
        "endpoints": [
            {
                "base_url": "https://api.openai.com/",
                "custom_headers": {"Authorization": "Bearer real"},
                "models": [{"name": "gpt-4o"}],
            },
            {
                "base_url": "https://api.anthropic.com",
                "models": [{"name": "claude-3-5-sonnet"}],
            },
        ],
    }))
    cfg = load_config(p)
    assert cfg.host == "0.0.0.0"
    assert cfg.port == 9000
    assert cfg.api_key == "secret"
    assert len(cfg.endpoints) == 2
    assert cfg.endpoints[0].model == "gpt-4o"
    assert cfg.endpoints[0].base_url == "https://api.openai.com"
    assert cfg.endpoints[0].custom_headers == {"Authorization": "Bearer real"}
    assert cfg.endpoints[1].custom_headers == {}


def test_load_config_defaults(tmp_path):
    p = tmp_path / "config.json"
    p.write_text(json.dumps({"server": {"api_key": "k"}, "endpoints": []}))
    cfg = load_config(p)
    assert cfg.host == "127.0.0.1"
    assert cfg.port == 8080


def test_load_config_requires_api_key(tmp_path):
    p = tmp_path / "config.json"
    p.write_text(json.dumps({"server": {"port": 8080}, "endpoints": []}))
    with pytest.raises(ConfigError, match="api_key"):
        load_config(p)


def test_load_config_requires_base_url(tmp_path):
    p = tmp_path / "config.json"
    p.write_text(json.dumps({
        "server": {"api_key": "k"},
        "endpoints": [{"models": [{"name": "m"}]}],
    }))
    with pytest.raises(ConfigError, match="base_url"):
        load_config(p)


def test_load_config_rejects_duplicate_models(tmp_path):
    p = tmp_path / "config.json"
    p.write_text(json.dumps({
        "server": {"api_key": "k"},
        "endpoints": [
            {"base_url": "https://a.com", "models": [{"name": "gpt-4o"}]},
            {"base_url": "https://b.com", "models": [{"name": "gpt-4o"}]},
        ],
    }))
    with pytest.raises(ConfigError, match="unique"):
        load_config(p)


def test_load_config_upstream_model(tmp_path):
    p = tmp_path / "config.json"
    p.write_text(json.dumps({
        "server": {"api_key": "k"},
        "endpoints": [
            {
                "base_url": "https://a.com",
                "models": [{"name": "local-name", "upstream_model": "real-name"}],
            },
            {"base_url": "https://b.com", "models": [{"name": "plain"}]},
        ],
    }))
    cfg = load_config(p)
    assert cfg.endpoints[0].upstream_model == "real-name"
    assert cfg.endpoints[1].upstream_model is None


def test_load_config_debug_log_default_false(tmp_path):
    p = tmp_path / "config.json"
    p.write_text(json.dumps({"server": {"api_key": "k"}, "endpoints": []}))
    assert load_config(p).debug_log is False


def test_load_config_system_prompt(tmp_path):
    p = tmp_path / "config.json"
    p.write_text(json.dumps({
        "server": {"api_key": "k"},
        "endpoints": [
            {
                "base_url": "https://a.com",
                "models": [{"name": "m1", "system_prompt": "you are x"}],
            },
            {"base_url": "https://b.com", "models": [{"name": "m2"}]},
        ],
    }))
    cfg = load_config(p)
    assert cfg.endpoints[0].system_prompt == "you are x"
    assert cfg.endpoints[1].system_prompt is None


def test_load_config_debug_log_enabled(tmp_path):
    p = tmp_path / "config.json"
    p.write_text(json.dumps({"server": {"api_key": "k", "debug_log": True}, "endpoints": []}))
    assert load_config(p).debug_log is True


def test_load_config_debug_log_rejects_non_bool(tmp_path):
    p = tmp_path / "config.json"
    p.write_text(json.dumps({"server": {"api_key": "k", "debug_log": "yes"}, "endpoints": []}))
    with pytest.raises(ConfigError, match="debug_log"):
        load_config(p)


def test_load_config_multiple_models_one_endpoint(tmp_path):
    p = tmp_path / "config.json"
    p.write_text(json.dumps({
        "server": {"api_key": "k"},
        "endpoints": [
            {
                "base_url": "https://a.com",
                "custom_headers": {"Authorization": "Bearer shared"},
                "models": [
                    {"name": "m1"},
                    {"name": "m2", "upstream_model": "m2-upstream"},
                ],
            },
        ],
    }))
    cfg = load_config(p)
    assert len(cfg.endpoints) == 2
    assert cfg.endpoints[0].model == "m1"
    assert cfg.endpoints[1].model == "m2"
    assert cfg.endpoints[0].base_url == "https://a.com"
    assert cfg.endpoints[1].base_url == "https://a.com"
    assert cfg.endpoints[0].custom_headers == {"Authorization": "Bearer shared"}
    assert cfg.endpoints[1].custom_headers == {"Authorization": "Bearer shared"}
    assert cfg.endpoints[1].upstream_model == "m2-upstream"


def test_load_config_missing_models_list(tmp_path):
    p = tmp_path / "config.json"
    p.write_text(json.dumps({
        "server": {"api_key": "k"},
        "endpoints": [{"base_url": "https://a.com"}],
    }))
    with pytest.raises(ConfigError, match="models"):
        load_config(p)


def test_load_config_empty_models_list_allowed(tmp_path):
    p = tmp_path / "config.json"
    p.write_text(json.dumps({
        "server": {"api_key": "k"},
        "endpoints": [{"base_url": "https://a.com", "models": []}],
    }))
    cfg = load_config(p)
    assert cfg.endpoints == []


def test_load_config_models_not_a_list(tmp_path):
    p = tmp_path / "config.json"
    p.write_text(json.dumps({
        "server": {"api_key": "k"},
        "endpoints": [{"base_url": "https://a.com", "models": {"name": "x"}}],
    }))
    with pytest.raises(ConfigError, match="models"):
        load_config(p)


def test_load_config_model_entry_missing_name(tmp_path):
    p = tmp_path / "config.json"
    p.write_text(json.dumps({
        "server": {"api_key": "k"},
        "endpoints": [
            {"base_url": "https://a.com", "models": [{"upstream_model": "r"}]},
        ],
    }))
    with pytest.raises(ConfigError, match="name"):
        load_config(p)


def test_load_config_model_entry_not_object(tmp_path):
    p = tmp_path / "config.json"
    p.write_text(json.dumps({
        "server": {"api_key": "k"},
        "endpoints": [{"base_url": "https://a.com", "models": ["gpt-4o"]}],
    }))
    with pytest.raises(ConfigError, match="object"):
        load_config(p)


def test_load_config_rejects_duplicate_base_url(tmp_path):
    p = tmp_path / "config.json"
    p.write_text(json.dumps({
        "server": {"api_key": "k"},
        "endpoints": [
            {"base_url": "https://a.com", "models": [{"name": "m1"}]},
            {"base_url": "https://a.com", "models": [{"name": "m2"}]},
        ],
    }))
    with pytest.raises(ConfigError, match="unique"):
        load_config(p)


def test_load_config_endpoint_entry_not_object(tmp_path):
    p = tmp_path / "config.json"
    p.write_text(json.dumps({
        "server": {"api_key": "k"},
        "endpoints": ["https://a.com"],
    }))
    with pytest.raises(ConfigError, match="object"):
        load_config(p)


def test_load_config_duplicate_models_in_same_list(tmp_path):
    p = tmp_path / "config.json"
    p.write_text(json.dumps({
        "server": {"api_key": "k"},
        "endpoints": [
            {
                "base_url": "https://a.com",
                "models": [{"name": "gpt-4o"}, {"name": "gpt-4o"}],
            },
        ],
    }))
    with pytest.raises(ConfigError, match="unique"):
        load_config(p)


def test_load_config_custom_headers_not_a_dict(tmp_path):
    p = tmp_path / "config.json"
    p.write_text(json.dumps({
        "server": {"api_key": "k"},
        "endpoints": [
            {"base_url": "https://a.com", "custom_headers": ["a"], "models": [{"name": "m"}]},
        ],
    }))
    with pytest.raises(ConfigError, match="custom_headers"):
        load_config(p)


def test_load_config_server_not_object(tmp_path):
    p = tmp_path / "config.json"
    p.write_text(json.dumps({"server": "x"}))
    with pytest.raises(ConfigError, match="server"):
        load_config(p)


def test_load_config_top_level_not_object(tmp_path):
    p = tmp_path / "config.json"
    p.write_text(json.dumps([1, 2]))
    with pytest.raises(ConfigError, match="object"):
        load_config(p)


def test_load_config_base_url_all_slashes(tmp_path):
    p = tmp_path / "config.json"
    p.write_text(json.dumps({
        "server": {"api_key": "k"},
        "endpoints": [{"base_url": "/", "models": [{"name": "m"}]}],
    }))
    with pytest.raises(ConfigError, match="base_url"):
        load_config(p)


def test_validate_config_round_trip(tmp_path):
    p = tmp_path / "config.json"
    p.write_text(json.dumps({
        "server": {"api_key": "k"},
        "endpoints": [
            {
                "base_url": "https://a.com",
                "custom_headers": {"User-Agent": "x"},
                "models": [
                    {"name": "m1", "upstream_model": "real", "system_prompt": "sys"},
                    {"name": "m2"},
                ],
            },
        ],
    }))
    cfg = load_config(p)
    assert validate_config(config_to_dict(cfg)) == cfg


def test_config_to_dict_groups_models_by_base_url():
    cfg = GatewayConfig(
        host="127.0.0.1",
        port=8080,
        api_key="k",
        endpoints=[
            EndpointConfig(model="m1", base_url="https://a.com"),
            EndpointConfig(model="m2", base_url="https://a.com"),
        ],
    )
    d = config_to_dict(cfg)
    assert len(d["endpoints"]) == 1
    assert [m["name"] for m in d["endpoints"][0]["models"]] == ["m1", "m2"]


def test_duplicate_base_url_trailing_slash_normalized(tmp_path):
    p = tmp_path / "config.json"
    p.write_text(json.dumps({
        "server": {"api_key": "k"},
        "endpoints": [
            {"base_url": "https://a.com/", "models": [{"name": "m1"}]},
            {"base_url": "https://a.com", "models": [{"name": "m2"}]},
        ],
    }))
    with pytest.raises(ConfigError, match="unique"):
        load_config(p)


BASE_RAW = {
    "server": {"api_key": "k"},
    "endpoints": [
        {"base_url": "https://u.example", "models": [{"name": "m"}]},
    ],
}


def test_pricing_accepted_and_roundtrips():
    raw = {"server": {"api_key": "k"}, "endpoints": [
        {"base_url": "https://u.example", "models": [{"name": "m", "pricing": {"input": 3}}]},
    ]}
    config = validate_config(raw)
    assert config.endpoints[0].pricing == {"input": 3.0}
    out = config_to_dict(config)
    assert out["endpoints"][0]["models"][0]["pricing"] == {"input": 3.0}


def test_pricing_all_four_fields():
    raw = {"server": {"api_key": "k"}, "endpoints": [
        {"base_url": "https://u.example", "models": [{"name": "m", "pricing": {
            "input": 1, "cache_read": 0.2, "cache_write": 2, "output": 5}}]},
    ]}
    config = validate_config(raw)
    assert config.endpoints[0].pricing == {"input": 1.0, "cache_read": 0.2,
                                           "cache_write": 2.0, "output": 5.0}


def test_pricing_absent_and_empty_ok():
    config = validate_config(BASE_RAW)
    assert config.endpoints[0].pricing is None
    raw = {"server": {"api_key": "k"}, "endpoints": [
        {"base_url": "https://u.example", "models": [{"name": "m", "pricing": {}}]},
    ]}
    assert validate_config(raw).endpoints[0].pricing == {}


def test_pricing_negative_rejected():
    raw = {"server": {"api_key": "k"}, "endpoints": [
        {"base_url": "https://u.example", "models": [{"name": "m", "pricing": {"input": -1}}]},
    ]}
    with pytest.raises(ConfigError, match="non-negative"):
        validate_config(raw)


def test_pricing_non_numeric_rejected():
    raw = {"server": {"api_key": "k"}, "endpoints": [
        {"base_url": "https://u.example", "models": [{"name": "m", "pricing": {"input": "abc"}}]},
    ]}
    with pytest.raises(ConfigError, match="must be a number"):
        validate_config(raw)


def test_pricing_unknown_field_rejected():
    raw = {"server": {"api_key": "k"}, "endpoints": [
        {"base_url": "https://u.example", "models": [{"name": "m", "pricing": {"foo": 1}}]},
    ]}
    with pytest.raises(ConfigError, match="unknown pricing field"):
        validate_config(raw)


def test_pricing_nan_and_infinity_rejected():
    for bad in ({"input": 1e999}, {"input": float("nan")}):
        raw = {"server": {"api_key": "k"}, "endpoints": [
            {"base_url": "https://u.example", "models": [{"name": "m", "pricing": bad}]},
        ]}
        with pytest.raises(ConfigError, match="finite"):
            validate_config(raw)


def test_config_server_extra_body_roundtrip(tmp_path):
    p = tmp_path / "config.json"
    p.write_text(json.dumps({
        "server": {"api_key": "k", "extra_body": {"openai_responses": {"store": False, "include": ["a"]}}},
        "endpoints": [{"base_url": "https://a.com", "models": [{"name": "m1"}]}],
    }))
    config = load_config(p)
    assert config.extra_body == {"openai_responses": {"store": False, "include": ["a"]}}
    d = config_to_dict(config)
    assert d["server"]["extra_body"] == {"openai_responses": {"store": False, "include": ["a"]}}


def test_config_server_extra_body_must_be_object(tmp_path):
    p = tmp_path / "config.json"
    p.write_text(json.dumps({
        "server": {"api_key": "k", "extra_body": [1, 2]},
        "endpoints": [],
    }))
    with pytest.raises(ConfigError, match="extra_body"):
        load_config(p)


def test_config_server_extra_body_unknown_protocol_key(tmp_path):
    p = tmp_path / "config.json"
    p.write_text(json.dumps({
        "server": {"api_key": "k", "extra_body": {"gopher": {"a": 1}}},
        "endpoints": [],
    }))
    with pytest.raises(ConfigError, match="extra_body"):
        load_config(p)


def test_models_news_valid():
    raw = {
        "server": {"api_key": "k"},
        "endpoints": [{
            "base_url": "https://a.com",
            "models": [{"name": "m1"}],
            "models_news": ["new-1", "new-2"],
        }],
    }
    cfg = validate_config(raw)
    assert cfg.endpoints[0].models_news == ["new-1", "new-2"]


def test_models_news_optional_defaults_empty():
    raw = {
        "server": {"api_key": "k"},
        "endpoints": [{"base_url": "https://a.com", "models": [{"name": "m1"}]}],
    }
    cfg = validate_config(raw)
    assert cfg.endpoints[0].models_news == []


@pytest.mark.parametrize("bad", [
    "not-a-list",
    [1, 2],
    [""],
    ["ok", "ok"],
])
def test_models_news_invalid_raises(bad):
    raw = {
        "server": {"api_key": "k"},
        "endpoints": [{
            "base_url": "https://a.com",
            "models": [{"name": "m1"}],
            "models_news": bad,
        }],
    }
    with pytest.raises(ConfigError, match="models_news"):
        validate_config(raw)


def test_config_to_dict_roundtrips_models_news():
    raw = {
        "server": {"api_key": "k"},
        "endpoints": [{
            "base_url": "https://a.com",
            "models": [{"name": "m1"}, {"name": "m2"}],
            "models_news": ["v2", "v3"],
        }],
    }
    cfg = validate_config(raw)
    d = config_to_dict(cfg)
    assert d["endpoints"][0]["models_news"] == ["v2", "v3"]


def test_config_to_dict_omits_empty_models_news():
    raw = {
        "server": {"api_key": "k"},
        "endpoints": [{"base_url": "https://a.com", "models": [{"name": "m1"}]}],
    }
    d = config_to_dict(validate_config(raw))
    assert "models_news" not in d["endpoints"][0]


def test_prompt_template_round_trips(tmp_path):
    p = tmp_path / "config.json"
    p.write_text(json.dumps({
        "server": {"api_key": "k"},
        "endpoints": [{"base_url": "https://a.com", "models": [
            {"name": "m1", "system_prompt": "hand", "prompt_template": "system-prompt"},
        ]}],
    }))
    cfg = load_config(p)
    assert cfg.endpoints[0].prompt_template == "system-prompt"
    assert cfg.endpoints[0].system_prompt == "hand"      # 手写内容保留
    out = config_to_dict(cfg)
    model = out["endpoints"][0]["models"][0]
    assert model["prompt_template"] == "system-prompt"
    assert model["system_prompt"] == "hand"


def test_prompt_template_absent_stays_absent():
    cfg = validate_config({"server": {"api_key": "k"}, "endpoints": [
        {"base_url": "https://a.com", "models": [{"name": "m1"}]}]})
    assert cfg.endpoints[0].prompt_template is None
    assert "prompt_template" not in config_to_dict(cfg)["endpoints"][0]["models"][0]


@pytest.mark.parametrize("bad", ["", "  ", ".", "..", "a/b", "a\\b"])
def test_prompt_template_rejects_illegal_names(bad):
    with pytest.raises(ConfigError, match="prompt_template"):
        validate_config({"server": {"api_key": "k"}, "endpoints": [
            {"base_url": "https://a.com", "models": [{"name": "m1", "prompt_template": bad}]}]})


def test_prompt_template_rejects_non_string():
    with pytest.raises(ConfigError, match="prompt_template"):
        validate_config({"server": {"api_key": "k"}, "endpoints": [
            {"base_url": "https://a.com", "models": [{"name": "m1", "prompt_template": 5}]}]})


@pytest.mark.parametrize("bad", [" .", " .. ", "a/b "])
def test_prompt_template_rejects_names_after_stripping(bad):
    # 名字先 strip 再校验：否则 " ." / " .. " 会被存成 "." / ".."。
    with pytest.raises(ConfigError, match="prompt_template"):
        validate_config({"server": {"api_key": "k"}, "endpoints": [
            {"base_url": "https://a.com", "models": [{"name": "m1", "prompt_template": bad}]}]})


def test_prompt_template_trims_surrounding_whitespace():
    raw = {"server": {"api_key": "k"}, "endpoints": [
        {"base_url": "https://a.com",
         "models": [{"name": "m1", "prompt_template": " system-prompt "}]}]}
    cfg = validate_config(raw)
    assert cfg.endpoints[0].prompt_template == "system-prompt"
    assert config_to_dict(cfg)["endpoints"][0]["models"][0]["prompt_template"] == "system-prompt"
