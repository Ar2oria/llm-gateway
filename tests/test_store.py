import json

import pytest

from gateway.config import ConfigError, GatewayConfig, load_config
from gateway.store import ConfigStore, NotFoundError, StoreError


def make_store(tmp_path):
    p = tmp_path / "config.json"
    p.write_text(json.dumps({
        "server": {"api_key": "k"},
        "endpoints": [
            {
                "base_url": "https://a.com",
                "custom_headers": {"Authorization": "Bearer a"},
                "models": [{"name": "m1"}, {"name": "m2", "upstream_model": "m2-real"}],
            },
            {"base_url": "https://b.com", "models": [{"name": "m3"}]},
        ],
    }))
    return ConfigStore(load_config(p), p), p


def test_add_endpoint_writes_file_and_routes(tmp_path):
    store, p = make_store(tmp_path)
    store.add_endpoint("https://c.com", {"User-Agent": "ua/1.0"})
    store.add_model("https://c.com", "mc")
    assert "mc" in store.router.list_models()
    data = json.loads(p.read_text())
    entry = next(e for e in data["endpoints"] if e["base_url"] == "https://c.com")
    assert entry["custom_headers"] == {"User-Agent": "ua/1.0"}
    assert [m["name"] for m in entry["models"]] == ["mc"]
    assert store.endpoint_dicts(base_url="https://c.com")[0]["custom_headers"] == {"User-Agent": "ua/1.0"}


def test_add_endpoint_duplicate_base_url_raises_and_unchanged(tmp_path):
    store, p = make_store(tmp_path)
    before = p.read_text()
    with pytest.raises(ConfigError, match="unique"):
        store.add_endpoint("https://a.com")
    assert p.read_text() == before
    # config.endpoints is flattened per model, so compare the unique base_urls.
    assert {e.base_url for e in store.config.endpoints} == {"https://a.com", "https://b.com"}
    assert {f.name for f in tmp_path.iterdir()} == {"config.json"}


def test_add_model_with_slash_in_name(tmp_path):
    store, p = make_store(tmp_path)
    store.add_model("https://a.com", "mcli/deepseek-v4-flash")
    assert store.router.route("mcli/deepseek-v4-flash").base_url == "https://a.com"
    data = json.loads(p.read_text())
    names = [m["name"] for e in data["endpoints"] for m in e["models"]]
    assert "mcli/deepseek-v4-flash" in names


def test_add_model_duplicate_name_raises(tmp_path):
    store, p = make_store(tmp_path)
    with pytest.raises(ConfigError, match="unique"):
        store.add_model("https://a.com", "m1")


def test_add_model_duplicate_name_across_endpoints_raises(tmp_path):
    store, p = make_store(tmp_path)
    with pytest.raises(ConfigError, match="unique"):
        store.add_model("https://b.com", "m1")


def test_update_endpoint_headers_replaces(tmp_path):
    store, p = make_store(tmp_path)
    store.update_endpoint("https://a.com", {"User-Agent": "new/1.0"})
    assert store.router.route("m1").custom_headers == {"User-Agent": "new/1.0"}
    data = json.loads(p.read_text())
    entry = next(e for e in data["endpoints"] if e["base_url"] == "https://a.com")
    assert entry["custom_headers"] == {"User-Agent": "new/1.0"}
    assert store.endpoint_dicts(base_url="https://a.com")[0]["custom_headers"] == {"User-Agent": "new/1.0"}


def test_delete_endpoint_removes_models_from_router_and_file(tmp_path):
    store, p = make_store(tmp_path)
    store.delete_endpoint("https://a.com")
    assert store.router.list_models() == ["m3"]
    data = json.loads(p.read_text())
    assert [e["base_url"] for e in data["endpoints"]] == ["https://b.com"]
    with pytest.raises(NotFoundError):
        store.delete_endpoint("https://a.com")


def test_update_model_sets_and_clears_fields(tmp_path):
    store, p = make_store(tmp_path)
    ep = store.update_model("https://a.com", "m1", upstream_model="real-1", system_prompt="sys")
    assert ep.upstream_model == "real-1"
    assert ep.system_prompt == "sys"
    store.update_model("https://a.com", "m1", upstream_model=None, system_prompt=None)
    data = json.loads(p.read_text())
    entry = next(e for e in data["endpoints"] if e["base_url"] == "https://a.com")
    m1 = next(m for m in entry["models"] if m["name"] == "m1")
    assert "upstream_model" not in m1
    assert "system_prompt" not in m1


def test_delete_model_leaves_empty_models(tmp_path):
    store, p = make_store(tmp_path)
    store.delete_model("https://b.com", "m3")
    assert "m3" not in store.router.list_models()
    data = json.loads(p.read_text())
    entry = next(e for e in data["endpoints"] if e["base_url"] == "https://b.com")
    assert entry["models"] == []


def test_empty_models_endpoint_preserved_after_write(tmp_path):
    p = tmp_path / "config.json"
    p.write_text(json.dumps({
        "server": {"api_key": "k"},
        "endpoints": [
            {"base_url": "https://empty.com", "models": []},
            {"base_url": "https://a.com", "models": [{"name": "m1"}]},
        ],
    }))
    store = ConfigStore(load_config(p), p)
    store.add_model("https://empty.com", "new-model")
    data = json.loads(p.read_text())
    assert [e["base_url"] for e in data["endpoints"]] == ["https://empty.com", "https://a.com"]
    entry = next(e for e in data["endpoints"] if e["base_url"] == "https://empty.com")
    assert [m["name"] for m in entry["models"]] == ["new-model"]


def test_endpoint_dicts_includes_empty_models_endpoint(tmp_path):
    p = tmp_path / "config.json"
    p.write_text(json.dumps({
        "server": {"api_key": "k"},
        "endpoints": [
            {"base_url": "https://empty.com", "models": []},
            {"base_url": "https://a.com", "models": [{"name": "m1", "upstream_model": "r"}]},
        ],
    }))
    store = ConfigStore(load_config(p), p)
    assert store.endpoint_dicts() == [
        {"base_url": "https://empty.com", "custom_headers": {}, "models_news": [], "models": []},
        {
            "base_url": "https://a.com",
            "custom_headers": {},
            "models_news": [],
            "models": [{"name": "m1", "upstream_model": "r", "system_prompt": None,
                        "prompt_template": None, "pricing": None, "protocol": None}],
        },
    ]
    assert store.endpoint_dicts(base_url="https://empty.com") == [
        {"base_url": "https://empty.com", "custom_headers": {}, "models_news": [], "models": []}
    ]


def test_store_without_path_raises_store_error(tmp_path):
    p = tmp_path / "config.json"
    p.write_text(json.dumps({
        "server": {"api_key": "k"},
        "endpoints": [{"base_url": "https://a.com", "models": [{"name": "m"}]}],
    }))
    store = ConfigStore(load_config(p))
    assert store.path is None
    with pytest.raises(StoreError):
        store.add_endpoint("https://b.com")
    assert store.router.list_models() == ["m"]


def test_config_without_endpoints_key_supported(tmp_path):
    p = tmp_path / "config.json"
    p.write_text(json.dumps({"server": {"api_key": "k"}}))
    store = ConfigStore(load_config(p), p)
    assert store.endpoint_dicts() == []
    store.add_endpoint("https://a.com")
    assert [e["base_url"] for e in store.endpoint_dicts()] == ["https://a.com"]


def test_debug_log_from_caller_survives_write(tmp_path):
    p = tmp_path / "config.json"
    p.write_text(json.dumps({
        "server": {"api_key": "k", "debug_log": False},
        "endpoints": [{"base_url": "https://a.com", "models": [{"name": "m"}]}],
    }))
    config = load_config(p)
    config.debug_log = True  # simulates the --debug-log CLI override in __main__
    store = ConfigStore(config, p)
    store.add_endpoint("https://b.com")
    data = json.loads(p.read_text())
    assert data["server"]["debug_log"] is True
    assert store.config.debug_log is True


def test_atomic_write_failure_keeps_state_and_no_litter(tmp_path, monkeypatch):
    store, p = make_store(tmp_path)
    before = p.read_text()
    def boom(src, dst):
        raise OSError("disk full")
    monkeypatch.setattr("gateway.store.os.replace", boom)
    with pytest.raises(StoreError, match="failed to write config"):
        store.add_endpoint("https://c.com")
    assert p.read_text() == before
    assert store.router.list_models() == ["m1", "m2", "m3"]
    assert {f.name for f in tmp_path.iterdir()} == {"config.json"}


def test_old_snapshot_not_mutated(tmp_path):
    store, p = make_store(tmp_path)
    old_config = store.config
    old_router = store.router
    old_endpoints = list(old_config.endpoints)
    store.add_endpoint("https://c.com")
    assert store.config is not old_config
    assert store.router is not old_router
    # config.endpoints is flattened per model, so compare the unique base_urls.
    assert {e.base_url for e in old_config.endpoints} == {"https://a.com", "https://b.com"}
    assert old_router.list_models() == ["m1", "m2", "m3"]
    assert old_config.endpoints == old_endpoints


def test_add_model_with_pricing(tmp_path):
    store, p = make_store(tmp_path)
    ep = store.add_model("https://a.com", "new-model", pricing={"input": 2.0})
    assert ep.pricing == {"input": 2.0}
    entry = store.endpoint_dicts(base_url="https://a.com")[0]
    model = next(m for m in entry["models"] if m["name"] == "new-model")
    assert model["pricing"] == {"input": 2.0}
    data = json.loads(p.read_text())
    names = [m["name"] for e in data["endpoints"] for m in e["models"]]
    assert "new-model" in names
    # survives reload from disk
    store2 = ConfigStore(load_config(p), p)
    ep2 = next(e for e in store2.config.endpoints if e.model == "new-model")
    assert ep2.pricing == {"input": 2.0}


def test_update_model_pricing_remove_and_keep(tmp_path):
    store, p = make_store(tmp_path)
    store.add_model("https://a.com", "m9", pricing={"input": 2.0})
    # explicit None removes
    store.update_model("https://a.com", "m9", pricing=None)
    ep = next(e for e in store.config.endpoints if e.model == "m9")
    assert ep.pricing is None
    # absent keeps
    store.add_model("https://a.com", "m8", pricing={"input": 2.0})
    store.update_model("https://a.com", "m8")
    ep = next(e for e in store.config.endpoints if e.model == "m8")
    assert ep.pricing == {"input": 2.0}
    # empty dict removes
    store.update_model("https://a.com", "m8", pricing={})
    ep = next(e for e in store.config.endpoints if e.model == "m8")
    assert ep.pricing is None


def test_add_model_invalid_pricing_rejected(tmp_path):
    store, p = make_store(tmp_path)
    with pytest.raises(ConfigError, match="non-negative"):
        store.add_model("https://a.com", "bad", pricing={"input": -1})


def test_update_extra_body_writes_and_clears(tmp_path):
    p = tmp_path / "config.json"
    p.write_text(json.dumps({
        "server": {"api_key": "k"},
        "endpoints": [{"base_url": "https://a.com", "models": [{"name": "m1"}]}],
    }))
    store = ConfigStore(load_config(p), p)
    assert store.config.extra_body is None
    store.update_extra_body({"openai_responses": {"store": False}})
    assert store.config.extra_body == {"openai_responses": {"store": False}}
    data = json.loads(p.read_text())
    assert data["server"]["extra_body"] == {"openai_responses": {"store": False}}
    store.update_extra_body(None)
    assert store.config.extra_body is None
    data = json.loads(p.read_text())
    assert "extra_body" not in data["server"]


def test_set_models_news_writes_and_clears(tmp_path):
    store, p = make_store(tmp_path)
    store.set_models_news("https://a.com", ["v2-new", "v3-new"])
    ep = next(e for e in store.config.endpoints if e.base_url == "https://a.com")
    assert ep.models_news == ["v2-new", "v3-new"]
    data = json.loads(p.read_text())
    assert data["endpoints"][0]["models_news"] == ["v2-new", "v3-new"]
    store.set_models_news("https://a.com", None)
    ep = next(e for e in store.config.endpoints if e.base_url == "https://a.com")
    assert ep.models_news == []
    data = json.loads(p.read_text())
    assert "models_news" not in data["endpoints"][0]


def test_set_models_news_replaces_previous_list(tmp_path):
    store, p = make_store(tmp_path)
    store.set_models_news("https://a.com", ["one"])
    store.set_models_news("https://a.com", ["two"])
    data = json.loads(p.read_text())
    assert data["endpoints"][0]["models_news"] == ["two"]


def test_set_models_news_empty_list_clears(tmp_path):
    store, p = make_store(tmp_path)
    store.set_models_news("https://a.com", ["one"])
    store.set_models_news("https://a.com", [])
    data = json.loads(p.read_text())
    assert "models_news" not in data["endpoints"][0]


def test_set_models_news_unknown_endpoint(tmp_path):
    store, p = make_store(tmp_path)
    with pytest.raises(NotFoundError):
        store.set_models_news("https://nope.com", ["x"])


def test_endpoint_dicts_exposes_models_news(tmp_path):
    store, p = make_store(tmp_path)
    store.set_models_news("https://a.com", ["v2"])
    dicts = store.endpoint_dicts()
    assert dicts[0]["models_news"] == ["v2"]
    assert dicts[1]["models_news"] == []


def test_set_prompt_template_writes_name_and_keeps_handwritten_prompt(tmp_path):
    store, p = make_store(tmp_path)
    store.update_model("https://a.com", "m1", system_prompt="hand")
    ep = store.set_prompt_template("https://a.com", "m1", "system-prompt")
    assert ep.prompt_template == "system-prompt"
    assert ep.system_prompt == "hand"
    data = json.loads(p.read_text())
    m1 = next(m for e in data["endpoints"] for m in e["models"] if m["name"] == "m1")
    assert m1 == {"name": "m1", "system_prompt": "hand", "prompt_template": "system-prompt"}


def test_set_prompt_template_none_clears_only_template(tmp_path):
    store, p = make_store(tmp_path)
    store.update_model("https://a.com", "m1", system_prompt="hand")
    store.set_prompt_template("https://a.com", "m1", "system-prompt")
    ep = store.set_prompt_template("https://a.com", "m1", None)
    assert ep.prompt_template is None
    assert ep.system_prompt == "hand"


def test_set_prompt_template_unknown_model_raises(tmp_path):
    store, _ = make_store(tmp_path)
    with pytest.raises(NotFoundError):
        store.set_prompt_template("https://a.com", "nope", "system-prompt")


def test_update_model_omitting_system_prompt_keeps_it(tmp_path):
    store, p = make_store(tmp_path)
    store.update_model("https://a.com", "m1", system_prompt="hand")
    store.update_model("https://a.com", "m1", upstream_model="real")
    m1 = json.loads(p.read_text())["endpoints"][0]["models"][0]
    assert m1["system_prompt"] == "hand"


def test_endpoint_dicts_exposes_prompt_template(tmp_path):
    store, _ = make_store(tmp_path)
    store.set_prompt_template("https://a.com", "m1", "system-prompt")
    models = store.endpoint_dicts(base_url="https://a.com")[0]["models"]
    assert models[0]["prompt_template"] == "system-prompt"
    assert models[1]["prompt_template"] is None
