import json
from urllib.parse import quote

import httpx
from fastapi.testclient import TestClient

from gateway.app import build_app
from gateway.config import load_config


def _enc(s: str) -> str:
    return quote(s, safe="")


def make_app(tmp_path, config_path=True, captures=None):
    p = tmp_path / "config.json"
    p.write_text(json.dumps({
        "server": {"api_key": "k"},
        "endpoints": [
            {
                "base_url": "https://upstream.example",
                "custom_headers": {"Authorization": "Bearer real"},
                "models": [{"name": "gpt-4o"}],
            },
            {
                "base_url": "https://anthropic.example",
                "models": [{"name": "claude-3-5-sonnet"}],
            },
        ],
    }))
    config = load_config(p)

    def handler(request: httpx.Request) -> httpx.Response:
        if captures is not None:
            captures["url"] = str(request.url)
            captures["body"] = request.content
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            stream=httpx.ByteStream(b"{}"),
        )

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    path = str(p) if config_path else None
    # explicit tmp cache keeps build_app's default (/tmp/llm-gateway-models-cache.json)
    # out of the tests
    app = build_app(config, client=http_client, config_path=path,
                    watcher_cache_path=tmp_path / "watcher-cache.json" if path else None)
    return TestClient(app), p


def test_admin_page_served(tmp_path):
    client, _ = make_app(tmp_path)
    resp = client.get("/admin")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/html")
    assert "接入点管理" in resp.text


def test_admin_page_has_copy_curl_wiring(tmp_path):
    # 模型页「复制curl」按钮与协议选择弹窗的接线（纯函数由 tests/curl_test.mjs 覆盖）
    client, _ = make_app(tmp_path)
    html = client.get("/admin").text
    assert "复制curl" in html
    assert "copyModelCurl" in html
    assert "showProtocolPicker" in html
    assert "/admin/api/server" in html
    # 模型页不再渲染「编辑」按钮（编辑入口保留在接入点卡片展开视图）
    assert 'actionBtn("编辑", () => openModelModal(baseUrl, model))' not in html
    assert 'actionBtn("编辑", () => openModelModal(ep.base_url, m))' in html


def test_admin_page_persists_tab_in_url_hash(tmp_path):
    # 切换标签页后刷新仍停留在当前标签页：hash 写入 + 恢复（纯解析函数由 tests/tab_test.mjs 覆盖）
    client, _ = make_app(tmp_path)
    html = client.get("/admin").text
    assert "tabFromHash(location.hash)" in html
    assert "history.replaceState" in html
    assert 'switchTab("endpoints")' in html


def test_admin_page_models_tab_sorting(tmp_path):
    # 模型页排序：点击表头列排序（纯函数由 tests/models_sort_test.mjs 覆盖）
    client, _ = make_app(tmp_path)
    html = client.get("/admin").text
    assert "buildModelsHeader" in html
    assert "nextSortState(modelsSort, key)" in html
    assert "th.classList.add(\"sortable\")" in html
    # 回归：tbody 必须遍历排序后的数组（曾误遍历未排序的 rows，点击表头无效果）
    assert "of sorted)" in html.replace("{ model, baseUrl } of sorted", "of sorted)")
    assert "of rows)" not in html.replace("of sorted)", "")
    # 工具条方案已移除
    assert "buildModelsToolbar" not in html


def test_server_info(tmp_path):
    client, _ = make_app(tmp_path)
    resp = client.get("/admin/api/server")
    assert resp.status_code == 200
    # api_key 哨兵是生成可用 curl（激活接入点鉴权默认值）所必需的
    assert resp.json() == {"host": "127.0.0.1", "port": 8080, "api_key": "k"}


def test_list_endpoints_shape(tmp_path):
    client, _ = make_app(tmp_path)
    assert client.get("/admin/api/endpoints").json() == [
        {
            "base_url": "https://upstream.example",
            "custom_headers": {"Authorization": "Bearer real"},
            "models_news": [],
            "models": [{"name": "gpt-4o", "upstream_model": None, "system_prompt": None,
                        "prompt_template": None, "pricing": None, "protocol": None}],
        },
        {
            "base_url": "https://anthropic.example",
            "custom_headers": {},
            "models_news": [],
            "models": [{"name": "claude-3-5-sonnet", "upstream_model": None, "system_prompt": None,
                        "prompt_template": None, "pricing": None, "protocol": None}],
        },
    ]


def test_list_shows_empty_models_endpoint(tmp_path):
    client, _ = make_app(tmp_path)
    client.post("/admin/api/endpoints", json={"base_url": "https://empty.example"})
    data = client.get("/admin/api/endpoints").json()
    assert {"base_url": "https://empty.example", "custom_headers": {},
            "models_news": [], "models": []} in data


def test_add_endpoint_then_model_hot_reloads(tmp_path):
    captures = {}
    client, _ = make_app(tmp_path, captures=captures)
    resp = client.post("/admin/api/endpoints", json={"base_url": "https://new.example"})
    assert resp.status_code == 201
    resp = client.post(
        "/admin/api/endpoints/models?base_url=" + _enc("https://new.example"),
        json={"name": "new-model"},
    )
    assert resp.status_code == 201
    models = client.get("/v1/models").json()["data"]
    assert any(m["id"] == "new-model" for m in models)
    resp = client.post("/v1/chat/completions", json={"model": "new-model", "messages": []})
    assert resp.status_code == 200
    assert captures["url"] == "https://new.example/v1/chat/completions"


def test_add_endpoint_duplicate_base_url_400(tmp_path):
    client, _ = make_app(tmp_path)
    resp = client.post("/admin/api/endpoints", json={"base_url": "https://upstream.example"})
    assert resp.status_code == 400
    assert resp.json() == {"error": {"message": "endpoint base_url values must be unique"}}


def test_add_endpoint_missing_base_url_400(tmp_path):
    client, _ = make_app(tmp_path)
    resp = client.post("/admin/api/endpoints", json={})
    assert resp.status_code == 400
    assert resp.json() == {"error": {"message": "base_url is required"}}


def test_add_endpoint_non_object_body_422(tmp_path):
    client, _ = make_app(tmp_path)
    resp = client.post("/admin/api/endpoints", json=["not", "an", "object"])
    assert resp.status_code == 422


def test_update_endpoint_headers_200(tmp_path):
    client, _ = make_app(tmp_path)
    resp = client.put(
        "/admin/api/endpoints?base_url=" + _enc("https://upstream.example"),
        json={"custom_headers": {"User-Agent": "my-agent/1.0"}},
    )
    assert resp.status_code == 200
    assert resp.json()["custom_headers"] == {"User-Agent": "my-agent/1.0"}


def test_update_endpoint_not_found_404(tmp_path):
    client, _ = make_app(tmp_path)
    resp = client.put(
        "/admin/api/endpoints?base_url=" + _enc("https://nope.example"),
        json={"custom_headers": {}},
    )
    assert resp.status_code == 404
    assert resp.json() == {"error": {"message": "endpoint not found: https://nope.example"}}


def test_delete_endpoint_204_then_proxy_404(tmp_path):
    client, _ = make_app(tmp_path)
    resp = client.delete("/admin/api/endpoints?base_url=" + _enc("https://anthropic.example"))
    assert resp.status_code == 204
    models = client.get("/v1/models").json()["data"]
    assert all(m["id"] != "claude-3-5-sonnet" for m in models)
    resp = client.post("/v1/messages", json={"model": "claude-3-5-sonnet", "messages": []})
    assert resp.status_code == 404


def test_add_model_duplicate_400(tmp_path):
    client, _ = make_app(tmp_path)
    resp = client.post(
        "/admin/api/endpoints/models?base_url=" + _enc("https://upstream.example"),
        json={"name": "gpt-4o"},
    )
    assert resp.status_code == 400
    assert resp.json() == {"error": {"message": "endpoint model names must be unique"}}


def test_update_model_rewrites_proxy_body(tmp_path):
    captures = {}
    client, _ = make_app(tmp_path, captures=captures)
    resp = client.put(
        "/admin/api/endpoints/models?base_url=" + _enc("https://upstream.example") + "&name=gpt-4o",
        json={"upstream_model": "gpt-4o-turbo", "system_prompt": "you are claude code"},
    )
    assert resp.status_code == 200
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 200
    body = json.loads(captures["body"])
    assert body["model"] == "gpt-4o-turbo"
    assert body["messages"][0]["role"] == "system"


def test_delete_model_204(tmp_path):
    client, _ = make_app(tmp_path)
    resp = client.delete(
        "/admin/api/endpoints/models?base_url=" + _enc("https://anthropic.example")
        + "&name=claude-3-5-sonnet",
    )
    assert resp.status_code == 204
    assert client.get("/v1/models").json()["data"] == [{"id": "gpt-4o", "type": "model"}]


def test_delete_all_models_keeps_endpoint_with_empty_models(tmp_path):
    client, _ = make_app(tmp_path)
    client.delete(
        "/admin/api/endpoints/models?base_url=" + _enc("https://anthropic.example")
        + "&name=claude-3-5-sonnet",
    )
    data = client.get("/admin/api/endpoints").json()
    ep = next(e for e in data if e["base_url"] == "https://anthropic.example")
    assert ep["models"] == []


def test_model_name_with_slash_full_cycle(tmp_path):
    client, _ = make_app(tmp_path)
    resp = client.post(
        "/admin/api/endpoints/models?base_url=" + _enc("https://upstream.example"),
        json={"name": "mcli/deepseek-v4-flash"},
    )
    assert resp.status_code == 201
    assert any(
        m["id"] == "mcli/deepseek-v4-flash"
        for m in client.get("/v1/models").json()["data"]
    )
    resp = client.delete(
        "/admin/api/endpoints/models?base_url=" + _enc("https://upstream.example")
        + "&name=" + _enc("mcli/deepseek-v4-flash"),
    )
    assert resp.status_code == 204


def test_write_back_persists_to_config_file(tmp_path):
    client, p = make_app(tmp_path)
    client.post(
        "/admin/api/endpoints",
        json={"base_url": "https://new.example", "custom_headers": {"User-Agent": "ua/1.0"}},
    )
    data = json.loads(p.read_text())
    entry = next(e for e in data["endpoints"] if e["base_url"] == "https://new.example")
    assert entry["custom_headers"] == {"User-Agent": "ua/1.0"}


def test_write_api_without_config_path_503_reads_ok(tmp_path):
    client, _ = make_app(tmp_path, config_path=False)
    assert client.get("/admin").status_code == 200
    assert client.get("/admin/api/endpoints").status_code == 200
    resp = client.post("/admin/api/endpoints", json={"base_url": "https://x.example"})
    assert resp.status_code == 503
    assert resp.json() == {
        "error": {"message": "config file path is not set; admin write operations are disabled"}
    }


def test_add_endpoint_write_failure_503(tmp_path, monkeypatch):
    def boom(src, dst):
        raise OSError("disk full")
    monkeypatch.setattr("gateway.store.os.replace", boom)
    client, p = make_app(tmp_path)
    before = p.read_text()
    resp = client.post("/admin/api/endpoints", json={"base_url": "https://x.example"})
    assert resp.status_code == 503
    assert "failed to write config" in resp.json()["error"]["message"]
    assert p.read_text() == before


def test_add_model_unknown_endpoint_404(tmp_path):
    client, _ = make_app(tmp_path)
    resp = client.post(
        "/admin/api/endpoints/models?base_url=" + _enc("https://nope.example"),
        json={"name": "m"},
    )
    assert resp.status_code == 404
    assert resp.json() == {"error": {"message": "endpoint not found: https://nope.example"}}


def test_update_endpoint_missing_base_url_400(tmp_path):
    client, _ = make_app(tmp_path)
    resp = client.put("/admin/api/endpoints", json={"custom_headers": {}})
    assert resp.status_code == 400
    assert resp.json() == {"error": {"message": "base_url query parameter is required"}}


def test_delete_model_missing_name_400(tmp_path):
    client, _ = make_app(tmp_path)
    resp = client.delete("/admin/api/endpoints/models?base_url=" + _enc("https://a.example"))
    assert resp.status_code == 400
    assert resp.json() == {"error": {"message": "base_url and name query parameters are required"}}


def test_model_pricing_roundtrip(tmp_path):
    client, p = make_app(tmp_path)
    resp = client.post("/admin/api/endpoints/models?base_url=" + _enc("https://upstream.example"),
                       json={"name": "new-model", "pricing": {"input": 3}})
    assert resp.status_code == 201
    assert resp.json()["pricing"] == {"input": 3.0}
    resp = client.put("/admin/api/endpoints/models?base_url=" + _enc("https://upstream.example")
                      + "&name=" + _enc("new-model"),
                      json={"upstream_model": None, "system_prompt": None,
                            "pricing": {"input": 4, "output": 10}})
    assert resp.status_code == 200
    assert resp.json()["pricing"] == {"input": 4.0, "output": 10.0}
    # list includes pricing
    data = client.get("/admin/api/endpoints").json()
    model = next(m for ep in data for m in ep["models"] if m["name"] == "new-model")
    assert model["pricing"] == {"input": 4.0, "output": 10.0}
    # clear pricing with explicit null
    resp = client.put("/admin/api/endpoints/models?base_url=" + _enc("https://upstream.example")
                      + "&name=" + _enc("new-model"),
                      json={"upstream_model": None, "system_prompt": None, "pricing": None})
    assert resp.status_code == 200
    assert resp.json()["pricing"] is None


def _fetch_app(tmp_path, handler, base_url="https://upstream.example",
               custom_headers=None, cache_path=None):
    p = tmp_path / "config.json"
    p.write_text(json.dumps({
        "server": {"api_key": "k"},
        "endpoints": [{
            "base_url": base_url,
            "custom_headers": custom_headers or {"User-Agent": "llm-gateway/1.0"},
            "models": [{"name": "gpt-4o"}],
        }],
    }))
    config = load_config(p)
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    # default to a per-test cache file: keep build_app's default
    # (/tmp/llm-gateway-models-cache.json) out of the tests
    if cache_path is None:
        cache_path = tmp_path / "watcher-cache.json"
    app = build_app(config, client=http_client, config_path=str(p),
                    watcher_cache_path=cache_path)
    return TestClient(app)


def test_fetch_upstream_models_returns_deduped_ids(tmp_path):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["headers"] = dict(request.headers)
        return httpx.Response(200, json={
            "object": "list",
            "data": [
                {"id": "deepseek-v4-pro", "object": "model"},
                {"id": "glm-5.3", "object": "model"},
                {"id": "deepseek-v4-pro", "object": "model"},   # duplicate
                {"object": "model"},                             # no id -> skipped
            ],
        })

    client = _fetch_app(tmp_path, handler)
    resp = client.get("/admin/api/upstream-models?base_url=" + _enc("https://upstream.example"))
    assert resp.status_code == 200
    assert resp.json() == {"models": ["deepseek-v4-pro", "glm-5.3"], "new_models": []}
    # GET against base_url + /v1/models, carrying the endpoint's configured UA
    assert captured["url"] == "https://upstream.example/v1/models"
    assert captured["headers"]["user-agent"] == "llm-gateway/1.0"


def test_fetch_upstream_models_bare_list_and_empty(tmp_path):
    def bare(request):
        return httpx.Response(200, json=[{"id": "x"}, {"id": "y"}])

    client = _fetch_app(tmp_path, bare)
    resp = client.get("/admin/api/upstream-models?base_url=" + _enc("https://upstream.example"))
    assert resp.status_code == 200
    assert resp.json() == {"models": ["x", "y"], "new_models": []}

    client = _fetch_app(tmp_path, lambda r: httpx.Response(200, json={}))
    resp = client.get("/admin/api/upstream-models?base_url=" + _enc("https://upstream.example"))
    assert resp.status_code == 200
    assert resp.json() == {"models": [], "new_models": []}


def test_fetch_upstream_models_sends_configured_custom_headers(tmp_path):
    captured = {}

    def handler(request):
        captured["headers"] = dict(request.headers)
        return httpx.Response(200, json={"data": [{"id": "m1"}]})

    client = _fetch_app(
        tmp_path, handler, base_url="https://u.example",
        custom_headers={"User-Agent": "my-ua", "Authorization": "Bearer cfg-key"},
    )
    resp = client.get("/admin/api/upstream-models?base_url=" + _enc("https://u.example"))
    assert resp.status_code == 200
    assert captured["headers"]["user-agent"] == "my-ua"
    assert captured["headers"]["authorization"] == "Bearer cfg-key"


def test_fetch_upstream_models_unknown_endpoint_404(tmp_path):
    client = _fetch_app(tmp_path, lambda r: httpx.Response(200, json={}))
    resp = client.get("/admin/api/upstream-models?base_url=" + _enc("https://nope.example"))
    assert resp.status_code == 404


def test_fetch_upstream_models_missing_base_url_400(tmp_path):
    client = _fetch_app(tmp_path, lambda r: httpx.Response(200, json={}))
    resp = client.get("/admin/api/upstream-models")
    assert resp.status_code == 400
    assert "base_url" in resp.json()["error"]["message"]


def test_fetch_upstream_models_upstream_error_502(tmp_path):
    client = _fetch_app(tmp_path, lambda r: httpx.Response(401, text="no key"))
    resp = client.get("/admin/api/upstream-models?base_url=" + _enc("https://upstream.example"))
    assert resp.status_code == 502
    assert "401" in resp.json()["error"]["message"]


def test_fetch_upstream_models_unreachable_502(tmp_path):
    def handler(request):
        raise httpx.ConnectError("connection refused")

    client = _fetch_app(tmp_path, handler)
    resp = client.get("/admin/api/upstream-models?base_url=" + _enc("https://upstream.example"))
    assert resp.status_code == 502


def test_fetch_upstream_models_non_json_502(tmp_path):
    client = _fetch_app(tmp_path, lambda r: httpx.Response(200, content=b"<html>"))
    resp = client.get("/admin/api/upstream-models?base_url=" + _enc("https://upstream.example"))
    assert resp.status_code == 502


def test_chart_js_served(tmp_path):
    client, _ = make_app(tmp_path)
    resp = client.get("/static/chart.umd.min.js")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("application/javascript")
    assert b"Chart" in resp.content


def test_admin_page_has_stats_tab(tmp_path):
    client, _ = make_app(tmp_path)
    resp = client.get("/admin")
    assert 'id="tab-stats"' in resp.text
    assert 'src="/static/chart.umd.min.js"' in resp.text


def test_admin_page_has_request_protocol_and_effort_columns(tmp_path):
    client, _ = make_app(tmp_path)
    html = client.get("/admin").text
    assert "协议（请求 → 实际）" in html
    assert "思考强度" in html
    assert "protocolFlowLabel(it)" in html
    assert "reasoningEffortLabel(it)" in html


def test_model_pricing_invalid_400(tmp_path):
    client, p = make_app(tmp_path)
    resp = client.post("/admin/api/endpoints/models?base_url=" + _enc("https://upstream.example"),
                       json={"name": "bad1", "pricing": {"input": "abc"}})
    assert resp.status_code == 400
    resp = client.post("/admin/api/endpoints/models?base_url=" + _enc("https://upstream.example"),
                       json={"name": "bad2", "pricing": {"input": -1}})
    assert resp.status_code == 400
    resp = client.post("/admin/api/endpoints/models?base_url=" + _enc("https://upstream.example"),
                       json={"name": "bad3", "pricing": "not-an-object"})
    assert resp.status_code == 400


def test_model_protocol_roundtrip_and_clear(tmp_path):
    client, p = make_app(tmp_path)
    resp = client.post("/admin/api/endpoints/models?base_url=" + _enc("https://upstream.example"),
                       json={"name": "new-model", "protocol": "anthropic_messages"})
    assert resp.status_code == 201
    assert resp.json()["protocol"] == "anthropic_messages"
    # update protocol
    resp = client.put("/admin/api/endpoints/models?base_url=" + _enc("https://upstream.example")
                      + "&name=" + _enc("new-model"),
                      json={"protocol": "openai_chat"})
    assert resp.status_code == 200
    assert resp.json()["protocol"] == "openai_chat"
    # clear protocol with explicit null
    resp = client.put("/admin/api/endpoints/models?base_url=" + _enc("https://upstream.example")
                      + "&name=" + _enc("new-model"),
                      json={"protocol": None})
    assert resp.status_code == 200
    assert resp.json()["protocol"] is None
    # list shape includes protocol key
    data = client.get("/admin/api/endpoints").json()
    model = next(m for ep in data for m in ep["models"] if m["name"] == "new-model")
    assert "protocol" in model
    assert model["protocol"] is None


def test_model_protocol_invalid_value_400(tmp_path):
    client, p = make_app(tmp_path)
    resp = client.post("/admin/api/endpoints/models?base_url=" + _enc("https://upstream.example"),
                       json={"name": "new-model", "protocol": "gopher"})
    assert resp.status_code == 400
    assert "protocol" in resp.json()["error"]["message"]


def test_global_extra_body_roundtrip_and_clear(tmp_path):
    client, p = make_app(tmp_path)
    # initially empty
    resp = client.get("/admin/api/extra-body")
    assert resp.status_code == 200
    assert resp.json() == {"extra_body": None}
    # set
    resp = client.put("/admin/api/extra-body", json={"extra_body": {"openai_responses": {"store": False}}})
    assert resp.status_code == 200
    assert resp.json()["extra_body"] == {"openai_responses": {"store": False}}
    # persists to config file
    data = json.loads(p.read_text())
    assert data["server"]["extra_body"] == {"openai_responses": {"store": False}}
    # update
    resp = client.put("/admin/api/extra-body", json={"extra_body": {"anthropic_messages": {"include": ["a"]}}})
    assert resp.status_code == 200
    assert resp.json()["extra_body"] == {"anthropic_messages": {"include": ["a"]}}
    # clear with null
    resp = client.put("/admin/api/extra-body", json={"extra_body": None})
    assert resp.status_code == 200
    assert resp.json()["extra_body"] is None
    data = json.loads(p.read_text())
    assert "extra_body" not in data["server"]


def test_global_extra_body_invalid_400(tmp_path):
    client, p = make_app(tmp_path)
    resp = client.put("/admin/api/extra-body", json={"extra_body": [1, 2]})
    assert resp.status_code == 400
    assert "extra_body" in resp.json()["error"]["message"]
    resp = client.put("/admin/api/extra-body", json={"extra_body": {"gopher": {"a": 1}}})
    assert resp.status_code == 400
    assert "extra_body" in resp.json()["error"]["message"]


def test_list_endpoints_includes_models_news(tmp_path):
    client, _ = make_app(tmp_path)
    body = client.get("/admin/api/endpoints").json()
    assert all("models_news" in e for e in body)
    assert body[0]["models_news"] == []


def test_fetch_upstream_models_marks_new_ids(tmp_path):
    cache_path = tmp_path / "cache.json"
    cache_path.write_text(json.dumps({
        "https://upstream.example": {"ids": ["known-1"], "ts": 1},
    }))
    client = _fetch_app(tmp_path, lambda r: httpx.Response(200, json={
        "data": [{"id": "known-1"}, {"id": "brand-new"}],
    }), cache_path=cache_path)
    resp = client.get("/admin/api/upstream-models?base_url=" + _enc("https://upstream.example"))
    assert resp.status_code == 200
    assert resp.json()["new_models"] == ["brand-new"]
    # manual fetch refreshes the baseline: both ids now cached
    cache = json.loads(cache_path.read_text())
    assert cache["https://upstream.example"]["ids"] == ["known-1", "brand-new"]


def test_fetch_upstream_models_no_cache_no_new(tmp_path):
    client = _fetch_app(tmp_path, lambda r: httpx.Response(200, json={
        "data": [{"id": "a"}],
    }), cache_path=tmp_path / "cache.json")
    resp = client.get("/admin/api/upstream-models?base_url=" + _enc("https://upstream.example"))
    assert resp.status_code == 200
    assert resp.json()["new_models"] == []


def test_fetch_upstream_models_no_baseline_no_new(tmp_path):
    # cache 文件存在但该接入点无基线（首轮语义）-> 不标记任何新模型
    client = _fetch_app(tmp_path, lambda r: httpx.Response(200, json={
        "data": [{"id": "a"}],
    }))
    resp = client.get("/admin/api/upstream-models?base_url=" + _enc("https://upstream.example"))
    assert resp.status_code == 200
    assert resp.json()["new_models"] == []


def test_clear_models_news(tmp_path):
    client, p = make_app(tmp_path)
    # seed news via the store directly (watcher's write path)
    from gateway.store import ConfigStore
    store = ConfigStore(load_config(p), p)
    store.set_models_news("https://upstream.example", ["new-1"])
    resp = client.post("/admin/api/endpoints/models-news",
                       json={"base_url": "https://upstream.example"})
    assert resp.status_code == 200
    body = client.get("/admin/api/endpoints").json()
    assert body[0]["models_news"] == []


def test_clear_models_news_missing_base_url_400(tmp_path):
    client, _ = make_app(tmp_path)
    resp = client.post("/admin/api/endpoints/models-news", json={})
    assert resp.status_code == 400


def test_clear_models_news_unknown_endpoint_404(tmp_path):
    client, _ = make_app(tmp_path)
    resp = client.post("/admin/api/endpoints/models-news",
                       json={"base_url": "https://nope.example"})
    assert resp.status_code == 404


def test_clear_models_news_readonly_503(tmp_path):
    client, _ = make_app(tmp_path, config_path=False)
    resp = client.post("/admin/api/endpoints/models-news",
                       json={"base_url": "https://upstream.example"})
    assert resp.status_code == 503


def test_admin_page_has_models_news_wiring(tmp_path):
    # 「模型上新」徽标 + 弹窗新模型标记 + 打开即静默（纯函数由 tests/models_news_test.mjs 覆盖）
    client, _ = make_app(tmp_path)
    html = client.get("/admin").text
    assert "newsBadgeText" in html
    assert "isNewModel" in html
    assert "badge news" in html
    assert "/admin/api/endpoints/models-news" in html
    assert "renderModelPicker(ep, data.models || [], data.new_models || [])" in html
    # 「新」标记占独立固定宽度列（表头 + 每行占位），不内联挤压其余列
    assert "pick-new-col" in html


def test_admin_page_model_picker_defaults_and_alignment(tmp_path):
    # 获取模型弹窗：默认不预选任何模型；「全选」复选框随上新列水平对齐
    #（纯函数由 tests/pick_test.mjs 覆盖）
    client, _ = make_app(tmp_path)
    html = client.get("/admin").text
    # 默认勾选走 initialPickChecked（恒不预选），旧的「未存在即勾选」内联表达式已移除
    assert "initialPickChecked(name, systemNames)" in html
    assert "!systemNames.has(name)" not in html
    # 上新列出现时工具栏加 has-news 缩进，让「全选」与行内复选框对齐
    assert "pickToolbarClass(showNewsCol)" in html
    assert ".pick-toolbar.has-news" in html


def test_admin_page_picker_header_drops_news_label(tmp_path):
    # 上新列表头不写「上新」二字，仅保留空占位列与行内「新」标记对齐
    #（纯函数由 tests/pick_test.mjs 覆盖）
    client, _ = make_app(tmp_path)
    html = client.get("/admin").text
    assert "pickHeadHtml(showNewsCol)" in html
    assert '<span class="pick-new-col">上新</span>' not in html


def test_scan_models_triggers_one_scan(tmp_path):
    # GET /admin/api/scan-models 手动触发一轮全量扫描，返回逐接入点结果
    p = tmp_path / "config.json"
    p.write_text(json.dumps({
        "server": {"api_key": "k"},
        "endpoints": [{
            "base_url": "https://a.com",
            "custom_headers": {"User-Agent": "ua"},
            "models": [{"name": "m1"}],
        }],
    }))
    cache_path = tmp_path / "cache.json"
    state = {"ids": ["m1", "m2"]}

    def handler(request):
        return httpx.Response(200, json={"data": [{"id": i} for i in state["ids"]]})

    app = build_app(load_config(p),
                    client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
                    config_path=str(p), watcher_cache_path=cache_path)
    client = TestClient(app)
    # 首轮：仅建基线，无上新
    resp = client.get("/admin/api/scan-models")
    assert resp.status_code == 200
    assert resp.json()["results"] == [{
        "base_url": "https://a.com", "ok": True, "models": ["m1", "m2"],
        "added": [], "removed": [], "models_news": [],
    }]
    # 上游出现新模型 -> 第二轮扫描报上新并点亮徽标
    state["ids"].append("m3")
    resp = client.get("/admin/api/scan-models")
    assert resp.json()["results"][0]["added"] == ["m3"]
    assert resp.json()["results"][0]["models_news"] == ["m3"]
    assert client.get("/admin/api/endpoints").json()[0]["models_news"] == ["m3"]


def test_scan_models_readonly_503(tmp_path):
    client, _ = make_app(tmp_path, config_path=False)
    resp = client.get("/admin/api/scan-models")
    assert resp.status_code == 503


def test_fetch_upstream_models_marks_watcher_detected_news(tmp_path):
    # 复现真实时序：watcher 发现新模型时同一轮已把新 id 折进 cache 基线
    # （models_news 点亮徽标 + 基线含新 id）。弹窗标记不能只 diff 基线，
    # 否则徽标点亮的模型恰恰永远不被标记。
    cache_path = tmp_path / "cache.json"
    cache_path.write_text(json.dumps({
        "https://upstream.example": {"ids": ["known-1", "new-1"], "ts": 1},
    }))
    p = tmp_path / "config.json"
    p.write_text(json.dumps({
        "server": {"api_key": "k"},
        "endpoints": [{
            "base_url": "https://upstream.example",
            "custom_headers": {"User-Agent": "llm-gateway/1.0"},
            "models": [{"name": "gpt-4o"}],
            "models_news": ["new-1"],
        }],
    }))
    app = build_app(load_config(p),
                    client=httpx.AsyncClient(transport=httpx.MockTransport(
                        lambda r: httpx.Response(200, json={
                            "data": [{"id": "known-1"}, {"id": "new-1"}]}))),
                    config_path=str(p), watcher_cache_path=cache_path)
    client = TestClient(app)
    resp = client.get("/admin/api/upstream-models?base_url=" + _enc("https://upstream.example"))
    assert resp.status_code == 200
    assert resp.json()["new_models"] == ["new-1"]


def test_fetch_upstream_models_unions_news_and_fresh_diff(tmp_path):
    # models_news（watcher 已发现、已在基线中）与本次拉取相对基线的新增
    # （两次扫描之间出现、尚未被 watcher 发现）都应标记
    cache_path = tmp_path / "cache.json"
    cache_path.write_text(json.dumps({
        "https://upstream.example": {"ids": ["known-1", "seen-by-watcher"], "ts": 1},
    }))
    p = tmp_path / "config.json"
    p.write_text(json.dumps({
        "server": {"api_key": "k"},
        "endpoints": [{
            "base_url": "https://upstream.example",
            "custom_headers": {"User-Agent": "llm-gateway/1.0"},
            "models": [{"name": "gpt-4o"}],
            "models_news": ["seen-by-watcher"],
        }],
    }))
    app = build_app(load_config(p),
                    client=httpx.AsyncClient(transport=httpx.MockTransport(
                        lambda r: httpx.Response(200, json={
                            "data": [{"id": "known-1"}, {"id": "seen-by-watcher"},
                                     {"id": "fresh-1"}]}))),
                    config_path=str(p), watcher_cache_path=cache_path)
    client = TestClient(app)
    resp = client.get("/admin/api/upstream-models?base_url=" + _enc("https://upstream.example"))
    assert resp.json()["new_models"] == ["seen-by-watcher", "fresh-1"]


def test_fetch_upstream_models_readonly_marks_config_news_only(tmp_path):
    # 只读模式（无 config_path / cache）：仍标记配置中的 models_news
    p = tmp_path / "config.json"
    p.write_text(json.dumps({
        "server": {"api_key": "k"},
        "endpoints": [{
            "base_url": "https://upstream.example",
            "custom_headers": {},
            "models": [{"name": "gpt-4o"}],
            "models_news": ["flagged-1"],
        }],
    }))
    app = build_app(load_config(p),
                    client=httpx.AsyncClient(transport=httpx.MockTransport(
                        lambda r: httpx.Response(200, json={
                            "data": [{"id": "flagged-1"}, {"id": "other"}]}))),
                    config_path=None)
    client = TestClient(app)
    resp = client.get("/admin/api/upstream-models?base_url=" + _enc("https://upstream.example"))
    assert resp.status_code == 200
    assert resp.json()["new_models"] == ["flagged-1"]


def test_build_app_defaults_watcher_cache_when_config_path(tmp_path):
    # 生产路径（__main__ 不传 watcher_cache_path）：build_app 必须解析默认
    # 路径并让 watcher 与 admin 路由共享同一文件——曾因各自为政导致弹窗
    # new_models 永远为空、人工获取不刷基线。
    p = tmp_path / "config.json"
    p.write_text(json.dumps({
        "server": {"api_key": "k"},
        "endpoints": [{"base_url": "https://a.com", "models": [{"name": "m1"}]}],
    }))
    from gateway.watcher import default_cache_path
    app = build_app(load_config(p), config_path=str(p))
    assert app.state.watcher_cache_path == default_cache_path()
    assert app.state.model_watcher is not None


def test_admin_page_script_is_valid_js(tmp_path):
    # 整个 <script> 必须可被 node 解析——SyntaxError 会让整个 admin 页面失效
    # （曾因重复 const 声明全页挂掉，而子串断言测试无法发现）
    import re
    import subprocess
    import tempfile
    client, _ = make_app(tmp_path)
    html = client.get("/admin").text
    scripts = re.findall(r"<script>([\s\S]*?)</script>", html)
    assert scripts, "no inline script found in admin.html"
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as fh:
        fh.write("\n".join(scripts))
        path = fh.name
    proc = subprocess.run(["node", "--check", path], capture_output=True, text=True)
    import os
    os.unlink(path)
    assert proc.returncode == 0, f"admin.html script has a JS syntax error:\n{proc.stderr}"


def test_admin_page_persists_requests_filters(tmp_path):
    # 明细页筛选状态持久化：F5 后恢复时间范围/接入点/模型（纯函数由 tests/reqstate_test.mjs 覆盖）
    client, _ = make_app(tmp_path)
    html = client.get("/admin").text
    assert "encodeRequestsState" in html
    assert "decodeRequestsState" in html
    assert "restoreRequestsState()" in html
    assert "persistRequestsState()" in html
