import datetime
from urllib.parse import quote

import httpx
from fastapi.testclient import TestClient

from gateway.app import build_app
from gateway.config import EndpointConfig, GatewayConfig


def make_config() -> GatewayConfig:
    return GatewayConfig(
        host="127.0.0.1", port=8080, api_key="local-secret",
        endpoints=[
            EndpointConfig(model="gpt-4o", base_url="https://upstream.example",
                           custom_headers={"Authorization": "Bearer real-key"},
                           pricing={"input": 3.0}),
            EndpointConfig(model="claude-sonnet", base_url="https://anthropic.example",
                           pricing={"input": 3.0, "cache_read": 0.3,
                                    "cache_write": 3.75, "output": 15.0}),
        ],
    )


def make_client(tmp_path):
    config = make_config()
    http_client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda r: httpx.Response(404)))
    app = build_app(config, client=http_client, stats_db_path=tmp_path / "usage.db")
    return TestClient(app), app.state.stats_db, app.state.cost_worker


def _today_range() -> tuple[str, int]:
    today = datetime.date.today()
    start = int(datetime.datetime(today.year, today.month, today.day).timestamp())
    return today.isoformat(), start


def _seed(db, start, i=10, cr=2, cw=1, o=5, status=200, base_url="https://upstream.example",
          model="gpt-4o", ts_offset=10, duration_ms=None, ttft_ms=None):
    db.insert(ts=start + ts_offset, base_url=base_url, model=model, input_tokens=i,
              cache_read_tokens=cr, cache_write_tokens=cw, output_tokens=o, status=status,
              duration_ms=duration_ms, ttft_ms=ttft_ms)


def test_stats_disabled_without_path():
    config = make_config()
    app = build_app(config)
    client = TestClient(app)
    resp = client.get("/admin/api/stats/day?date=2026-08-19")
    assert resp.status_code == 503
    assert "disabled" in resp.json()["error"]["message"]
    resp = client.get("/admin/api/requests?from=2026-08-19&to=2026-08-19")
    assert resp.status_code == 503


def test_day_returns_minutes_and_models(tmp_path):
    client, db, worker = make_client(tmp_path)
    day, start = _today_range()
    _seed(db, start, model="gpt-4o", i=100, cr=50, cw=0, o=20)
    _seed(db, start, model="claude-sonnet", i=40, cr=10, cw=5, o=8,
          base_url="https://anthropic.example", ts_offset=70)
    worker.process_pending()
    resp = client.get(f"/admin/api/stats/day?date={day}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["date"] == day
    assert data["models"] == ["gpt-4o", "claude-sonnet"]   # token desc: 170 > 63
    assert len(data["minutes"]) == 2
    m0 = data["minutes"][0]
    assert m0["tokens"]["gpt-4o"]["input"] == 100
    assert m0["tokens"]["gpt-4o"]["cache_read"] == 50
    assert m0["costs"]["gpt-4o"] > 0


def test_day_requires_valid_date(tmp_path):
    client, db, worker = make_client(tmp_path)
    assert client.get("/admin/api/stats/day").status_code == 400
    assert client.get("/admin/api/stats/day?date=2026-13-99").status_code == 400


def test_day_empty(tmp_path):
    client, db, worker = make_client(tmp_path)
    resp = client.get(f"/admin/api/stats/day?date={datetime.date.today().isoformat()}")
    assert resp.json() == {"date": datetime.date.today().isoformat(), "models": [], "minutes": []}


def test_endpoint_card(tmp_path):
    client, db, worker = make_client(tmp_path)
    day, start = _today_range()
    _seed(db, start, model="gpt-4o", i=40, cr=60, cw=0, o=0)
    _seed(db, start, model="gpt-4o", i=100, cr=0, cw=0, o=0, ts_offset=70)
    _seed(db, start, model="claude-sonnet", i=5, cr=0, cw=0, o=5,
          base_url="https://anthropic.example")
    worker.process_pending()
    url = f"/admin/api/stats/endpoints?date={day}&base_url={quote('https://upstream.example')}"
    resp = client.get(url)
    assert resp.status_code == 200
    data = resp.json()
    assert data["models"] == ["gpt-4o"]
    assert data["total_tokens"] == 200
    assert data["total_cost"] > 0
    row = data["per_model"][0]
    assert row["cache_hit_rate"] == 0.3   # 60 / 200
    assert row["calls"] == 2
    # model filter
    resp = client.get(url + "&models=claude-sonnet")
    assert resp.json()["per_model"] == []
    # missing base_url → 400
    assert client.get(f"/admin/api/stats/endpoints?date={day}").status_code == 400


def test_requests_list(tmp_path):
    client, db, worker = make_client(tmp_path)
    day, start = _today_range()
    for i in range(3):
        _seed(db, start, model="gpt-4o" if i % 2 == 0 else "claude-sonnet",
              ts_offset=10 + i, i=10 + i, cr=0, cw=0, o=1)
    worker.process_pending()
    resp = client.get(f"/admin/api/requests?from={day}&to={day}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 3
    assert len(data["items"]) == 3
    assert data["items"][0]["ts"] >= data["items"][1]["ts"]  # desc
    item = data["items"][0]
    for key in ("ts", "base_url", "model", "input_tokens", "cache_read_tokens",
                "cache_write_tokens", "output_tokens", "status", "cost_usd"):
        assert key in item
    # model filter
    resp = client.get(f"/admin/api/requests?from={day}&to={day}&model=gpt-4o")
    assert resp.json()["total"] == 2
    # pagination
    resp = client.get(f"/admin/api/requests?from={day}&to={day}&limit=2&offset=1")
    data = resp.json()
    assert len(data["items"]) == 2
    assert data["total"] == 3
    # datetime range THH:MM
    resp = client.get(f"/admin/api/requests?from={day}T00:00&to={day}T23:59")
    assert resp.status_code == 200
    assert resp.json()["total"] == 3


def test_requests_validation(tmp_path):
    client, db, worker = make_client(tmp_path)
    assert client.get("/admin/api/requests").status_code == 400
    assert client.get("/admin/api/requests?from=2026-08-19").status_code == 400
    assert client.get("/admin/api/requests?from=2026-08-20&to=2026-08-19").status_code == 400
    assert client.get("/admin/api/requests?from=2026-08-19&to=2026-08-19&limit=0").status_code == 400
    assert client.get("/admin/api/requests?from=2026-08-19&to=2026-08-19&limit=201").status_code == 400
    assert client.get("/admin/api/requests?from=2026-08-19&to=2026-08-19&offset=-1").status_code == 400
    assert client.get("/admin/api/requests?from=not-a-date&to=2026-08-19").status_code == 400


def test_endpoint_timeseries(tmp_path):
    client, db, worker = make_client(tmp_path)
    day, start = _today_range()
    _seed(db, start, base_url="https://upstream.example", model="gpt-4o", i=10, cr=2, cw=1, o=5)
    _seed(db, start, base_url="https://upstream.example", model="claude-sonnet", i=3, cr=0, cw=0, o=1)
    _seed(db, start + 120, base_url="https://upstream.example", model="gpt-4o", i=5, cr=0, cw=0, o=2)
    worker.process_pending()
    url = f"/admin/api/stats/endpoints?from={day}T00:00&to={day}T23:59&base_url={quote('https://upstream.example')}"
    resp = client.get(url)
    assert resp.status_code == 200
    data = resp.json()
    assert "from" in data
    assert "to" in data
    assert "minutes" in data
    assert data["granularity"] == "minute"
    assert len(data["minutes"]) >= 2
    assert data["per_model"][0]["model"] == "gpt-4o"
    # model filter
    url2 = url + "&models=gpt-4o"
    resp2 = client.get(url2)
    assert len(resp2.json()["minutes"]) == 2
    # backward compat: single date mode still works
    resp3 = client.get(f"/admin/api/stats/endpoints?date={day}&base_url={quote('https://upstream.example')}")
    assert resp3.status_code == 200
    assert "date" in resp3.json()
    # missing base_url → 400
    assert client.get(f"/admin/api/stats/endpoints?from={day}T00:00&to={day}T23:59").status_code == 400
    # bad granularity → 400
    assert client.get(f"/admin/api/stats/endpoints?from={day}T00:00&to={day}T23:59"
                      f"&base_url={quote('https://upstream.example')}&granularity=decade").status_code == 400


def test_model_tps_endpoint(tmp_path):
    client, db, worker = make_client(tmp_path)
    day, start = _today_range()
    # gpt-4o: 5 samples; decode tps = out/(dur-ttft)
    #   (100/(2.0-0.5), 200/(4.0-0.5), 300/(2.0-0.5), 100/(2.0-0.5), 100/(2.0-0.5))
    #   = (66.7, 57.1, 200, 66.7, 66.7) -> avg 91.4; avg ttft = 500
    _seed(db, start, model="gpt-4o", o=100, duration_ms=2000, ttft_ms=500, ts_offset=10)
    _seed(db, start, model="gpt-4o", o=200, duration_ms=4000, ttft_ms=500, ts_offset=20)
    _seed(db, start, model="gpt-4o", o=300, duration_ms=2000, ttft_ms=500, ts_offset=30)
    _seed(db, start, model="gpt-4o", o=100, duration_ms=2000, ttft_ms=500, ts_offset=40)
    _seed(db, start, model="gpt-4o", o=100, duration_ms=2000, ttft_ms=500, ts_offset=50)
    # claude-sonnet: only 2 samples -> omitted
    _seed(db, start, model="claude-sonnet", o=100, duration_ms=1000, ts_offset=60,
          base_url="https://anthropic.example")
    _seed(db, start, model="claude-sonnet", o=100, duration_ms=1000, ts_offset=70,
          base_url="https://anthropic.example")
    resp = client.get(f"/admin/api/stats/model-tps?date={day}")
    assert resp.status_code == 200
    data = resp.json()
    assert data == [{"model": "gpt-4o", "tps": 91.4, "samples": 5, "avg_ttft_ms": 500.0}]


def test_model_tps_requires_date(tmp_path):
    client, db, worker = make_client(tmp_path)
    assert client.get("/admin/api/stats/model-tps").status_code == 400
    assert client.get("/admin/api/stats/model-tps?date=bad").status_code == 400


def test_endpoint_timeseries_token_totals(tmp_path):
    # 时间范围模式返回 token 分项总和 + 总缓存命中率（接入点卡片直接展示）
    client, db, worker = make_client(tmp_path)
    day, start = _today_range()
    _seed(db, start, i=10, cr=2, cw=1, o=5)
    _seed(db, start + 120, i=100, cr=20, cw=10, o=50)
    worker.process_pending()
    url = f"/admin/api/stats/endpoints?from={day}T00:00&to={day}T23:59&base_url={quote('https://upstream.example')}"
    data = client.get(url).json()
    assert data["total_input"] == 110
    assert data["total_cache_read"] == 22
    assert data["total_cache_write"] == 11
    assert data["total_output"] == 55
    assert data["total_tokens"] == 110 + 22 + 11 + 55
    # 总命中率 = cache_read / (input + cache_read + cache_write)
    assert data["cache_hit_rate"] == round(22 / (110 + 22 + 11), 4)


def test_requests_includes_filter_scoped_totals(tmp_path):
    # 使用明细响应携带与过滤条件同口径的分项汇总（base_url/model 可选过滤）
    client, db, worker = make_client(tmp_path)
    day, start = _today_range()
    _seed(db, start, i=10, cr=2, cw=1, o=5, base_url="https://upstream.example", model="gpt-4o")
    _seed(db, start + 60, i=100, cr=20, cw=10, o=50, base_url="https://other.example", model="glm-5")
    worker.process_pending()
    url = f"/admin/api/requests?from={day}T00:00&to={day}T23:59"
    data = client.get(url).json()
    assert data["totals"]["input"] == 110
    assert data["totals"]["cache_read"] == 22
    assert data["totals"]["cache_write"] == 11
    assert data["totals"]["output"] == 55
    assert data["totals"]["cache_hit_rate"] == round(22 / (110 + 22 + 11), 4)
    # base_url 过滤后口径跟随
    data2 = client.get(url + "&base_url=" + quote("https://other.example")).json()
    assert data2["totals"]["input"] == 100
    assert data2["totals"]["output"] == 50
    # model 过滤后口径跟随
    data3 = client.get(url + "&model=gpt-4o").json()
    assert data3["totals"]["input"] == 10
