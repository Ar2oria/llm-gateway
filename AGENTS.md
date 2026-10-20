# AGENTS.md

## Setup & Run

- `uv sync` (`.venv` in repo). Python `>=3.11,<3.12` required; `uvicorn<0.30` pinned — don't bump (PyCharm 2023.3 debugger `loop_factory` bug, see `pyproject.toml:8`).
- Run: `uv run python -m gateway -c config.json [--debug-log]` — `--debug-log` overrides `server.debug_log` and logs full headers/bodies (includes real keys, trusted only).
- Background: `./gateway.sh start|stop|restart|status [config.json] [--debug]` — symlink-safe; defaults to `config.local.json` (gitignored, relative paths resolve against caller's cwd). PID `/tmp/llm-gateway.pid`, log `/tmp/llm-gateway.log`; on macOS also spawns `gateway.menubar` (`uv sync --extra menubar`; its right-click 重启服务 SIGTERMs the gateway, detaches `gateway.sh restart`, and quits).

## Tests

- `uv run pytest -v` — all tests, no network (upstream mocked via `httpx.MockTransport` + `TestClient`, see `tests/test_app.py:make_client`).
- `uv run pytest tests/test_app.py -k sse -v` — filter by name; `uv run pytest tests/test_config.py -v` — single file.
- `node tests/bucket_test.mjs` / `node tests/curl_test.mjs` / `node tests/tab_test.mjs` / `node tests/models_sort_test.mjs` / `node tests/models_news_test.mjs / node tests/reqstate_test.mjs` — JS tests of admin.html helpers (minute-bucketing, copy-curl builder, URL-hash tab persistence, models-tab sorting, model-news badge; extracted from `<script>` marker comments; NOT run by pytest).
- Test ↔ source mirror: `test_app.py`→`app.py` (+`test_app_helpers.py` for `_build_upstream_headers`/`_rewrite_body`), `test_config.py`→`config.py`, `test_convert.py`→`convert.py`+`converter/`, `test_store.py`→`store.py`, `test_admin.py`→`admin.py`, `test_menubar.py`→`menubar.py`, `test_fetch.py`+`test_watcher.py`→`fetch.py`+`watcher.py`, `test_stats_*.py`→`stats/`.

## Config

Schema validated in `gateway/config.py:validate_config`. `server.api_key` required (sentinel, not auth). `endpoints[].models` may be empty; `base_url` (trailing `/` stripped) and model `name` must be unique globally. `config_to_dict()` groups flat `EndpointConfig` list back by `base_url`.

- `server.extra_body`: per-protocol map keyed by `Protocol` values `openai_chat|openai_responses|anthropic_messages` — merged into *converted* upstream requests after conversion; `null` **deletes** field (required for upstreams that reject unknown params like `stop`). No per-model `extra_body`.
- `endpoints[].custom_headers`: shared by all models under that endpoint.
- `endpoints[].models[].protocol`: optional pin (`Protocol` values); mismatch triggers body conversion. `pricing`: optional `input|cache_read|cache_write|output` ($/1M tokens, missing derived from `input` ×1.25/0.1/5).
- `endpoints[].models_news`: internal — upstream ids flagged「模型上新」by the watcher; cleared when the fetch-models modal opens. Not round-tripped by `config_to_dict()`.

## Architecture

- `gateway/config.py`: `load_config`/`validate_config`→`GatewayConfig`; `ModelRouter.route()` exact-match lookup; `detect_format()` maps path to `Protocol` (`/v1/messages`→anthropic, `/v1/responses`→responses, else openai).
- `gateway/store.py`: `ConfigStore` hot-reload — single immutable `_Snapshot(config, router, raw)`; every mutation is validate→atomic write (`tmp+os.replace`, `fsync` data+dir, preserve mode)→swap. Needs `config_path` else writes raise `StoreError`/503. Seeds `raw` from file to preserve empty-model endpoints; preserves effective `debug_log` across first write.
- `gateway/app.py:build_app(config, client, config_path, stats_db_path, watcher_cache_path, watcher_interval=12h)`: creates FastAPI; `GET /v1/models` registered before catch-all `/{full_path:path}` (GET/POST/PUT/PATCH/DELETE/HEAD/OPTIONS, no TRACE). Proxy reads `store.router`/`store.config` per-request. Module-global `working_model` lets non-body methods (only POST/PUT read body) reuse last model — not per-client, concurrent clients can see each other's. Helpers: `_build_upstream_headers`, `_rewrite_body`, `_sse_response`/`_buffered_response`, `_decompressor`.
- `gateway/admin.py:register_admin_routes`: serves `static/admin.html` (zh-CN) + `static/chart.umd.min.js` at `/admin`; CRUD `/admin/api/endpoints` + `/admin/api/upstream-models` (10s fetch of endpoint `/v1/models` via `custom_headers` only; response also carries `new_models` = union of config `models_news` and ids absent from the cache baseline, and refreshes that cache) + `/admin/api/endpoints/models-news` (clears 模型上新) + `/admin/api/scan-models` (GET: manually triggers one watcher scan, returns per-endpoint results) + `/admin/api/extra-body` + `/admin/api/server` (host/port/api_key sentinel for the 模型页 复制curl button; pure builder in `// CURL-START/END` markers, tested by `tests/curl_test.mjs`). Query-param IDs (models may contain `/`); error envelope `{"error":{"message":...}}` (422 for malformed bodies is FastAPI `{"detail": [...]}`).
- `gateway/convert.py`+`converter/` (IR hub, wire behavior pinned to sub2api via `tests/test_convert_parity.py` goldens): `get_convert_stats`→`ConvertStats`; `_PROTOCOL_PATH` maps protocols to upstream paths. Converts request→pinned protocol, forwards to `get_target_path()`, converts response back. Failures: 400 request / 502 response. `extra_body` merged after conversion wins over defaults. `convert_sse_chunks` reassembles split SSE, re-emits 64B chunks, runs decoder `finalize()` when the stream ends without a termination event.
- `gateway/fetch.py`: shared `fetch_models()` — GET `{base_url}/v1/models` carrying only `custom_headers`; `FetchError` on transport/non-200/non-JSON. Used by admin route + watcher.
- `gateway/watcher.py`: `ModelWatcher` asyncio task in app lifespan — every `watcher_interval` (12h default) fetches each endpoint's model list, diffs against the baseline cache (`build_app` resolves the path once — explicit `watcher_cache_path` or default `/tmp/llm-gateway-models-cache.json` — and shares it with the admin fetch route), accumulates new ids into `endpoints[].models_news` via `store.set_models_news()`. First scan builds baseline only; per-endpoint failure or `StoreError` skips that endpoint (cache+config stay consistent); read-only mode never starts the watcher.
- `gateway/stats/`: `capture.py` extracts usage (scan-only, never mutates forwarded bytes; Responses `input_tokens` includes `input_tokens_details.cached_tokens`→split to cache_read; streams via `SseUsageScanner`), `db.py` SQLite (`usage.db` next to config, `cost_usd IS NULL` queue, WAL, migration for `duration_ms`/`ttft_ms`), `worker.py` background thread freezes cost, `api.py` `/admin/api/stats/*`+`/admin/api/requests`. Disabled when no `config_path`/`stats_db_path` (endpoints return 503). Compressed upstream: decompress copy (gzip/deflate via zlib) for capture only; `br`/`zstd` skip capture. TPS = `output/(duration-TTFT)` (non-streaming fakes 3s TTFT).

## Request Flow (`app.py:proxy_request`)

1. Body: POST/PUT parse JSON, else require `working_model` or 400. `model = data.get("model") or working_model`.
2. Route model → endpoint (400 no JSON/missing model, 404 unknown).
3. URL = `endpoint.base_url` + path (or `ConvertStats.get_target_path()` when converting) + query.
4. Headers: strip hop-by-hop → overlay `custom_headers` (lowercased, non-auth override wins) → sentinel check (`Authorization` Bearer value or `x-api-key` equals `server.api_key` → drop client auth, use endpoint default; no auth header → no auth upstream) → derive missing sibling (`Authorization`↔`x-api-key`) → filter `accept-encoding` to `gzip,deflate,identity` only (`br`/`zstd`/`*` stripped, `identity` if empty; converting strips response `content-encoding`/`content-length`).
5. Body rewrite (if non-empty): `convert_request_body` (if needed) → `_rewrite_body` (`upstream_model`, `system_prompt`→`system`/`instructions`/first system message, `extra_body[final_protocol.value]` with null-delete).
6. Forward via httpx (502 on `HTTPError`); SSE (`content-type: text/event-stream`) streamed with `aclose()` in `finally`, else buffered; converting transforms response back.

## Gotchas

- `server.api_key` is sentinel: never validates, just activates endpoint auth default. Real keys go in `endpoints[].custom_headers`.
- `accept-encoding` filtering is intentional — don't re-add `br`/`zstd`; conversion/capture can't decompress them.
- `working_model` is process-global; non-body GET/HEAD requests are racy across clients.
- Admin API unauthenticated, responses leak `custom_headers` — keep gateway on `127.0.0.1`.
- Writes validate→atomic `fsync`→swap; first write rewrites file with 2-space indent preserving original key order. External edits to `config.json` while running are ignored. Deleting endpoint mid-request: in-flight holds old `EndpointConfig`.
- Stats records client-facing `endpoint.model`, cost frozen at write time. TTFT = first content/thinking/tool delta (or `response.*.delta` for Responses), not first byte; non-streaming has no TTFT.
- Reference specs/plans in `docs/superpowers/`; `maybe-todo.md` is aspirational (partly shipped) — verify against code.
