# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Local, lightweight LLM forwarding gateway. Clients send a friendly `model` name; the gateway routes (exact match) to a configured upstream endpoint and forwards path, body and headers. Supports OpenAI Chat (`/v1/chat/completions`), Anthropic Messages (`/v1/messages`) and OpenAI Responses (`/v1/responses`) — rewrites `model`/`system_prompt` per model config and optionally converts between protocols when a model is pinned to a different format. SSE streams pass through (converted when needed). A model can also be *injected* with a prompt template (a `prompts/<name>/prompt.md` body that becomes its system prompt at request time). Per-request usage/cost stats go to a local SQLite DB. FastAPI + httpx, Python `>=3.11,<3.12`.

## Commands

```bash
uv sync                     # install deps (.venv in repo)
# uvicorn is pinned <0.30 — PyCharm 2023.3 debugger has no loop_factory (see pyproject.toml)

# Run
uv run python -m gateway -c config.json
uv run python -m gateway -c config.json --debug-log   # full request logging

# Background helper (symlink-safe, works from any dir)
./gateway.sh start|stop|restart|status [config.json] [--debug]
#  defaults to config.local.json (gitignored); relative paths resolve against caller's cwd
#  pid /tmp/llm-gateway.pid, log /tmp/llm-gateway.log

# Tests — no network (httpx.MockTransport + TestClient, see tests/test_app.py:make_client)
# pyproject testpaths = ["tests", "packages/apicompat/tests"], so plain pytest runs both
uv run pytest -v
uv run pytest tests/test_app.py -k sse -v         # filter by name
uv run pytest tests/test_config.py -v             # single file
uv run pytest packages/apicompat/tests -v         # SDK only (parity goldens + stream)
uv build --package apicompat                      # wheel + sdist in dist/

# Lint/type/fmt: no tooling configured — no ruff/black/mypy in pyproject.

# JS tests of admin.html helpers — NOT run by pytest, run each explicitly.
# They load gateway/static/admin.html and extract the helpers by `// X-START` /
# `// X-END` marker comments, so the tested code IS the shipped code; renaming
# or moving a marker block breaks its test.
node tests/bucket_test.mjs                        # stats minute-bucketing (BUCKETING)
node tests/curl_test.mjs                          # 复制curl command builder (CURL)
node tests/tab_test.mjs                           # URL-hash tab persistence (TABHASH)
node tests/models_sort_test.mjs                   # models-tab sorting (MODELS-SORT)
node tests/prompt_template_test.mjs               # prompt-template cell/button helpers (PROMPTS)
node tests/models_news_test.mjs                   # 模型上新 badge helpers (NEWS)
node tests/pick_test.mjs                          # fetch-models picker helpers (PICK)
node tests/reqstate_test.mjs                      # requests table state (REQSTATE)
node tests/requests_detail_test.mjs               # request-detail protocol/effort labels (REQUESTS-DETAIL)
```

Test ↔ source mirror: `test_app.py`→`app.py` (+ `test_app_helpers.py` for `_build_upstream_headers`/`_rewrite_body`), `test_config.py`→`config.py`, `test_convert.py`→`convert.py`; `packages/apicompat/tests/`→the SDK (`parity/` = golden suite), `test_store.py`→`store.py`, `test_admin.py`→`admin.py`, `test_menubar.py`→`menubar.py`, `test_fetch.py`+`test_watcher.py`→`fetch.py`+`watcher.py`, `test_prompt_templates.py`→`prompt_templates.py`, `test_stats_*.py`→`stats/`.

`docs/DEVELOPMENT.md` is the long-form user/developer guide (full config reference, admin API table, error mapping, parity-golden regeneration recipe). It lags the code in places — trust the code and this file first.

## Config schema (key fields)

Validated in `config.py:validate_config`. See `README.md` for full example.

| Field | Notes |
|-------|-------|
| `server.api_key` | **Required.** Sentinel, not real auth — see Gotchas. Activates endpoint auth defaults. |
| `server.host`/`port` | Defaults `127.0.0.1:8080`. |
| `server.debug_log` | `false` default; `--debug-log` CLI flag overrides. Logs real keys. |
| `server.extra_body` | Per-protocol map keyed by `Protocol` values (`openai_chat` \| `openai_responses` \| `anthropic_messages`) merged into *converted* upstream requests **after** conversion; `null` deletes the field (needed because some upstreams reject the param name itself, e.g. `stop`). No per-model `extra_body`. |
| `endpoints[].base_url` | Unique across endpoints (trailing `/` stripped). Client path appended verbatim. |
| `endpoints[].custom_headers` | Shared by all models under that `base_url`; non-auth defaults override client headers. |
| `endpoints[].models[].name` | Unique across all endpoints (exact match routing). |
| `endpoints[].models[].upstream_model` | Optional rewrite of `model` field. |
| `endpoints[].models[].system_prompt` | Optional — Anthropic `system` / Responses `instructions` / OpenAI first `role: system` (prepended if absent), keyed by `final_protocol` (`target or source`). |
| `endpoints[].models[].prompt_template` | Optional — prompt-template **name** (a folder under `prompts/`); its `prompt.md` body overrides `system_prompt` at request time. Set/cleared only via the admin inject/uninject routes. |
| `endpoints[].models[].protocol` | Optional pin — one of `PROTOCOL_VALUES` (`[e.value for e in Protocol]`). Mismatch triggers conversion. |
| `endpoints[].models[].pricing` | Optional `input`/`cache_read`/`cache_write`/`output` ($/1M tokens); missing `cache_write`/`cache_read`/`output` derived from `input` ×1.25/×0.1/×5. |
| `endpoints[].models_news` | Internal. Upstream ids flagged「模型上新」by the watcher; cleared when the fetch-models modal opens. Not round-tripped by `config_to_dict`. |

## Architecture

```
Client → app.py:proxy_request → ModelRouter.route() → header pipeline → convert/rewrite body → httpx → upstream
                                   ↑ store.py (hot reload)              ↓
                                   └─ admin.py (CRUD)            stats/capture → db → worker (cost)
```

| Module | Responsibility |
|--------|---------------|
| `config.py` | `load_config`/`validate_config`→`GatewayConfig`/`EndpointConfig` (one per model, flattened from `endpoints[].models`); `config_to_dict` groups back by `base_url`. `ModelRouter` (exact-match), `detect_format` (`/v1/messages`→`anthropic`, `/v1/responses`→`responses`, else `openai` → `Protocol` via `convert.FORMAT_TO_PROTOCOL`), `PROTOCOL_VALUES`. |
| `store.py` | `ConfigStore` — immutable `_Snapshot(config, router, raw)` + `threading.Lock`; every mutation validate → atomic write (`tmp+os.replace`, `fsync` data+dir, preserve mode) → swap. `StoreError` if no `config_path` (read-only/503). Seeds `raw` from file to preserve empty-model endpoints; preserves effective `debug_log` across first write. `update_extra_body` handles `server.extra_body`. |
| `app.py` | `build_app(config, client=None, config_path=None, stats_db_path=None, watcher_cache_path=None, watcher_interval=12h, prompts_dir=None)` — `GET /v1/models` before catch-all `/{full_path:path}` (GET/POST/PUT/PATCH/DELETE/HEAD/OPTIONS, no TRACE). `proxy_request` reads `store.router`/`store.config` per-request. Module-global `working_model` for non-body methods (`WITH_BODY_METHODS={"POST","PUT"}` only read body). Helpers: `_build_upstream_headers`, `_rewrite_body`, `_reasoning_effort`, `_sse_response`/`_buffered_response`, `_decompressor`, `_UsageCapture`. httpx timeout `connect 60s / read 300s / write 60s / pool 60s`. Wires `app.state.{stats_db,cost_worker,model_watcher,watcher_cache_path,prompt_templates}` — the prompt store is created here and shared with the admin routes (never construct a second one). |
| `admin.py` | `register_admin_routes(app, store, http_client, watcher_cache_path, model_watcher, prompt_store)` — `/admin` (`static/admin.html` zh-CN + `static/chart.umd.min.js`) + CRUD `/admin/api/endpoints`, `/admin/api/endpoints/models`, `GET /admin/api/upstream-models` (10s fetch via endpoint's `custom_headers` only, parses `data[].id`; response also carries `new_models` — **union of config `models_news` and ids absent from the cache baseline** (the watcher folds detected news into the baseline at detection time, so a pure cache diff would miss exactly the badge-lit models), and a successful fetch refreshes that cache entry so seen models aren't re-reported), `POST /admin/api/endpoints/models-news` (clears「模型上新」), `GET /admin/api/scan-models` (manually triggers one watcher scan; returns per-endpoint `{ok, models, added, removed, models_news}` or `{ok:false, error}`; 503 in read-only mode), `GET/PUT /admin/api/extra-body` (read/write `server.extra_body` via `store.update_extra_body`; keys validated against `PROTOCOL_VALUES`), `GET /admin/api/server` (host/port/`api_key` sentinel — feeds the admin UI 模型页 复制curl button; pure builder in `// CURL-START/END` markers, tested by `tests/curl_test.mjs`), and the prompt-template trio: `GET /admin/api/prompt-templates`, `POST`/`DELETE /admin/api/endpoints/models/prompt-template` (inject runs `prompt.py` hook via `asyncio.to_thread` **before** writing config, 409 if already injected, 502 + clears injection on `HookError`, then `preload` + `store.set_prompt_template`; un-inject runs no script). Query-param IDs (models may contain `/`); errors `{"error":{"message":…}}` (422 for malformed bodies is FastAPI `{"detail":…}`). |
| `prompt_templates.py` | `PromptTemplateStore(root)` — prompt-template folders under `prompts/` (`build_app`'s `prompts_dir` param, default `DEFAULT_PROMPTS_DIR` = repo-root `prompts/`; authoring contract in `prompts/README.md`). Root is `.resolve()`d (the hook subprocess runs with `cwd=<template dir>`, so a relative root would resolve the script path twice). `list_templates` rescans every call and returns **every subdirectory name** (content-agnostic — a folder with only `prompt.py`, or empty, must still be selectable), missing root → `[]`. `get_prompt(name)` caches for the process lifetime and resolves the body in this order: usable `prompt.md` → else, if `prompt.py` exists, run the hook then re-read `prompt.md` → else `None` (**md-only never spawns**). `HookError` inside `get_prompt` is swallowed into a negative cache entry so a broken hook can never 500 a proxy request. `aget_prompt` is the async twin for the request path (`asyncio.to_thread`; the hook may spend up to 30 s and must not block the event loop). `preload` drops the cache entry and reloads (used by the inject route so a hook-rewritten body wins); returns whether the body is usable. `run_load_hook(name)` runs the optional `prompt.py` in a subprocess (`python -I -B -c <runner> <script>`, cwd = template dir, stdin closed, `-I` isolation so `sys.path` has neither the script dir nor cwd, 30 s timeout) whose `load_prompt()` must not return a non-zero int / raise / exit non-zero — `0`, non-`int` and `bool` count as success, and results come back on stdout after a `__HOOK_RESULT__` marker (parsed with `rfind`, last one wins). Name validation (`_is_valid_name`) rejects blanks, `.`/`..` and separators. Construction does no I/O; body reads are lazy, so a broken template never fails boot. |
| `__main__.py` | CLI `-c/--config`, `--debug-log` (overrides `server.debug_log`), runs uvicorn. |
| `menubar.py` | macOS-only status-bar companion (bolt icon; left-click opens `/admin`, right-click menu open/restart/stop — restart SIGTERMs the gateway, detaches `gateway.sh restart` replacing gateway + menubar, then quits; exits when gateway dies). Spawned by `gateway.sh start`, removed by `stop`. Requires `menubar` extra. |
| `fetch.py` | Shared `fetch_models(base_url, custom_headers, client)` — server-side GET `{base_url}/v1/models` (10s timeout), returns deduped ids; `FetchError` on transport/non-200/non-JSON. Used by admin route + watcher. |
| `watcher.py` | `ModelWatcher` — asyncio task started in app lifespan; every 12h (`watcher_interval` param of `build_app`) fetches each endpoint's `/v1/models`, diffs against the baseline cache (`{"base_url": {"ids": [...], "ts": ...}}`), accumulates new ids into `endpoints[].models_news` via `store.set_models_news` (accumulates — badge lists everything since the user last looked). First scan builds baseline only; per-endpoint fetch failure or `StoreError` skips that endpoint without touching cache/config (they stay consistent). Read-only mode (no `config_path`) never starts the watcher. **`build_app` resolves the cache path once** (`watcher_cache_path` param or `default_cache_path()` = `/tmp/llm-gateway-models-cache.json` when `config_path` is set) and shares it between the watcher and the admin fetch-models route — never wire them separately. |
| `convert.py` + `apicompat` SDK | Conversion now lives in the workspace package `packages/apicompat/` (import `apicompat`), consumed by the gateway as a path/workspace dependency. **Wire behavior is pinned to sub2api's `apicompat`** by a 163-case golden parity suite shipped inside that package (`packages/apicompat/tests/parity/`; fixtures generated by running sub2api's Go converters — regenerate via the harness recipe in the parity test's docstring). SDK API: `convert_request`/`convert_response` (dict→dict) and `convert_stream`/`astream`/`StreamCodec` (bytes→bytes, protocol names as strings). `gateway/convert.py` only decides *when* to convert: `get_convert_stats(path, pinned_protocol)`→`ConvertStats`; `need_convert`/`get_target_path` (`_PROTOCOL_PATH`); `convert_request_body`/`convert_response_body`/`convert_sse_chunks` (delegates to `apicompat.astream`, which buffers split SSE lines, re-emits in 64-byte pieces and runs the decoder's `finalize()` when the upstream ends without a termination event). Request conversion failure → 400, response → 502. `server.extra_body` merged after conversion (wins; `null` deletes). Key semantics (all sub2api-derived): system → Responses `developer` items / Chat leading-merge+mid-downgrade / Anthropic top-level fold; thinking `signature` round-trips via `encrypted_content` (Claude 5.5 envelope) and the key is always present on thinking blocks; usage keeps all four Anthropic buckets with input-excludes-cache semantics; tool_use/tool_result pairing repaired on Anthropic and Chat targets; chat-stream terminal events deferred to `finalize()`; `max_output_tokens`/`max_completion_tokens` floored at 128 (A→R/A→C only); gpt-5+ drops sampling params; `top_k` dropped for Chat targets; Chat `refusal` routes to `message.refusal` (Chat) or a text block (Anthropic).
| `stats/` | `capture.py` (scan-only `extract_usage`/`SseUsageScanner`; Responses `input_tokens` includes `cached_tokens` → split to cache-read; Anthropic cache split across `message_start`+`message_delta`), `db.py` (SQLite `usage.db` next to config, `cost_usd IS NULL` queue, WAL, migrations for timing and `client_protocol`/`upstream_protocol`/`reasoning_effort`), `worker.py` (background thread, freezes cost via `pricing.py`), `api.py` (`/admin/api/stats/day|endpoints|model-tps` + `/admin/api/requests`; request items include client→upstream protocol flow and final post-conversion/`extra_body` effort; `stats/endpoints` supports `?date=` and `?from=&to=&granularity=minute|hour|day`; time-range mode also returns `total_input/total_cache_read/total_cache_write/total_output` + `cache_hit_rate` (via `db.token_totals`, which `/admin/api/requests` also reuses for its filter-scoped `totals` block)). Opened when `config_path` or `stats_db_path` given; else 503. Compressed upstreams: decompress copy (gzip/deflate via zlib) for capture only; `br`/`zstd` skip capture. |

### Request flow (`app.py:proxy_request`)

1. **Body** — only `POST`/`PUT` read body; others reuse `working_model` (400 if none on first request). `model = data.get("model") or working_model`.
2. **Route** — `store.router.route(model)` → `EndpointConfig` (400 bad JSON/missing `model`, 404 unknown).
3. **URL** — `endpoint.base_url` + client path (or `ConvertStats.get_target_path()` when converting) + query, verbatim.
4. **Headers** — strip hop-by-hop → overlay `custom_headers` (lowercased; non-auth override wins) → sentinel check (`Authorization` after `Bearer ` or `x-api-key` equals `server.api_key` → drop client auth, use endpoint default; no auth header → no auth upstream) → derive missing sibling (`Authorization`↔`x-api-key`) → filter `accept-encoding` to `gzip,deflate,identity` only (`br`/`zstd`/`*` stripped; `identity` if empty; when converting strip response `content-encoding`/`content-length`).
5. **Body rewrite** (if non-empty) — `convert_request_body` (if `need_convert`) → `_rewrite_body` (`upstream_model`, `system_prompt` per `final_protocol`, `extra_body[final_protocol.value]` with null-delete). If `endpoint.prompt_template` is set, its body is read via `app.state.prompt_templates.get_prompt(name)` and passed as the request-time `system_prompt` override; `None` (missing/blank/unreadable template) falls back to `endpoint.system_prompt`, and an empty string means "write nothing".
6. **Forward** via httpx (502 on `HTTPError`); SSE (`text/event-stream`) streamed with `aclose()` in `finally` (mid-stream drop emits `encode_stream_terminator` when converting), else buffered; converting transforms response back (buffered bodies handle gzip/deflate via magic-byte sniffing; streams via `convert_sse_chunks`).

### Runtime mutation (admin)

- `build_app(..., config_path=...)` enables write-back; without it writes return 503 (reads still work).
- Hot reload via `store.router`/`store.config` per request. Consistency: validate → atomic write → snapshot swap.
- First write rewrites file with 2-space indent; key order preserved from original (store seeds from file, not `config_to_dict`).
- Admin API unauthenticated and leaks `custom_headers` — keep gateway on `127.0.0.1`.
- In-flight requests hold the old `EndpointConfig`; deleting an endpoint mid-request is safe. External edits to `config.json` while running are ignored.

### Response handling

- SSE streamed chunk-by-chunk; buffered otherwise (gzip passes through except decompressed copies for conversion/capture).
- When converting, response `content-encoding`/`content-length` stripped.
- **TTFT** = time to first model-output event (content/thinking/tool delta; for Responses any `response.*.delta` plus `response.output_item.added` — latter needed for reasoning models whose encrypted thinking isn't streamed as deltas), not first byte. Non-streaming has no TTFT (unobservable).

### Gotchas

- **`server.api_key` is a sentinel, never validated.** Put real upstream keys in `endpoints[].custom_headers`.
- **Auth derivation** adds the sibling header (`Authorization: Bearer X` ↔ `x-api-key: X`) unless `custom_headers` fixed both.
- **`accept-encoding` filtering is intentional** — don't re-add `br`/`zstd`; conversion/capture can't decompress them.
- **`extra_body` null-delete** — `{"openai_responses":{"stop":null}}` strips the field some upstreams reject even as `null`/`""`.
- **Body rewrites re-serialize JSON** — field order/indentation not preserved.
- **Stats records `endpoint.model` (client-facing), cost frozen at write time.** Decode TPS = `output / (duration - TTFT)` (queueing included in TTFT); non-streaming fakes 3 s TTFT (no offset if duration ≤ 3 s).
- **`working_model` is process-global**, not per-client — concurrent non-body requests can observe each other's last model.
- **`prompt_template` 只存模板名，正文按名从 `prompts/` 读入内存并常驻。** 正文解析顺序：可用的 `prompt.md` → 否则若有 `prompt.py` 先跑钩子再读 md → 否则视为不可用（md-only 不起子进程；钩子失败/没写出 md 都只是“不可用”，请求回落手写 `system_prompt`，不会 500）。正文只在两个时机载入：点「确认注入」（`preload` 丢缓存重读，钩子改写的正文才不会被旧缓存盖住）和重启后的首次使用（冷缓存，钩子在线程里跑）。载入后改 `prompt.md` 或删目录都要重启才生效。注入接口现在会**校验正文可用**（钩子成功但没有 md → 400），并清掉注入状态；对已注入的模型再注入仍是 409。
- **Debug logging logs real keys** — trusted contexts only.
- **`admin.html` is one big inline-script page** — the JS unit tests extract helper functions by `// NAME-START` / `// NAME-END` markers (or by `function renderStatsTop()` / `renderModels(content)` headers). Keep those markers when editing; renaming or re-wrapping the block is what breaks `node tests/*.mjs`.
- `AGENTS.md` is a condensed duplicate of this file and drifts independently (it still cites `gateway/converter/` and `tests/test_convert_parity.py`, both of which moved into `packages/apicompat/`). When they disagree, this file and the code win.
