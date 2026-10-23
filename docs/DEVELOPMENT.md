# llm-gateway — developer guide

Everything that a *user* doesn't need: the full configuration reference, the
admin API, stats internals, error mapping and architecture notes.

For install / start / a first request, see [README.md](../README.md).

## Contents

- [Configuration reference](#configuration-reference)
- [Protocol conversion](#protocol-conversion)
- [Auth header handling](#auth-header-handling)
- [Admin API](#admin-api)
- [Usage stats](#usage-stats)
- [Error mapping](#error-mapping)
- [macOS menu bar](#macos-menu-bar)
- [Development](#development)
- [Architecture](#architecture)

---

## Configuration reference

The gateway reads a JSON file. `config.json` in the repo root is a template;
copy it to `config.local.json` (git-ignored) and edit.

```json
{
  "server": {
    "host": "127.0.0.1",
    "port": 8088,
    "api_key": "apikey",
    "debug_log": false
  },
  "endpoints": [
    {
      "base_url": "https://api.anthropic.com",
      "custom_headers": {
        "x-api-key": "sk-ant-...",
        "anthropic-version": "2023-06-01"
      },
      "models": [
        { "name": "claude-sonnet-5-5", "upstream_model": "claude-sonnet-5-5" }
      ]
    },
    {
      "base_url": "https://your-relay.example.com",
      "custom_headers": { "Authorization": "Bearer sk-relay-key" },
      "models": [
        {
          "name": "relay/kimi-k3",
          "upstream_model": "kimi-k3",
          "protocol": "anthropic_messages",
          "system_prompt": "You are a helpful assistant."
        }
      ]
    }
  ]
}
```

### `server`

| Field | Default | Meaning |
|---|---|---|
| `api_key` | **required** | Sentinel, not authentication. Clients that must send *some* key use this value; the gateway replaces it with the endpoint's configured key. The gateway never validates it. |
| `host` / `port` | `127.0.0.1:8080` | Bind address. Keep it on loopback — the admin API is unauthenticated. |
| `debug_log` | `false` | Log every request in full, **including real upstream keys**. Trusted contexts only. `--debug-log` overrides. |
| `extra_body` | — | Per-protocol overrides merged into *converted* requests. See below. |

### `endpoints[]`

One entry per upstream base URL; `base_url` must be unique.

| Field | Meaning |
|---|---|
| `base_url` | Upstream root. The client's path is appended verbatim (`POST /v1/messages` → `{base_url}/v1/messages`). |
| `custom_headers` | Default headers for every model under this endpoint. Auth usually lives here. Non-auth defaults override same-named client headers; a real client auth key overrides the default. |
| `socks5_proxy` | Optional `socks5://[user:pass@]host:port` URL (no path/query). Applied to proxied requests, fetch-models, and the model watcher. |
| `models` | Models this endpoint serves. May be empty. |
| `models_news` | Internal. Managed by the watcher; do not edit. |

### `endpoints[].models[]`

| Field | Meaning |
|---|---|
| `name` | Client-facing model name. Unique across all endpoints. |
| `upstream_model` | Optional. Replaces the body's `model` before forwarding. |
| `system_prompt` | Optional. Replaces the client's system prompt (OpenAI: first `role: system` message, prepended if absent; Anthropic: top-level `system`; Responses: top-level `instructions`). |
| `protocol` | Optional pin: `openai_chat` \| `openai_responses` \| `anthropic_messages`. Setting a value that differs from the client's format triggers conversion. |
| `pricing` | Optional, $ per 1M tokens: `input`, `cache_read`, `cache_write`, `output`. Missing `cache_write`/`cache_read`/`output` derive from `input` (×1.25 / ×0.1 / ×5). No pricing → cost 0. |

### `server.extra_body`

Merged into converted upstream requests **after** conversion. Values win over
converted defaults; `null` **deletes** the field — some upstreams reject the
parameter name itself, even when null.

```json
{
  "server": {
    "api_key": "apikey",
    "extra_body": {
      "openai_responses": { "store": false, "include": ["reasoning.encrypted_content"] },
      "openai_chat": { "stop": null }
    }
  }
}
```

Keys must be one of the three protocol names. There is no per-model
`extra_body`.

---

## Protocol conversion

Each model may pin a protocol. The gateway compares it with the format implied
by the request path:

| Client path | Detected format |
|---|---|
| `/v1/messages` | `anthropic_messages` |
| `/v1/responses` | `openai_responses` |
| anything else | `openai_chat` |

- **No pin, or a match** → forwarded as-is.
- **Mismatch** → request body converted to the pinned protocol, sent to that
  protocol's path, and the response (streaming or not) converted back.

Conversion covers thinking blocks and signatures, tool call/result pairing,
cache token accounting, refusals, and stream terminator synthesis. Wire
behaviour is pinned to sub2api's `apicompat` by a 163-case golden parity suite
that ships with the SDK.

**SSE handling.** Chunks are buffered across arbitrary byte boundaries,
converted event by event, and re-emitted in 64-byte pieces so streaming starts
promptly. If the upstream ends without a termination event, the decoder's
`finalize()` pass synthesizes one (`message_stop` / `response.completed` /
`[DONE]`) so the client's parser closes cleanly. A dropped upstream connection
mid-stream emits the target protocol's terminator instead of leaving a dangling
block.

**Compression.** The forwarded `accept-encoding` is narrowed to
`gzip,deflate,identity`; `br`/`zstd` are stripped so the gateway never receives
an encoding it cannot decode for conversion or usage capture. The bytes sent to
the client are untouched (gzip responses are decompressed only into a copy used
for stats).

---

## Auth header handling

1. Hop-by-hop headers are removed.
2. `custom_headers` are overlaid — non-auth defaults beat same-named client
   headers.
3. If the client's auth equals `server.api_key` (the sentinel), the client's
   header is dropped and the endpoint's configured default is used. Otherwise a
   real client key wins.
4. No client auth at all → nothing is sent upstream (the endpoint default is not
   a fallback).
5. If exactly one of `Authorization` / `x-api-key` is non-empty, the other is
   derived.

---

## Admin API

All routes are unauthenticated and some return `custom_headers` (your upstream
keys). Keep the gateway on `127.0.0.1`; use an SSH tunnel for remote access.

| Method | Path | Result |
|---|---|---|
| `GET` | `/admin/api/endpoints` | Endpoints with models |
| `POST` | `/admin/api/endpoints` | Create endpoint (201) |
| `PUT` | `/admin/api/endpoints?base_url=` | Update endpoint headers |
| `DELETE` | `/admin/api/endpoints?base_url=` | Delete endpoint + models (204) |
| `POST` | `/admin/api/endpoints/models?base_url=` | Add model (201) |
| `PUT` | `/admin/api/endpoints/models?base_url=&name=` | Update model (omit `pricing` to keep it) |
| `DELETE` | `/admin/api/endpoints/models?base_url=&name=` | Delete model (204) |
| `GET` | `/admin/api/upstream-models?base_url=` | Fetch the upstream's model list |
| `POST` | `/admin/api/endpoints/models-news` | Clear the「模型上新」badge |
| `GET` | `/admin/api/scan-models` | Run one watcher scan now |
| `GET`/`PUT` | `/admin/api/extra-body` | Read / write `server.extra_body` |
| `GET` | `/admin/api/server` | Host/port/sentinel (feeds the copy-curl button) |

Identifiers travel as query parameters (URL-encoded) because model names may
contain `/`.

```bash
# Add an endpoint
curl -X POST http://127.0.0.1:8088/admin/api/endpoints \
  -H 'Content-Type: application/json' \
  -d '{"base_url": "https://api.example.com", "custom_headers": {"User-Agent": "my-agent/1.0"}}'

# Add a model under it
curl -X POST "http://127.0.0.1:8088/admin/api/endpoints/models?base_url=https%3A%2F%2Fapi.example.com" \
  -H 'Content-Type: application/json' \
  -d '{"name": "my-model"}'
```

Every write is validated, written atomically (tmp + `os.replace`, `fsync`), and
hot-reloaded — no restart. In read-only mode (no config path) writes return 503.

### Admin UI

`http://127.0.0.1:8088/admin` (Chinese UI):

- **端点 / 模型** — CRUD; 「复制curl」 produces a ready-to-run command using the
  model's pinned protocol.
- **获取模型** — fetch the upstream's model list, add with one click. Endpoints
  that gained models show a red「模型上新」badge, cleared when the dialog opens.
- **统计** — token/cost charts, TPS, per-request detail.
- **extra_body** — edit the global per-protocol overrides.

### Model watcher

Every 12 hours the gateway fetches each endpoint's `/v1/models`, diffs against
a baseline cache (`/tmp/llm-gateway-models-cache.json`), and accumulates new ids
into `endpoints[].models_news`. The first scan only builds the baseline.
Per-endpoint fetch failures are skipped without touching the cache or config.
Read-only mode never starts the watcher.

---

## Usage stats

Every proxied request records input / cache-read / cache-write / output tokens
into `usage.db` next to the config. A background worker computes and **freezes**
the cost from the model's `pricing` — later price changes don't rewrite history.
Capture failures never break proxying.

Recording is format-agnostic: all three protocols are handled, including
streaming (Responses usage arrives in the final `response.completed` event, and
its `input_tokens_details.cached_tokens` is split into cache-read).
Each request row also records `client_protocol` → `upstream_protocol` and the
final reasoning effort present after conversion and `extra_body` overrides
(`reasoning_effort`, `reasoning.effort`, `output_config.effort`, or Anthropic
`thinking` budget/disabled/adaptive). Historical rows without this metadata
return `null` and render as `—`.

Formulas:

- Cache hit rate = cache read ÷ (input + cache read + cache write)
- Decode TPS = output ÷ (duration − TTFT)
- Non-streaming requests have no observable TTFT, so 3 s is assumed when
  estimating TPS (no offset when duration ≤ 3 s).

Read-only APIs (times use the server's local timezone; `to` is inclusive of its
minute):

- `GET /admin/api/stats/day?date=YYYY-MM-DD`
- `GET /admin/api/stats/endpoints?date=…` or `?from=&to=&granularity=minute|hour|day`
- `GET /admin/api/stats/model-tps?date=YYYY-MM-DD`
- `GET /admin/api/requests?from=…&to=…&base_url=…&model=…&limit=50&offset=0`

Stats are disabled when the app is built without a config path; those endpoints
then return 503.

---

## Error mapping

| Condition | Status |
|---|---|
| Missing `model` (or a body-less request with no prior model) | 400 |
| Model not configured | 404 |
| Request cannot be converted | 400 |
| Upstream unreachable | 502 |
| Upstream response cannot be converted | 502 |
| Upstream error | passed through unchanged |
| Admin validation error | 400 / 422 |
| Admin not found | 404 |
| Admin write with no config path (read-only mode) | 503 |
| Admin write failure (disk / permissions) | 503 |

---

## macOS menu bar

With the `menubar` extra installed, `./gateway.sh start` also launches a
status-bar bolt icon: left-click opens the admin page, right-click offers open /
restart / stop. The icon disappears when the gateway stops. `./gateway.sh stop`
removes it.

```bash
uv sync --extra menubar
uv run python -m gateway.menubar -c config.local.json   # standalone
```

---

## Development

```bash
uv run pytest -v                          # gateway + SDK
uv run pytest packages/apicompat/tests    # SDK only (parity + stream)
node tests/*.mjs                          # admin UI helper tests
```

Layout:

```
gateway/                 FastAPI app, routing, admin UI, stats
packages/apicompat/      standalone conversion SDK (uv workspace member)
tests/                   gateway tests + admin UI JS tests
docs/                    this guide
config.json              template config (git-ignored copy: config.local.json)
```

Test ↔ source mirror: `test_app.py`→`app.py`, `test_config.py`→`config.py`,
`test_convert.py`→`convert.py`, `test_store.py`→`store.py`,
`test_admin.py`→`admin.py`, `test_menubar.py`→`menubar.py`,
`test_fetch.py`+`test_watcher.py`→`fetch.py`+`watcher.py`,
`test_stats_*.py`→`stats/`, `packages/apicompat/tests/`→the SDK.

The SDK is a uv workspace member:

```bash
uv run --package apicompat pytest packages/apicompat/tests
uv build --package apicompat              # wheel + sdist in dist/
```

### Regenerating the parity goldens

```bash
cd ~/workspace/sub2api/backend
GOLDEN_DIR=/tmp/sub2api-golden \
GOLDEN_FIXTURES=<this-repo>/packages/apicompat/tests/parity/golden/fixtures.json \
go test ./internal/pkg/apicompat/ -run TestZZGoldenDump
cp /tmp/sub2api-golden/*.json <this-repo>/packages/apicompat/tests/parity/golden/
```

---

## Architecture

```
Client ─▶ app.py:proxy_request ─▶ ModelRouter.route() ─▶ header pipeline
                                       │                      │
                                  store.py (hot reload)   convert/rewrite body
                                       │                      │
                                  admin.py (CRUD)         httpx ─▶ upstream
                                                              │
                                                    stats/capture ─▶ db ─▶ worker
```

| Module | Responsibility |
|---|---|
| `config.py` | `load_config` / `validate_config` → `GatewayConfig` / `EndpointConfig` (flattened, one per model); `config_to_dict` groups back by `base_url`. `ModelRouter` (exact match), `detect_format`, `PROTOCOL_VALUES`. |
| `store.py` | `ConfigStore` — immutable `_Snapshot(config, router, raw)` + `threading.Lock`. Every mutation: validate → atomic write → swap. Seeds `raw` from the file so empty-model endpoints survive a write-back. |
| `app.py` | `build_app(...)`. `GET /v1/models` before the catch-all `/{full_path:path}`. `proxy_request` reads `store.router` / `store.config` per request. Lazily caches one httpx client per endpoint `socks5_proxy` and shares the selector with fetch-models/watcher. Helpers: `_build_upstream_headers`, `_rewrite_body`, `_reasoning_effort`, `_sse_response`, `_buffered_response`, `_decompressor`, `_UsageCapture`. httpx timeout: connect 60 s / read 300 s / write 60 s / pool 60 s. |
| `convert.py` | Decides *when* to convert: `get_convert_stats`, `convert_request_body`, `convert_response_body`, `convert_sse_chunks` (delegates to `apicompat.astream`). |
| `admin.py` | Admin page + CRUD API, fetch-models, model-news, scan, extra-body. |
| `watcher.py` | `ModelWatcher` asyncio task started in the app lifespan. |
| `fetch.py` | Shared `fetch_models(base_url, custom_headers, client)`. |
| `stats/` | `capture.py` (scan-only usage extraction), `db.py` (SQLite, `cost_usd IS NULL` queue), `worker.py` (cost freezing), `api.py` (stats + requests endpoints), `pricing.py`. |
| `menubar.py` | macOS-only status-bar companion. |
| `__main__.py` | CLI `-c/--config`, `--debug-log`. |
| `packages/apicompat/` | The IR-hub conversion SDK — see its own README. |

### Request flow (`app.py:proxy_request`)

1. **Body** — only `POST`/`PUT` read the body; others reuse the process-global
   `working_model` (400 if none yet).
2. **Route** — `store.router.route(model)` (400 bad JSON/x missing model, 404
   unknown).
3. **URL** — `endpoint.base_url` + client path (or the pinned protocol's path
   when converting) + query, verbatim.
4. **Headers** — see [Auth header handling](#auth-header-handling).
5. **Body rewrite** — `convert_request_body` (if converting) then
   `_rewrite_body` (`upstream_model`, `system_prompt`, `extra_body` with
   null-delete).
6. **Forward** with the endpoint's direct or SOCKS5 httpx client (502 on
   `HTTPError`). SSE is streamed with
   `aclose()` in `finally`; otherwise buffered. Converting transforms the
   response back.

### Gotchas

- **`server.api_key` is a sentinel, never validated.** Real upstream keys belong
  in `endpoints[].custom_headers`.
- **`accept-encoding` filtering is intentional** — `br`/`zstd` are stripped
  because conversion and capture cannot decode them.
- **`extra_body` null-delete** strips fields some upstreams reject even as
  `null`/`""`.
- **Body rewrites re-serialize JSON** — field order and indentation are not
  preserved.
- **Stats record the client-facing model name**, and cost is frozen at write
  time.
- **`working_model` is process-global**, not per-client — concurrent body-less
  requests can observe each other's last model.
- **Debug logging logs real keys** — trusted contexts only.
- **External edits to the config file while running are ignored** (the store
  holds an in-memory snapshot; use the admin UI or restart).
