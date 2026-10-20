# llm-gateway

**English** | [中文](README.zh.md)

A local LLM gateway. Point any OpenAI-, Anthropic- or Codex-compatible client
at it and it forwards each request by model name to the configured upstream.
It supports model-name replacement, system-prompt injection, and **automatic
protocol conversion** when the client and upstream use different protocols.

<img src="docs/images/architecture.svg" alt="llm-gateway converts between client and upstream protocols" width="760">

## Features

- **Protocol conversion** — OpenAI Chat, OpenAI Responses and Anthropic
  Messages are mutually compatible across all 12 directions. A model can be
  pinned to any protocol; clients using a different one still work.
- **Model routing** — exact model-name matching to an upstream, with per-model
  name mapping and system-prompt overrides.
- **Web admin** — manage endpoints and models in the browser; changes apply
  immediately without a restart.
- **Usage stats** — tokens, cost, TTFT and decode throughput per request,
  stored in a local SQLite database.
- **Reusable SDK** — the conversion core ships as a standalone package,
  [`apicompat`](#apicompat-python-sdk), usable in other projects.

## Install

Requires Python 3.11 and [uv](https://docs.astral.sh/uv/).

```bash
git clone <repo> && cd llm-gateway
uv sync
```

### macOS menu bar

On macOS the gateway can live in the menu bar, so you can check its state and
control it without a terminal.

Install the optional extra first:

```bash
uv sync --reinstall --all-extras --python=3.11
```

Then start with `./gateway.sh start` and a ⚡ icon appears in the menu bar:

- **Left-click**: opens the admin page
- **Right-click**: menu with open admin page, restart service, or stop service
- The icon disappears when the gateway exits; `./gateway.sh stop` removes it too

![Menu bar](docs/images/menubar.png)

## Start

Create a config file — one upstream endpoint is the minimum:

```json
{
  "server": { "api_key": "apikey" },
  "endpoints": [
    {
      "base_url": "https://api.anthropic.com",
      "custom_headers": { "x-api-key": "sk-ant-..." },
      "models": [{ "name": "claude-sonnet-5-5" }]
    }
  ]
}
```

Save it as `config.local.json`, then start:

```bash
./gateway.sh start
```

The gateway listens on `http://127.0.0.1:8088`; the admin UI is at `/admin`.


Other ways to run it:

```bash
uv run python -m gateway -c config.local.json      # foreground, logs to the terminal
./gateway.sh start my.json --debug                 # custom config with full request logging
./gateway.sh status | stop | restart
```

## Usage

Send a request using a configured model name:

```bash
curl http://127.0.0.1:8088/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer apikey" \
  -d '{"model": "claude-sonnet-5-5", "messages": [{"role": "user", "content": "hi"}]}'
```

`apikey` is the sentinel configured in `server.api_key`; the gateway replaces
it with the upstream key configured on the endpoint.

### Claude Code

```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:8088
export ANTHROPIC_AUTH_TOKEN=apikey
claude --model claude-sonnet-5-5
```

Any OpenAI- or Anthropic-compatible client works the same way: base URL
`http://127.0.0.1:8088`, API key `apikey`, model = any configured `name`.

## Admin UI

Open `http://127.0.0.1:8088/admin` to manage endpoints and models and to view
usage stats. Every change is validated, atomically written back to the config
file and hot-reloaded — no restart.

| Endpoint management | Usage overview |
|---|---|
| <img src="docs/images/start.png" width="420"> | <img src="docs/images/start1.png" width="420"> |
| Per-endpoint statistics | Request details |
| <img src="docs/images/start2.png" width="420"> | <img src="docs/images/start3.png" width="420"> |


## apicompat (Python SDK)

The conversion core is a standalone package you can install into any Python
project:

```bash
pip install apicompat
```

```python
from apicompat import convert_request, convert_response, convert_stream

# Requests: dict in, dict out
upstream_body = convert_request("anthropic_messages", "openai_responses", body)

# Non-streaming responses
client_body = convert_response("openai_responses", "anthropic_messages", upstream)

# SSE streams: bytes in, bytes out
for chunk in convert_stream("openai_responses", "anthropic_messages", upstream_chunks):
    write(chunk)
```

Protocol names are strings: `"openai_chat"`, `"openai_responses"`,
`"anthropic_messages"`. Use `astream` for asyncio. Full API in
[`packages/apicompat/README.md`](packages/apicompat/README.md); runnable
examples in `packages/apicompat/examples/demo.py`.

## Documentation

| | |
|---|---|
| [README.zh.md](README.zh.md) | 中文 |
| [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) | Full config reference, admin API, stats internals, architecture (for developers) |
| [packages/apicompat/README.md](packages/apicompat/README.md) | SDK API reference |

> **Security**: keep the gateway bound to `127.0.0.1`. The admin API has no
> authentication and returns upstream keys to the caller.
