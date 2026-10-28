from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit
from apicompat import Protocol


class ConfigError(Exception):
    pass


class ModelNotFoundError(Exception):
    pass


ANTHROPIC_PATH = "/v1/messages"
RESPONSES_PATH = "/v1/responses"
OPENAI_CHAT_PATH = "/v1/chat/completions"

#: The only request paths the gateway proxies, as strict suffixes. Everything
#: else (Anthropic/OpenAI Files, ``count_tokens``, batch, …) is rejected with
#: 501: the gateway speaks exactly three inference protocols plus GET
#: ``/v1/models``, and silently treating unknown paths as OpenAI — the old
#: fallback — forwarded them to ``/v1/responses`` with a rewritten body.
_PROTOCOL_SUFFIXES: tuple[tuple[str, str], ...] = (
    (ANTHROPIC_PATH, "anthropic"),
    (RESPONSES_PATH, "responses"),
    (OPENAI_CHAT_PATH, "openai"),
)

#: Protocol identifiers a model can be pinned to. The request path's detected
#: format is compared against the model's pinned protocol; a mismatch triggers
#: body conversion (see gateway.converters / app.py). Values mirror the
#: converter's Protocol enum (openai_chat <-> openai, anthropic_messages <->
#: anthropic).
PROTOCOL_VALUES: list[str] = [e.value for e in Protocol]


def detect_format(path: str) -> str | None:
    """Classify a request path as anthropic / responses / openai, or None.

    Match the protocol suffix rather than requiring an exact path: gateways
    are often mounted behind a prefix (e.g. ``/anthropic/v1/messages``), and
    clients may include a trailing slash.

    Returns None for any path that is not one of the three supported protocol
    endpoints. Callers must treat None as "not proxied" (501) — there is
    deliberately no fallback, so lookalike paths (``/v1/messages-extra``,
    ``/v1/messages/count_tokens``, ``/v1/files``) are never routed as OpenAI.
    """
    normalized = path.rstrip("/") or "/"
    for suffix, fmt in _PROTOCOL_SUFFIXES:
        if normalized.endswith(suffix):
            return fmt
    return None


class ModelRouter:
    """Exact-match model → endpoint lookup, derived from a GatewayConfig."""

    def __init__(self, config: GatewayConfig) -> None:
        self._by_model = {e.model: e for e in config.endpoints}

    def route(self, model: str) -> EndpointConfig:
        endpoint = self._by_model.get(model)
        if endpoint is None:
            raise ModelNotFoundError(model)
        return endpoint

    def list_models(self) -> list[str]:
        return sorted(self._by_model)


@dataclass
class EndpointConfig:
    model: str
    base_url: str
    custom_headers: dict[str, str] = field(default_factory=dict)
    socks5_proxy: str | None = None
    upstream_model: str | None = None
    system_prompt: str | None = None
    prompt_template: str | None = None  # template NAME (not body); resolved per request by gateway.prompt_templates
    pricing: dict[str, float] | None = None
    protocol: str | None = None  # pinned protocol: "anthropic_messages" | "openai_chat" | "openai_responses"
    models_news: list[str] = field(default_factory=list)  # upstream ids flagged 模型上新; written by the watcher only


@dataclass
class GatewayConfig:
    host: str
    port: int
    api_key: str
    endpoints: list[EndpointConfig]
    debug_log: bool = False
    extra_body: dict[str, dict[str, object]] | None = None  # per-protocol extra fields merged into converted upstream requests: {"anthropic_messages": {...}, "openai_chat": {...}, "openai_responses": {...}}


def normalize_base_url(url: str) -> str:
    return url.rstrip("/")


def _validate_socks5_proxy(raw: object) -> str | None:
    if raw is None:
        return None
    if not isinstance(raw, str):
        raise ConfigError("endpoint 'socks5_proxy' must be a string")
    proxy = raw.strip()
    if not proxy:
        return None
    try:
        parsed = urlsplit(proxy)
        port = parsed.port
    except ValueError as exc:
        raise ConfigError(f"invalid endpoint 'socks5_proxy': {exc}") from exc
    if parsed.scheme != "socks5":
        raise ConfigError("endpoint 'socks5_proxy' must use the socks5:// scheme")
    if not parsed.hostname:
        raise ConfigError("endpoint 'socks5_proxy' must include a host")
    if port is None or not 1 <= port <= 65535:
        raise ConfigError("endpoint 'socks5_proxy' must include a valid port")
    if parsed.path or parsed.query or parsed.fragment:
        raise ConfigError("endpoint 'socks5_proxy' must not include a path, query, or fragment")
    return proxy


def _validate_pricing(raw: object) -> dict[str, float] | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ConfigError("model 'pricing' must be an object")
    result: dict[str, float] = {}
    for key, value in raw.items():
        k = str(key)
        if k not in ("input", "cache_read", "cache_write", "output"):
            raise ConfigError(f"unknown pricing field: {k}")
        try:
            num = float(value)
        except (TypeError, ValueError):
            raise ConfigError(f"pricing.{k} must be a number")
        if not math.isfinite(num):
            raise ConfigError(f"pricing.{k} must be a finite number")
        if num < 0:
            raise ConfigError(f"pricing.{k} must be non-negative")
        result[k] = num
    return result


def validate_config(raw: dict) -> GatewayConfig:
    server = raw.get("server") or {}
    if not isinstance(server, dict):
        raise ConfigError("server must be an object")
    api_key = server.get("api_key")
    if not api_key:
        raise ConfigError("server.api_key is required")
    debug_log = server.get("debug_log", False)
    if not isinstance(debug_log, bool):
        raise ConfigError("server.debug_log must be a boolean")
    # extra_body: {"anthropic_messages": {...}, "openai_chat": {...}, "openai_responses": {...}}
    # — per-protocol extra fields merged into converted upstream requests.
    extra_body = server.get("extra_body")
    if extra_body is not None and not isinstance(extra_body, dict):
        raise ConfigError("server.extra_body must be an object")
    if extra_body is not None:
        for key, value in extra_body.items():
            if key not in PROTOCOL_VALUES:
                raise ConfigError(
                    f"unknown extra_body protocol: {key} (expected one of {', '.join(PROTOCOL_VALUES)})"
                )
            if not isinstance(value, dict):
                raise ConfigError(f"server.extra_body.{key} must be an object")
    endpoints_raw = raw.get("endpoints", [])
    if not isinstance(endpoints_raw, list):
        raise ConfigError("endpoints must be a list")
    endpoints: list[EndpointConfig] = []
    seen_base_urls: set[str] = set()
    for item in endpoints_raw:
        if not isinstance(item, dict):
            raise ConfigError("each endpoint must be an object")
        base_url = normalize_base_url(str(item.get("base_url") or ""))
        models = item.get("models")
        if not base_url or not isinstance(models, list):
            raise ConfigError("each endpoint requires 'base_url' and a 'models' list")
        if base_url in seen_base_urls:
            raise ConfigError("endpoint base_url values must be unique")
        seen_base_urls.add(base_url)
        custom_headers = item.get("custom_headers")
        if custom_headers is not None and not isinstance(custom_headers, dict):
            raise ConfigError("custom_headers must be an object")
        headers = {str(k): str(v) for k, v in (custom_headers or {}).items()}
        socks5_proxy = _validate_socks5_proxy(item.get("socks5_proxy"))
        models_news_raw = item.get("models_news")
        if models_news_raw is None:
            models_news: list[str] = []
        else:
            if not isinstance(models_news_raw, list) or not all(
                isinstance(v, str) and v for v in models_news_raw
            ):
                raise ConfigError("endpoint 'models_news' must be a list of non-empty strings")
            if len(set(models_news_raw)) != len(models_news_raw):
                raise ConfigError("endpoint 'models_news' must not contain duplicates")
            models_news = list(models_news_raw)
        for model_entry in models:
            if not isinstance(model_entry, dict):
                raise ConfigError("each model entry must be an object")
            name = model_entry.get("name")
            if not name:
                raise ConfigError("each model entry requires a 'name'")
            pricing = _validate_pricing(model_entry.get("pricing"))
            protocol = model_entry.get("protocol")
            if protocol is not None and protocol not in PROTOCOL_VALUES:
                raise ConfigError(
                    f"unknown protocol: {protocol} (expected one of {', '.join(PROTOCOL_VALUES)})"
                )
            prompt_template = model_entry.get("prompt_template")
            if prompt_template is not None:
                if not isinstance(prompt_template, str) or not prompt_template.strip():
                    raise ConfigError("model 'prompt_template' must be a non-empty string")
                # Normalize before the dot/separator checks: " .. " must be rejected
                # as "..", not stored as a traversing-but-useless name.
                prompt_template = prompt_template.strip()
                if prompt_template in (".", "..") or "/" in prompt_template or "\\" in prompt_template:
                    raise ConfigError(f"invalid prompt_template name: {prompt_template}")
            endpoints.append(EndpointConfig(
                model=str(name),
                base_url=base_url,
                custom_headers=dict(headers),
                socks5_proxy=socks5_proxy,
                upstream_model=str(model_entry["upstream_model"]) if model_entry.get("upstream_model") else None,
                system_prompt=str(model_entry["system_prompt"]) if model_entry.get("system_prompt") else None,
                prompt_template=prompt_template,
                pricing=pricing,
                protocol=str(protocol) if protocol else None,
                models_news=models_news,
            ))
    if len({e.model for e in endpoints}) != len(endpoints):
        raise ConfigError("endpoint model names must be unique")
    return GatewayConfig(
        host=str(server.get("host", "127.0.0.1")),
        port=int(server.get("port", 8080)),
        api_key=str(api_key),
        endpoints=endpoints,
        debug_log=debug_log,
        extra_body=dict(extra_body) if extra_body else None,
    )


def config_to_dict(config: GatewayConfig) -> dict:
    # Invariant: base_urls are unique per endpoint entry (enforced by
    # validate_config), so the first endpoint's headers win when grouping.
    by_base: dict[str, dict] = {}
    for ep in config.endpoints:
        entry = by_base.setdefault(
            ep.base_url,
            {
                "base_url": ep.base_url,
                "custom_headers": dict(ep.custom_headers),
                **({"socks5_proxy": ep.socks5_proxy} if ep.socks5_proxy else {}),
                "models": [],
            },
        )
        # validate_config copies the endpoint-level models_news onto every
        # model under the base_url, so the first non-empty list is the
        # endpoint's (keep the key absent when cleared).
        if ep.models_news and "models_news" not in entry:
            entry["models_news"] = list(ep.models_news)
        model_entry: dict[str, object] = {"name": ep.model}
        if ep.upstream_model is not None:
            model_entry["upstream_model"] = ep.upstream_model
        if ep.system_prompt is not None:
            model_entry["system_prompt"] = ep.system_prompt
        if ep.pricing:
            model_entry["pricing"] = dict(ep.pricing)
        if ep.protocol:
            model_entry["protocol"] = ep.protocol
        if ep.prompt_template:
            model_entry["prompt_template"] = ep.prompt_template
        entry["models"].append(model_entry)
    return {
        "server": {
            "host": config.host,
            "port": config.port,
            "api_key": config.api_key,
            "debug_log": config.debug_log,
            **({"extra_body": dict(config.extra_body)} if config.extra_body else {}),
        },
        "endpoints": list(by_base.values()),
    }


def load_config(path: str | Path) -> GatewayConfig:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ConfigError("config must be a JSON object")
    return validate_config(raw)
