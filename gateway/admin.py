from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable
from pathlib import Path

import httpx
from fastapi import FastAPI, Response
from fastapi.responses import FileResponse

from gateway.config import PROTOCOL_VALUES, ConfigError
from gateway.fetch import FETCH_MODELS_TIMEOUT, FetchError, fetch_models
from gateway.prompt_templates import DEFAULT_PROMPTS_DIR, HookError, PromptTemplateStore
from gateway.store import ConfigStore, NO_PATH_MESSAGE, NotFoundError, StoreError, _UNSET
from gateway.watcher import ModelWatcher, load_cache, save_cache

ADMIN_HTML = Path(__file__).resolve().parent / "static" / "admin.html"
STATIC_DIR = Path(__file__).resolve().parent / "static"
CHART_JS = STATIC_DIR / "chart.umd.min.js"


def _json_response(data: object, status: int = 200) -> Response:
    return Response(
        content=json.dumps(data),
        status_code=status,
        media_type="application/json",
        headers={"Cache-Control": "no-store"},
    )


def _error(status: int, message: str) -> Response:
    return _json_response({"error": {"message": message}}, status=status)


def _endpoint_payload(store: ConfigStore, base_url: str) -> dict:
    for entry in store.endpoint_dicts(base_url=base_url):
        return entry
    raise NotFoundError(f"endpoint not found: {base_url}")


def _model_payload(store: ConfigStore, base_url: str, name: str) -> dict:
    for entry in store.endpoint_dicts(base_url=base_url):
        for model_entry in entry["models"]:
            if model_entry["name"] == name:
                return model_entry
    raise NotFoundError(f"model not found: {name}")


def _write_guard(store: ConfigStore) -> Response | None:
    return None if store.path is not None else _error(503, NO_PATH_MESSAGE)


def _parse_pricing(payload: dict) -> dict[str, float] | None:
    pricing = payload.get("pricing")
    if pricing is None:
        return None
    if not isinstance(pricing, dict):
        raise ValueError("pricing must be an object")
    result: dict[str, float] = {}
    for key, value in pricing.items():
        try:
            result[str(key)] = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"pricing.{key} must be a number")
    return result


def register_admin_routes(app: FastAPI, store: ConfigStore,
                          http_client: httpx.AsyncClient | None = None,
                          http_client_for: Callable[[str | None], httpx.AsyncClient] | None = None,
                          watcher_cache_path: str | Path | None = None,
                          model_watcher: ModelWatcher | None = None,
                          prompt_store: PromptTemplateStore | None = None) -> None:
    if prompt_store is None:
        prompt_store = PromptTemplateStore(DEFAULT_PROMPTS_DIR)

    @app.api_route("/admin", methods=["GET"])
    async def admin_page() -> FileResponse:
        return FileResponse(ADMIN_HTML, headers={"Cache-Control": "no-store"})

    @app.api_route("/static/chart.umd.min.js", methods=["GET"])
    async def chart_js() -> FileResponse:
        return FileResponse(CHART_JS, media_type="application/javascript")

    @app.api_route("/admin/api/server", methods=["GET"])
    async def server_info() -> Response:
        # api_key is the sentinel, not a secret upstream key — the admin UI
        # needs it to generate curl commands that activate the endpoint's
        # configured auth headers (see app.py:_build_upstream_headers).
        cfg = store.config
        return _json_response({"host": cfg.host, "port": cfg.port, "api_key": cfg.api_key})

    @app.api_route("/admin/api/endpoints", methods=["GET"])
    async def list_endpoints() -> Response:
        return _json_response(store.endpoint_dicts())

    @app.api_route("/admin/api/endpoints", methods=["POST"])
    async def add_endpoint(payload: dict) -> Response:
        guard = _write_guard(store)
        if guard is not None:
            return guard
        base_url = payload.get("base_url")
        if not base_url:
            return _error(400, "base_url is required")
        headers = payload.get("custom_headers") or {}
        if not isinstance(headers, dict):
            return _error(400, "custom_headers must be an object")
        socks5_proxy = payload.get("socks5_proxy") or None
        try:
            store.add_endpoint(
                str(base_url),
                {str(k): str(v) for k, v in headers.items()},
                socks5_proxy=socks5_proxy,
            )
        except ConfigError as exc:
            return _error(400, str(exc))
        except StoreError as exc:
            return _error(503, str(exc))
        return _json_response(_endpoint_payload(store, str(base_url)), status=201)

    @app.api_route("/admin/api/endpoints", methods=["PUT"])
    async def update_endpoint(payload: dict, base_url: str | None = None) -> Response:
        guard = _write_guard(store)
        if guard is not None:
            return guard
        if not base_url:
            return _error(400, "base_url query parameter is required")
        if "custom_headers" not in payload:
            return _error(400, "custom_headers is required")
        headers = payload["custom_headers"]
        if not isinstance(headers, dict):
            return _error(400, "custom_headers must be an object")
        socks5_proxy = payload["socks5_proxy"] if "socks5_proxy" in payload else _UNSET
        try:
            store.update_endpoint(
                base_url,
                {str(k): str(v) for k, v in headers.items()},
                socks5_proxy=socks5_proxy,
            )
        except NotFoundError as exc:
            return _error(404, str(exc))
        except ConfigError as exc:
            return _error(400, str(exc))
        except StoreError as exc:
            return _error(503, str(exc))
        return _json_response(_endpoint_payload(store, base_url))

    @app.api_route("/admin/api/endpoints", methods=["DELETE"])
    async def delete_endpoint(base_url: str | None = None) -> Response:
        guard = _write_guard(store)
        if guard is not None:
            return guard
        if not base_url:
            return _error(400, "base_url query parameter is required")
        try:
            store.delete_endpoint(base_url)
        except NotFoundError as exc:
            return _error(404, str(exc))
        except StoreError as exc:
            return _error(503, str(exc))
        return Response(status_code=204)

    @app.api_route("/admin/api/endpoints/models", methods=["POST"])
    async def add_model(payload: dict, base_url: str | None = None) -> Response:
        guard = _write_guard(store)
        if guard is not None:
            return guard
        if not base_url:
            return _error(400, "base_url query parameter is required")
        name = payload.get("name")
        if not name:
            return _error(400, "name is required")
        try:
            pricing = _parse_pricing(payload)
        except ValueError as exc:
            return _error(400, str(exc))
        upstream_model = payload.get("upstream_model") or None
        system_prompt = payload.get("system_prompt") or None
        protocol = payload.get("protocol") or None
        try:
            ep = store.add_model(base_url, str(name), upstream_model, system_prompt, pricing, protocol)
        except NotFoundError as exc:
            return _error(404, str(exc))
        except ConfigError as exc:
            return _error(400, str(exc))
        except StoreError as exc:
            return _error(503, str(exc))
        return _json_response(_model_payload(store, ep.base_url, ep.model), status=201)

    @app.api_route("/admin/api/endpoints/models", methods=["PUT"])
    async def update_model(payload: dict, base_url: str | None = None, name: str | None = None) -> Response:
        guard = _write_guard(store)
        if guard is not None:
            return guard
        if not base_url or not name:
            return _error(400, "base_url and name query parameters are required")
        try:
            pricing = _parse_pricing(payload) if "pricing" in payload else _UNSET
        except ValueError as exc:
            return _error(400, str(exc))
        upstream_model = payload.get("upstream_model") or None
        # Absent key must mean "leave it alone" (the read-only edit dialog omits
        # system_prompt so the handwritten text survives); an explicit empty
        # value still clears it — mirroring the protocol handling below.
        system_prompt = (
            (payload.get("system_prompt") or None)
            if "system_prompt" in payload else _UNSET
        )
        protocol = payload.get("protocol") if "protocol" in payload else _UNSET
        try:
            ep = store.update_model(base_url, name, upstream_model, system_prompt, pricing, protocol)
        except NotFoundError as exc:
            return _error(404, str(exc))
        except ConfigError as exc:
            return _error(400, str(exc))
        except StoreError as exc:
            return _error(503, str(exc))
        return _json_response(_model_payload(store, ep.base_url, ep.model))

    @app.api_route("/admin/api/endpoints/models", methods=["DELETE"])
    async def delete_model(base_url: str | None = None, name: str | None = None) -> Response:
        guard = _write_guard(store)
        if guard is not None:
            return guard
        if not base_url or not name:
            return _error(400, "base_url and name query parameters are required")
        try:
            store.delete_model(base_url, name)
        except NotFoundError as exc:
            return _error(404, str(exc))
        except StoreError as exc:
            return _error(503, str(exc))
        return Response(status_code=204)

    @app.api_route("/admin/api/prompt-templates", methods=["GET"])
    async def list_prompt_templates() -> Response:
        return _json_response({"templates": prompt_store.list_templates()})

    @app.api_route("/admin/api/endpoints/models/prompt-template", methods=["POST"])
    async def inject_prompt_template(payload: dict) -> Response:
        """Swap the model's effective system prompt for a prompt template.

        Order matters: preamble → hook → read body → config write. The hook runs
        *before* the config write (a failing hook must never leave a
        half-injected model behind) and off the event loop — run_load_hook
        blocks for up to HOOK_TIMEOUT. A hook that succeeds without leaving a
        usable ``prompt.md`` is also a failure: injecting a name whose body can
        never be read would silently fall back to the handwritten prompt.
        Failure clears any template name already on the model; the handwritten
        system_prompt is never touched."""
        guard = _write_guard(store)
        if guard is not None:
            return guard
        base_url = payload.get("base_url")
        name = payload.get("name")
        template = payload.get("template")
        if not base_url or not name or not template:
            return _error(400, "base_url, name and template are required")
        try:
            model = _model_payload(store, str(base_url), str(name))
        except NotFoundError as exc:
            return _error(404, str(exc))
        if model.get("prompt_template"):
            return _error(409, "model is already injected; cancel the injection first")
        if str(template) not in prompt_store.list_templates():
            return _error(400, f"unknown prompt template: {template}")
        try:
            await asyncio.to_thread(prompt_store.run_load_hook, str(template))
        except HookError as exc:
            try:
                store.set_prompt_template(str(base_url), str(name), None)
            except StoreError:
                pass
            return _error(502, f"load_prompt 执行失败：{exc}；已清除注入")
        # preload 丢缓存重新读盘：钩子可能刚改写了 prompt.md，旧缓存不能赢。
        if not prompt_store.preload(str(template)):
            try:
                store.set_prompt_template(str(base_url), str(name), None)
            except StoreError:
                pass
            return _error(400, f"模板 {template} 没有可用的 prompt.md（正文为空或不存在）；已清除注入")
        try:
            store.set_prompt_template(str(base_url), str(name), str(template))
        except NotFoundError as exc:
            return _error(404, str(exc))
        except StoreError as exc:
            return _error(503, str(exc))
        return _json_response(_model_payload(store, str(base_url), str(name)))

    @app.api_route("/admin/api/endpoints/models/prompt-template", methods=["DELETE"])
    async def uninject_prompt_template(base_url: str | None = None,
                                       name: str | None = None) -> Response:
        guard = _write_guard(store)
        if guard is not None:
            return guard
        if not base_url or not name:
            return _error(400, "base_url and name query parameters are required")
        try:
            model = _model_payload(store, base_url, name)
        except NotFoundError as exc:
            return _error(404, str(exc))
        if not model.get("prompt_template"):
            return Response(status_code=204)
        try:
            store.set_prompt_template(base_url, name, None)
        except StoreError as exc:
            return _error(503, str(exc))
        return _json_response({"name": name, "prompt_template": None})

    @app.api_route("/admin/api/extra-body", methods=["GET"])
    async def get_extra_body() -> Response:
        return _json_response({"extra_body": store.config.extra_body})

    @app.api_route("/admin/api/extra-body", methods=["PUT"])
    async def set_extra_body(payload: dict) -> Response:
        guard = _write_guard(store)
        if guard is not None:
            return guard
        extra_body = payload.get("extra_body")
        if extra_body is not None and not isinstance(extra_body, dict):
            return _error(400, "extra_body must be an object")
        if extra_body is not None:
            for key, value in extra_body.items():
                if key not in PROTOCOL_VALUES:
                    return _error(400, f"unknown extra_body protocol: {key}")
                if not isinstance(value, dict):
                    return _error(400, f"server.extra_body.{key} must be an object")
        try:
            store.update_extra_body(extra_body)
        except StoreError as exc:
            return _error(503, str(exc))
        return _json_response({"extra_body": store.config.extra_body})

    @app.api_route("/admin/api/upstream-models", methods=["GET"])
    async def fetch_upstream_models(base_url: str | None = None) -> Response:
        """Fetch an endpoint's /v1/models server-side, carrying only the
        endpoint's configured custom headers (typically a User-Agent; the list
        endpoint is usually public). Only configured base_urls are accepted.
        ``new_models`` lists ids absent from the watcher's baseline cache; a
        successful fetch refreshes that baseline so seen models aren't
        re-reported as news."""
        if not base_url:
            return _error(400, "base_url query parameter is required")
        entries = store.endpoint_dicts(base_url=base_url)
        if not entries:
            return _error(404, f"endpoint not found: {base_url}")
        custom = entries[0].get("custom_headers") or {}
        socks5_proxy = entries[0].get("socks5_proxy")
        news = [str(i) for i in entries[0].get("models_news") or []]
        try:
            if http_client_for is not None:
                ids = await fetch_models(base_url, custom, http_client_for(socks5_proxy))
            elif http_client is not None:
                ids = await fetch_models(base_url, custom, http_client)
            else:
                async with httpx.AsyncClient(timeout=FETCH_MODELS_TIMEOUT) as client:
                    ids = await fetch_models(base_url, custom, client)
        except FetchError as exc:
            return _error(502, str(exc))
        # Chips mark everything "new since the user last looked": models the
        # watcher flagged (models_news — already folded into the cache baseline
        # at detection time, so a pure cache diff would miss them) plus models
        # that appeared since the last baseline update and are only discovered
        # by this very fetch.
        new_models = news
        if watcher_cache_path is not None:
            cache = load_cache(Path(watcher_cache_path))
            record = cache.get(base_url)
            old_ids = record.get("ids") if isinstance(record, dict) else None
            if old_ids is not None:
                old = set(old_ids)
                known = set(news)
                new_models = news + [i for i in ids if i not in old and i not in known]
            cache[base_url] = {"ids": ids, "ts": int(time.time())}
            try:
                save_cache(Path(watcher_cache_path), cache)
            except OSError:
                pass   # baseline refresh is best-effort; never block the UI
        return _json_response({"models": ids, "new_models": new_models})

    @app.api_route("/admin/api/endpoints/models-news", methods=["POST"])
    async def clear_models_news(payload: dict) -> Response:
        """Silence the endpoint's「模型上新」badge (set when the fetch-models
        modal opens — seen counts as acknowledged)."""
        guard = _write_guard(store)
        if guard is not None:
            return guard
        base_url = payload.get("base_url")
        if not base_url:
            return _error(400, "base_url is required")
        try:
            store.set_models_news(str(base_url), None)
        except NotFoundError as exc:
            return _error(404, str(exc))
        except StoreError as exc:
            return _error(503, str(exc))
        return _json_response({"base_url": str(base_url), "models_news": []})

    @app.api_route("/admin/api/scan-models", methods=["GET"])
    async def scan_models() -> Response:
        """Manually run one watcher scan over all endpoints (testing aid; no
        UI). Returns per-endpoint fetch/diff results."""
        if model_watcher is None:
            return _error(503, "model watcher is not running (read-only mode)")
        results = await model_watcher.scan_once()
        return _json_response({"results": results})
