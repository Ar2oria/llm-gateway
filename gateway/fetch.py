"""Shared server-side fetch of an endpoint's /v1/models list. Used by the
admin「获取模型」route and the periodic model watcher (watcher.py)."""

from __future__ import annotations

import httpx

FETCH_MODELS_PATH = "/v1/models"
FETCH_MODELS_TIMEOUT = 10.0   # a hung upstream must not hang the admin UI


class FetchError(Exception):
    """Upstream unreachable / non-200 / non-JSON for /v1/models."""


def extract_model_ids(payload: object) -> list[str]:
    """Model ids from an OpenAI-style /v1/models response
    ({"data": [{"id": ...}]}); tolerates a bare list, skips junk, dedupes."""
    items = payload.get("data") if isinstance(payload, dict) else payload
    if not isinstance(items, list):
        return []
    ids: list[str] = []
    for item in items:
        mid = item.get("id") if isinstance(item, dict) else None
        if isinstance(mid, str) and mid and mid not in ids:
            ids.append(mid)
    return ids


async def fetch_models(base_url: str, custom_headers: dict[str, str],
                       http_client: httpx.AsyncClient) -> list[str]:
    """Fetch {base_url}/v1/models carrying only the endpoint's configured
    headers. Returns the deduped id list. Raises FetchError on transport
    failure, non-200 status, or non-JSON body."""
    url = base_url.rstrip("/") + FETCH_MODELS_PATH
    headers = {str(k).lower(): str(v) for k, v in custom_headers.items()}
    try:
        resp = await http_client.get(url, headers=headers, timeout=FETCH_MODELS_TIMEOUT)
    except httpx.HTTPError as exc:
        raise FetchError(f"failed to reach upstream /v1/models: {exc}") from exc
    if resp.status_code != 200:
        raise FetchError(f"upstream returned HTTP {resp.status_code} for /v1/models")
    try:
        payload = resp.json()
    except ValueError as exc:
        raise FetchError("upstream returned non-JSON for /v1/models") from exc
    return extract_model_ids(payload)
