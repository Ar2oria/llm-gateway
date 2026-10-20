from __future__ import annotations

import contextlib
import copy
import json
import os
import stat
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path

from gateway.config import (
    ConfigError,
    EndpointConfig,
    GatewayConfig,
    ModelRouter,
    config_to_dict,
    normalize_base_url,
    validate_config,
)


class StoreError(Exception):
    pass


class NotFoundError(StoreError):
    pass


NO_PATH_MESSAGE = "config file path is not set; admin write operations are disabled"

# Sentinel for update_model: distinguishes "pricing not provided (keep)"
# from "pricing explicitly cleared (remove)".
_UNSET = object()


@dataclass(frozen=True)
class _Snapshot:
    config: GatewayConfig
    router: ModelRouter
    raw: dict


class ConfigStore:
    def __init__(self, config: GatewayConfig, config_path: str | Path | None = None) -> None:
        self._path = Path(config_path) if config_path is not None else None
        if self._path is not None:
            # Seed raw from the file itself: config_to_dict(config) cannot
            # represent endpoint entries with an empty models list, which
            # would silently vanish on write-back.
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            raw.setdefault("endpoints", [])
            for entry in raw["endpoints"]:
                if isinstance(entry, dict) and entry.get("base_url"):
                    entry["base_url"] = normalize_base_url(str(entry["base_url"]))
            # The caller's GatewayConfig may carry an effective value the file
            # does not (e.g. the --debug-log CLI override in __main__); keep it
            # authoritative so it survives the first admin write.
            server = raw.setdefault("server", {})
            server["debug_log"] = config.debug_log
        else:
            raw = config_to_dict(config)
        self._snapshot = _Snapshot(config, ModelRouter(config), raw)
        self._lock = threading.Lock()

    @property
    def config(self) -> GatewayConfig:
        return self._snapshot.config

    @property
    def router(self) -> ModelRouter:
        return self._snapshot.router

    @property
    def path(self) -> Path | None:
        return self._path

    def update_extra_body(self, extra_body: dict[str, object] | None) -> None:
        """Set (or clear) the global server.extra_body merged into converted
        upstream requests. ``None`` removes the key entirely."""
        with self._lock:
            raw = copy.deepcopy(self._snapshot.raw)
            server = raw.setdefault("server", {})
            if extra_body:
                server["extra_body"] = dict(extra_body)
            else:
                server.pop("extra_body", None)
            self._commit(raw)

    def set_models_news(self, base_url: str, ids: list[str] | None) -> None:
        """Set (or clear with ``None``/empty) the endpoint's models_news flag
        list — upstream ids the watcher saw as new since the user last opened
        the fetch-models modal."""
        with self._lock:
            raw = copy.deepcopy(self._snapshot.raw)
            base_url = normalize_base_url(str(base_url))
            entry = self._find_endpoint_raw(raw, base_url)
            if entry is None:
                raise NotFoundError(f"endpoint not found: {base_url}")
            if ids:
                entry["models_news"] = [str(i) for i in ids]
            else:
                entry.pop("models_news", None)
            self._commit(raw)

    def add_endpoint(self, base_url: str, custom_headers: dict[str, str] | None = None) -> None:
        with self._lock:
            raw = copy.deepcopy(self._snapshot.raw)
            base_url = normalize_base_url(str(base_url))
            if self._find_endpoint_raw(raw, base_url) is not None:
                raise ConfigError("endpoint base_url values must be unique")
            raw["endpoints"].append({
                "base_url": base_url,
                "custom_headers": {str(k): str(v) for k, v in (custom_headers or {}).items()},
                "models": [],
            })
            self._commit(raw)

    def update_endpoint(self, base_url: str, custom_headers: dict[str, str]) -> None:
        with self._lock:
            raw = copy.deepcopy(self._snapshot.raw)
            base_url = normalize_base_url(str(base_url))
            entry = self._find_endpoint_raw(raw, base_url)
            if entry is None:
                raise NotFoundError(f"endpoint not found: {base_url}")
            entry["custom_headers"] = {str(k): str(v) for k, v in custom_headers.items()}
            self._commit(raw)

    def delete_endpoint(self, base_url: str) -> None:
        with self._lock:
            raw = copy.deepcopy(self._snapshot.raw)
            base_url = normalize_base_url(str(base_url))
            if self._find_endpoint_raw(raw, base_url) is None:
                raise NotFoundError(f"endpoint not found: {base_url}")
            raw["endpoints"] = [e for e in raw["endpoints"] if e.get("base_url") != base_url]
            self._commit(raw)

    def add_model(
        self,
        base_url: str,
        name: str,
        upstream_model: str | None = None,
        system_prompt: str | None = None,
        pricing: dict[str, float] | None = None,
        protocol: str | None = None,
    ) -> EndpointConfig:
        with self._lock:
            raw = copy.deepcopy(self._snapshot.raw)
            base_url = normalize_base_url(str(base_url))
            entry = self._find_endpoint_raw(raw, base_url)
            if entry is None:
                raise NotFoundError(f"endpoint not found: {base_url}")
            name = str(name)
            if any(m.get("name") == name for m in entry["models"]):
                raise ConfigError("endpoint model names must be unique")
            model_entry: dict[str, object] = {"name": name}
            if upstream_model:
                model_entry["upstream_model"] = str(upstream_model)
            if system_prompt:
                model_entry["system_prompt"] = str(system_prompt)
            if pricing:
                model_entry["pricing"] = dict(pricing)
            if protocol:
                model_entry["protocol"] = str(protocol)
            entry["models"].append(model_entry)
            self._commit(raw)
            return self._find_model(base_url, name)

    def update_model(
        self,
        base_url: str,
        name: str,
        upstream_model: str | None = None,
        system_prompt: str | None = None,
        pricing: dict[str, float] | None | object = _UNSET,
        protocol: str | None | object = _UNSET,
    ) -> EndpointConfig:
        with self._lock:
            raw = copy.deepcopy(self._snapshot.raw)
            base_url = normalize_base_url(str(base_url))
            entry = self._find_endpoint_raw(raw, base_url)
            if entry is None:
                raise NotFoundError(f"endpoint not found: {base_url}")
            name = str(name)
            model_entry = self._find_model_raw(entry, name)
            if model_entry is None:
                raise NotFoundError(f"model not found: {name}")
            if upstream_model:
                model_entry["upstream_model"] = str(upstream_model)
            else:
                model_entry.pop("upstream_model", None)
            if system_prompt:
                model_entry["system_prompt"] = str(system_prompt)
            else:
                model_entry.pop("system_prompt", None)
            if pricing is not _UNSET:
                if pricing:
                    model_entry["pricing"] = dict(pricing)
                else:
                    model_entry.pop("pricing", None)
            if protocol is not _UNSET:
                if protocol:
                    model_entry["protocol"] = str(protocol)
                else:
                    model_entry.pop("protocol", None)
            self._commit(raw)
            return self._find_model(base_url, name)

    def delete_model(self, base_url: str, name: str) -> None:
        with self._lock:
            raw = copy.deepcopy(self._snapshot.raw)
            base_url = normalize_base_url(str(base_url))
            entry = self._find_endpoint_raw(raw, base_url)
            if entry is None:
                raise NotFoundError(f"endpoint not found: {base_url}")
            name = str(name)
            if self._find_model_raw(entry, name) is None:
                raise NotFoundError(f"model not found: {name}")
            entry["models"] = [m for m in entry["models"] if m.get("name") != name]
            self._commit(raw)

    def endpoint_dicts(self, base_url: str | None = None) -> list[dict]:
        """Grouped endpoint representation for the admin API, built from the
        canonical raw (the flattened GatewayConfig cannot represent
        endpoints with an empty models list)."""
        if base_url is not None:
            base_url = normalize_base_url(str(base_url))
        result = []
        for entry in self._snapshot.raw["endpoints"]:
            if base_url is not None and entry.get("base_url") != base_url:
                continue
            result.append({
                "base_url": entry["base_url"],
                "custom_headers": dict(entry.get("custom_headers") or {}),
                "models_news": [str(i) for i in entry.get("models_news") or []],
                "models": [
                    {
                        "name": m["name"],
                        "upstream_model": m.get("upstream_model"),
                        "system_prompt": m.get("system_prompt"),
                        "pricing": m.get("pricing"),
                        "protocol": m.get("protocol"),
                    }
                    for m in entry.get("models", [])
                ],
            })
        return result

    def _find_endpoint_raw(self, raw: dict, base_url: str) -> dict | None:
        for entry in raw["endpoints"]:
            if entry.get("base_url") == base_url:
                return entry
        return None

    @staticmethod
    def _find_model_raw(entry: dict, name: str) -> dict | None:
        for model_entry in entry["models"]:
            if model_entry.get("name") == name:
                return model_entry
        return None

    def _find_model(self, base_url: str, name: str) -> EndpointConfig:
        for ep in self.config.endpoints:
            if ep.base_url == base_url and ep.model == name:
                return ep
        raise NotFoundError(f"model not found: {name}")

    def _commit(self, raw: dict) -> None:
        config = validate_config(raw)
        if self._path is None:
            raise StoreError(NO_PATH_MESSAGE)
        try:
            self._atomic_write(raw)
        except OSError as exc:
            raise StoreError(f"failed to write config: {exc}") from exc
        self._snapshot = _Snapshot(config, ModelRouter(config), raw)

    def _atomic_write(self, raw: dict) -> None:
        assert self._path is not None
        data = json.dumps(raw, ensure_ascii=False, indent=2) + "\n"
        fd, tmp_name = tempfile.mkstemp(
            dir=str(self._path.parent), prefix=f".{self._path.name}.", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(data)
                fh.flush()
                os.fsync(fh.fileno())
            try:
                os.chmod(tmp_name, stat.S_IMODE(os.stat(self._path).st_mode))
            except FileNotFoundError:
                pass
            os.replace(tmp_name, self._path)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp_name)
            raise
        with contextlib.suppress(OSError):
            dfd = os.open(str(self._path.parent), os.O_RDONLY)
            try:
                os.fsync(dfd)
            finally:
                os.close(dfd)
