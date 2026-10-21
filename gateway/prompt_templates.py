"""Prompt template store (prompt-injection feature).

Discovers prompt-template folders under the ``prompts/`` root and lazily
caches each folder's ``prompt.md`` body in memory. Construction performs no
I/O at all — ``prompt.md`` is only read on the first ``get_prompt`` — so a
missing or broken template can never fail the gateway boot.
"""

from __future__ import annotations

import threading
from pathlib import Path

DEFAULT_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"

PROMPT_FILENAME = "prompt.md"


def _is_valid_name(name: str) -> bool:
    """A template name is a single path segment: non-blank, not ``.``/``..``,
    and free of path separators. Anything else must never touch the disk."""
    if not isinstance(name, str):
        return False
    return (
        name.strip() != ""
        and name not in {".", ".."}
        and "/" not in name
        and "\\" not in name
    )


class PromptTemplateStore:
    """目录扫描 + 正文懒加载缓存。

    - ``list_templates`` 每次实时扫目录（只读目录，不加锁）。
    - ``get_prompt`` 首次读取 ``prompt.md``，结果（含“不存在”这一负结果）
      在进程生命周期内保持缓存；``clear_negative`` 让下次调用重试。
    """

    def __init__(self, root: str | Path = DEFAULT_PROMPTS_DIR) -> None:
        self._root = Path(root)
        self._cache: dict[str, str | None] = {}
        self._lock = threading.Lock()

    def list_templates(self) -> list[str]:
        """升序返回所有带 ``prompt.md`` 的子目录名；根目录缺失返回空列表。"""
        try:
            entries = list(self._root.iterdir())
        except OSError:
            return []
        return sorted(
            p.name for p in entries if p.is_dir() and (p / PROMPT_FILENAME).is_file()
        )

    def get_prompt(self, name: str) -> str | None:
        """返回模板正文；名字非法、文件缺失、读取失败或正文全空白时返回 None。"""
        if not _is_valid_name(name):
            return None
        with self._lock:
            if name in self._cache:
                return self._cache[name]
            content: str | None = None
            try:
                text = (self._root / name / PROMPT_FILENAME).read_text(encoding="utf-8")
            except (OSError, ValueError):
                # ValueError 覆盖 UnicodeDecodeError（非 UTF-8 文件）与内嵌 NUL
                # 名字导致的 "embedded null byte"；都按“模板不可用”缓存负结果。
                text = ""
            if text.strip():
                content = text
            self._cache[name] = content
            return content

    def preload(self, name: str) -> bool:
        """清掉负结果后重试加载，返回模板是否可用。"""
        self.clear_negative(name)
        return self.get_prompt(name) is not None

    def clear_negative(self, name: str) -> None:
        """丢掉缓存的“不存在”结果，让下次 get_prompt 重新读盘。"""
        with self._lock:
            if self._cache.get(name, "missing") is None:
                del self._cache[name]
