"""Prompt template store (prompt-injection feature).

Discovers prompt-template folders under the ``prompts/`` root and lazily
caches each folder's ``prompt.md`` body in memory. Construction performs no
I/O at all — ``prompt.md`` is only read on the first ``get_prompt`` — so a
missing or broken template can never fail the gateway boot.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import threading
from pathlib import Path

DEFAULT_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"

PROMPT_FILENAME = "prompt.md"

HOOK_FILENAME = "prompt.py"

HOOK_TIMEOUT: float = 30.0

# 子进程侧的最小 runner：用 runpy 执行模板脚本，再调用它定义的 load_prompt。
# 结果以 ``_RESULT_MARKER`` 前缀 + JSON 回报（解析时取最后一次出现），标准输入
# 关闭、-I 隔离环境，脚本自身抛异常时靠非零退出码 + stderr 回溯来定位。
# 用 %-格式化而不是 f-string：runner 里全是 JSON 大括号，f-string 会把每个
# 括号都写成 {{ }}，可读性更差。
_RESULT_MARKER = "__HOOK_RESULT__"

_RUNNER = (
    "import json, runpy, sys\n"
    "ns = runpy.run_path(sys.argv[1], run_name='__main__')\n"
    "fn = ns.get('load_prompt')\n"
    "if not callable(fn):\n"
    "    print('%(marker)s' + json.dumps({'ok': False, 'error': 'load_prompt not defined'}))\n"
    "else:\n"
    "    print('%(marker)s' + json.dumps({'ok': True, 'result': None}))\n"
    "    r = fn()\n"
    "    ok = (not isinstance(r, int)) or isinstance(r, bool) or r == 0\n"
    "    print('%(marker)s' + json.dumps({'ok': ok, 'result': r if isinstance(r, int) else None}))\n"
) % {"marker": _RESULT_MARKER}


class HookError(Exception):
    """模板 ``prompt.py`` 的 ``load_prompt`` 钩子执行失败。"""


def _tail(text: str | bytes | None, limit: int = 500) -> str:
    """取 stderr 尾部供 admin UI 展示；非文本或替换解码后的字节也要能显示。"""
    if not text:
        return ""
    if isinstance(text, bytes):
        text = text.decode("utf-8", "replace")
    return text[-limit:]


def _parse_hook_result(stdout: str) -> dict | None:
    """取**最后一次** ``__HOOK_RESULT__`` 标记并解析其后的 JSON。

    用 ``rfind`` 而不是按行扫描：模板自己用 ``sys.stdout.write`` 或
    ``print(..., end='')`` 输出时，标记可能紧贴在前一段文本末尾而不在行首；
    此时行扫描会错误地回退到调用前打印的那行乐观结果。
    拿不到标记或 JSON 不合法时返回 None。
    """
    stdout = stdout or ""
    pos = stdout.rfind(_RESULT_MARKER)
    if pos < 0:
        return None
    try:
        payload, _ = json.JSONDecoder().raw_decode(stdout, pos + len(_RESULT_MARKER))
    except ValueError:
        return None
    return payload if isinstance(payload, dict) else None


def _read_prompt_file(folder: Path) -> str | None:
    """读取模板目录下的 ``prompt.md``；缺失、不可读、非 UTF-8 或全空白返回 None。"""
    try:
        text = (folder / PROMPT_FILENAME).read_text(encoding="utf-8")
    except (OSError, ValueError):
        # ValueError 覆盖 UnicodeDecodeError（非 UTF-8 文件）与内嵌 NUL 名字
        # 导致的 "embedded null byte"；都按“模板不可用”处理。
        return None
    return text if text.strip() else None


def _has_hook(folder: Path) -> bool:
    """目录下有普通文件形式的 ``prompt.py``（断链/目录不算）。"""
    script = folder / HOOK_FILENAME
    try:
        return script.is_file()
    except OSError:
        return False


def _is_valid_name(name: str) -> bool:
    """模板名是单个路径段：非空白、不是 ``.``/``..``、不含路径分隔符。

    其他一切名字都不允许碰磁盘。
    """
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
    - ``get_prompt`` 首次取正文时：目录里有 ``prompt.md`` 就直接读；只有
      ``prompt.py`` 时先跑钩子（期望它生成 ``prompt.md``）再读，读不到就是
      “模板不可用”。结果（含负结果）在进程生命周期内保持缓存；
      ``preload`` 强制重新加载。
    """

    def __init__(self, root: str | Path = DEFAULT_PROMPTS_DIR) -> None:
        self._root = Path(root).resolve()
        self._cache: dict[str, str | None] = {}
        self._lock = threading.Lock()

    def list_templates(self) -> list[str]:
        """升序返回 ``prompts/`` 下所有子目录名；根目录缺失返回空列表。

        只看目录名，不看内容：只有 ``prompt.py``（正文由钩子生成）或暂时为空的
        目录也要能在注入弹窗里选到，否则「先建目录再生成正文」永远无法注入。
        """
        try:
            entries = list(self._root.iterdir())
        except OSError:
            return []
        return sorted(p.name for p in entries if p.is_dir())

    def get_prompt(self, name: str) -> str | None:
        """返回模板正文；名字非法、正文缺失/读取失败/全空白时返回 None。

        目录里没有 ``prompt.md`` 但存在 ``prompt.py`` 时，先执行一次钩子
        （规则 3：py-only 模板在首次使用时生成正文）。钩子失败按不可用处理，
        绝不让代理请求 500 —— 调用方回落到手写 ``system_prompt``。
        """
        if not _is_valid_name(name):
            return None
        with self._lock:
            if name in self._cache:
                return self._cache[name]
            self._load_locked(name)
            return self._cache[name]

    async def aget_prompt(self, name: str) -> str | None:
        """``get_prompt`` 的异步版本：钩子在线程里跑，不阻塞事件循环。

        代理请求路径必须用这个：钩子最长可跑 30 秒，直接在事件循环里
        ``subprocess.run`` 会卡住整个网关。
        """
        if not _is_valid_name(name):
            return None
        with self._lock:
            if name in self._cache:
                return self._cache[name]
        # 冷缓存：可能触发钩子 → 让给线程。持有锁会阻塞其他请求，所以先放锁，
        # 由 get_prompt 内部重新加锁并写缓存（并发重复加载是良性的）。
        return await asyncio.to_thread(self.get_prompt, name)

    def _load_locked(self, name: str) -> None:
        """（持锁调用）读正文并写缓存：md 优先，缺失时跑钩子后重读。"""
        folder = self._root / name
        content = _read_prompt_file(folder)
        if content is None and _has_hook(folder):
            try:
                self.run_load_hook(name)
            except HookError:
                content = None
            else:
                content = _read_prompt_file(folder)
        self._cache[name] = content

    def preload(self, name: str) -> bool:
        """丢弃已有缓存后重新加载（注入时调用），返回模板是否可用。

        注入时钩子刚跑过、正文刚被改写，必须重新读盘；只清负结果会让旧正文
        继续生效，改 ``prompt.md`` 后注入看起来“没反应”。
        """
        if not _is_valid_name(name):
            return False
        with self._lock:
            self._cache.pop(name, None)
            self._load_locked(name)
            return self._cache[name] is not None

    def clear_negative(self, name: str) -> None:
        """丢掉缓存的“不存在”结果，让下次 get_prompt 重新读盘。"""
        with self._lock:
            if self._cache.get(name, "missing") is None:
                del self._cache[name]

    def run_load_hook(self, name: str) -> None:
        """执行模板目录下的 ``prompt.py`` 钩子的 ``load_prompt()``。

        - 没有 ``prompt.py``：直接返回（钩子可选）。
        - 名字非法：在起子进程之前就抛 ``HookError``。
        - 返回非 ``int``（含 ``None``）、或 ``True``、或 ``0``：成功。
        - 返回非 0 整数、未定义函数、抛异常/语法错、非零退出、超时：抛
          ``HookError``，消息带上 stderr 尾部供 admin UI 展示。
        """
        if not _is_valid_name(name):
            raise HookError(f"非法的模板名: {name!r}")
        script = self._root / name / HOOK_FILENAME
        # is_symlink() 兜住断链（exists() 对断链返回 False，会被误判成“钩子不存在”）。
        if not (script.exists() or script.is_symlink()):
            return
        if not script.is_file():
            # 存在但不是普通文件（目录、FIFO、断链）：配置事故，报错别静默跳过。
            raise HookError(f"{HOOK_FILENAME} 不是普通文件：{script}")
        try:
            completed = subprocess.run(
                [sys.executable, "-I", "-B", "-c", _RUNNER, str(script)],
                cwd=str(script.parent),
                capture_output=True,
                encoding="utf-8",
                errors="replace",
                timeout=HOOK_TIMEOUT,
                stdin=subprocess.DEVNULL,
            )
        except subprocess.TimeoutExpired as exc:
            # 超时被杀后 exc.stdout/err 里仍有已读到的输出，别丢掉。
            raise HookError(
                f"load_prompt 超时（> {HOOK_TIMEOUT:g}s）：{_tail(exc.stderr)}"
            ) from None
        except OSError as exc:
            raise HookError(f"load_prompt 无法启动：{exc}") from None
        stderr_tail = _tail(completed.stderr)
        if completed.returncode != 0:
            raise HookError(f"load_prompt 执行失败：{stderr_tail}")
        payload = _parse_hook_result(completed.stdout)
        if payload is None:
            raise HookError(f"load_prompt 输出无法解析：{stderr_tail}")
        if not payload.get("ok"):
            raise HookError(
                payload.get("error")
                or f"load_prompt 返回非 0 值 {payload.get('result')}"
            )
