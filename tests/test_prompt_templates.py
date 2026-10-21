import pathlib
import subprocess

import pytest

from gateway.prompt_templates import (
    _parse_hook_result,
    HookError,
    PromptTemplateStore,
)


def make_root(tmp_path):
    root = tmp_path / "prompts"
    (root / "alpha").mkdir(parents=True)
    (root / "alpha" / "prompt.md").write_text("ALPHA", encoding="utf-8")
    (root / "beta").mkdir()
    (root / "beta" / "prompt.md").write_text("BETA", encoding="utf-8")
    (root / "no-md").mkdir()          # 没有 prompt.md -> 不进列表
    (root / "alpha" / "notes.txt").write_text("x", encoding="utf-8")
    return root


def test_list_templates_only_dirs_with_prompt_md(tmp_path):
    store = PromptTemplateStore(make_root(tmp_path))
    assert store.list_templates() == ["alpha", "beta"]


def test_list_templates_missing_root_is_empty(tmp_path):
    assert PromptTemplateStore(tmp_path / "nope").list_templates() == []


def test_construct_does_not_read_prompt_md(tmp_path, monkeypatch):
    root = make_root(tmp_path)
    seen = []
    real = pathlib.Path.read_text
    monkeypatch.setattr(pathlib.Path, "read_text",
                        lambda self, *a, **k: seen.append(self) or real(self, *a, **k))
    PromptTemplateStore(root)
    assert seen == []


def test_get_prompt_reads_once_then_caches(tmp_path, monkeypatch):
    store = PromptTemplateStore(make_root(tmp_path))
    reads = []
    real = pathlib.Path.read_text
    monkeypatch.setattr(pathlib.Path, "read_text",
                        lambda self, *a, **k: reads.append(self) or real(self, *a, **k))
    assert store.get_prompt("alpha") == "ALPHA"
    assert store.get_prompt("alpha") == "ALPHA"
    assert len(reads) == 1


def test_get_prompt_missing_file_caches_negative_and_clear_negative_retries(tmp_path):
    root = make_root(tmp_path)
    store = PromptTemplateStore(root)
    assert store.get_prompt("no-md") is None
    (root / "no-md" / "prompt.md").write_text("LATE", encoding="utf-8")
    assert store.get_prompt("no-md") is None          # 负结果已缓存
    store.clear_negative("no-md")
    assert store.get_prompt("no-md") == "LATE"


@pytest.mark.parametrize("content", ["", "   \n\t", "\n\n"])
def test_blank_prompt_is_treated_as_missing(tmp_path, content):
    root = make_root(tmp_path)
    (root / "alpha" / "prompt.md").write_text(content, encoding="utf-8")
    assert PromptTemplateStore(root).get_prompt("alpha") is None


@pytest.mark.parametrize("name", ["", ".", "..", "a/b", "a\\b", "../alpha"])
def test_illegal_names_never_touch_disk(tmp_path, monkeypatch, name):
    store = PromptTemplateStore(make_root(tmp_path))
    reads = []
    monkeypatch.setattr(pathlib.Path, "read_text",
                        lambda self, *a, **k: reads.append(self) or "X")
    assert store.get_prompt(name) is None
    assert reads == []


def test_undecodable_prompt_is_treated_as_missing_and_cached(tmp_path, monkeypatch):
    root = make_root(tmp_path)
    (root / "alpha" / "prompt.md").write_bytes("中文提示词".encode("gbk"))
    store = PromptTemplateStore(root)
    assert store.get_prompt("alpha") is None        # 非法 UTF-8 -> 负结果
    reads = []
    monkeypatch.setattr(pathlib.Path, "read_text",
                        lambda self, *a, **k: reads.append(self) or "X")
    assert store.get_prompt("alpha") is None        # 负结果已缓存，不再读盘
    assert reads == []


def test_nul_byte_name_returns_none_without_raising(tmp_path, monkeypatch):
    store = PromptTemplateStore(make_root(tmp_path))
    reads = []
    real = pathlib.Path.read_text

    def spy(self, *a, **k):
        if "\x00" in str(self):                     # 真实 read_text 同样抛 ValueError
            raise ValueError("embedded null byte")
        reads.append(self)
        return real(self, *a, **k)

    monkeypatch.setattr(pathlib.Path, "read_text", spy)
    assert store.get_prompt("a\x00b") is None       # 内嵌 NUL -> ValueError 被吞掉
    assert store.get_prompt("a\x00b") is None       # 第二次仍返回 None，不抛
    assert reads == []


def test_preload_reports_whether_prompt_is_available(tmp_path):
    root = make_root(tmp_path)
    store = PromptTemplateStore(root)
    assert store.preload("alpha") is True
    assert store.preload("no-md") is False


def _template_with_py(tmp_path, body):
    root = tmp_path / "prompts"
    d = root / "tpl"
    d.mkdir(parents=True)
    (d / "prompt.md").write_text("P", encoding="utf-8")
    (d / "prompt.py").write_text(body, encoding="utf-8")
    return root


def _forbid_spawn(monkeypatch):
    def spawn(*a, **k):
        pytest.fail("spawned a subprocess")

    monkeypatch.setattr(subprocess, "run", spawn)


def test_hook_missing_py_is_success(tmp_path, monkeypatch):
    _forbid_spawn(monkeypatch)
    root = make_root(tmp_path)                      # alpha 只有 prompt.md
    PromptTemplateStore(root).run_load_hook("alpha")  # 不抛异常即通过


def test_hook_zero_return_is_success_and_runs_in_template_dir(tmp_path):
    root = _template_with_py(tmp_path,
        "import pathlib\n"
        "def load_prompt():\n"
        "    pathlib.Path('ran.txt').write_text('yes', encoding='utf-8')\n"
        "    return 0\n")
    PromptTemplateStore(root).run_load_hook("tpl")
    assert (root / "tpl" / "ran.txt").read_text(encoding="utf-8") == "yes"


def test_hook_non_int_return_is_success(tmp_path):
    root = _template_with_py(tmp_path, "def load_prompt():\n    return None\n")
    PromptTemplateStore(root).run_load_hook("tpl")


@pytest.mark.parametrize("literal", ["True", "False"],
                         ids=["returns_True", "returns_False"])
def test_hook_bool_return_is_success(tmp_path, literal):
    # 契约：bool 是 isinstance(int) 的特例，但 True/False 都不当作“非 0 失败”。
    root = _template_with_py(tmp_path,
        f"def load_prompt():\n    return {literal}\n")
    PromptTemplateStore(root).run_load_hook("tpl")


def test_hook_nonzero_return_raises(tmp_path):
    root = _template_with_py(tmp_path, "def load_prompt():\n    return 3\n")
    with pytest.raises(HookError, match="3"):
        PromptTemplateStore(root).run_load_hook("tpl")


def test_hook_exception_raises_with_stderr(tmp_path):
    root = _template_with_py(tmp_path,
        "def load_prompt():\n    raise RuntimeError('boom-marker')\n")
    with pytest.raises(HookError, match="boom-marker"):
        PromptTemplateStore(root).run_load_hook("tpl")


def test_hook_syntax_error_raises(tmp_path):
    root = _template_with_py(tmp_path, "def load_prompt(:\n")
    with pytest.raises(HookError):
        PromptTemplateStore(root).run_load_hook("tpl")


def test_hook_missing_function_raises(tmp_path):
    root = _template_with_py(tmp_path, "def other():\n    return 0\n")
    with pytest.raises(HookError, match="load_prompt"):
        PromptTemplateStore(root).run_load_hook("tpl")


def test_hook_timeout_raises(tmp_path, monkeypatch):
    import gateway.prompt_templates as pt
    monkeypatch.setattr(pt, "HOOK_TIMEOUT", 0.2)
    root = _template_with_py(tmp_path,
        "import time\ndef load_prompt():\n    time.sleep(5)\n    return 0\n")
    with pytest.raises(HookError, match="超时|timeout"):
        PromptTemplateStore(root).run_load_hook("tpl")


def test_hook_illegal_name_raises_without_spawn(tmp_path, monkeypatch):
    _forbid_spawn(monkeypatch)
    store = PromptTemplateStore(make_root(tmp_path))
    with pytest.raises(HookError):
        store.run_load_hook("../alpha")


def test_hook_unterminated_stdout_before_nonzero_return_raises(tmp_path):
    # 钩子写了不带换行的输出，结果标记不再位于行首；必须仍取最后一个标记。
    root = _template_with_py(tmp_path,
        "import sys\n"
        "def load_prompt():\n"
        "    sys.stdout.write('working')\n"
        "    return 3\n")
    with pytest.raises(HookError, match="3"):
        PromptTemplateStore(root).run_load_hook("tpl")


def test_hook_unterminated_stdout_before_zero_return_is_success(tmp_path):
    root = _template_with_py(tmp_path,
        "import sys\n"
        "def load_prompt():\n"
        "    sys.stdout.write('working')\n"
        "    return 0\n")
    PromptTemplateStore(root).run_load_hook("tpl")


def test_hook_non_utf8_stdout_still_works(tmp_path):
    root = _template_with_py(tmp_path,
        "import sys\n"
        "def load_prompt():\n"
        "    sys.stdout.buffer.write(b'\\xff\\xfe')\n"
        "    return 0\n")
    PromptTemplateStore(root).run_load_hook("tpl")


def test_hook_prompt_py_directory_raises(tmp_path, monkeypatch):
    _forbid_spawn(monkeypatch)
    root = _template_with_py(tmp_path, "def load_prompt():\n    return 0\n")
    (root / "tpl" / "prompt.py").unlink()
    (root / "tpl" / "prompt.py").mkdir()            # 目录而非普通文件 -> 配置事故
    with pytest.raises(HookError, match="普通文件"):
        PromptTemplateStore(root).run_load_hook("tpl")


@pytest.mark.parametrize("stdout", [
    "__HOOK_RESULT__{not json}",
    "__HOOK_RESULT__{\"ok\": tru",
])
def test_parse_hook_result_rejects_invalid_json_after_marker(stdout):
    assert _parse_hook_result(stdout) is None


@pytest.mark.parametrize("stdout", [
    "__HOOK_RESULT__[]",
    "__HOOK_RESULT__[1, 2]",
    "__HOOK_RESULT__3",
    '__HOOK_RESULT__"text"',
])
def test_parse_hook_result_rejects_json_non_dict(stdout):
    assert _parse_hook_result(stdout) is None


def test_parse_hook_result_without_marker_is_none():
    assert _parse_hook_result("") is None
    assert _parse_hook_result("no marker here\n") is None


def test_hook_present_unparseable_stdout_raises(tmp_path, monkeypatch):
    # runner 被替换成“什么都不打印”的等价行为：直接 monkeypatch _RUNNER 不打印结果。
    import gateway.prompt_templates as pt

    root = _template_with_py(tmp_path, "def load_prompt():\n    return 0\n")
    monkeypatch.setattr(pt, "_RUNNER", "import sys\nsys.exit(0)\n")
    with pytest.raises(HookError, match="无法解析"):
        PromptTemplateStore(root).run_load_hook("tpl")


def test_hook_broken_symlink_raises(tmp_path, monkeypatch):
    _forbid_spawn(monkeypatch)
    root = _template_with_py(tmp_path, "def load_prompt():\n    return 0\n")
    (root / "tpl" / "prompt.py").unlink()
    (root / "tpl" / "prompt.py").symlink_to(root / "tpl" / "missing.py")
    with pytest.raises(HookError, match="普通文件"):
        PromptTemplateStore(root).run_load_hook("tpl")
