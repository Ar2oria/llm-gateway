import pathlib
import shutil
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


def test_list_templates_lists_every_subdirectory(tmp_path):
    # 模板目录是否可用与目录里有没有文件无关：名字存在就要能在注入弹窗里选到。
    # 只有 prompt.py（重启后靠钩子重建正文）或暂时为空的目录同样要出现。
    store = PromptTemplateStore(make_root(tmp_path))
    assert store.list_templates() == ["alpha", "beta", "no-md"]


def test_list_templates_lists_directory_without_any_files(tmp_path):
    # 规则 1：只看目录名。空目录（还没生成 prompt.md）也要列出来。
    root = make_root(tmp_path)
    (root / "empty").mkdir()
    assert "empty" in PromptTemplateStore(root).list_templates()


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

# --- md-only / py-only 模板 ---

def _py_only_template(tmp_path, body, name="tpl"):
    root = tmp_path / "prompts"
    d = root / name
    d.mkdir(parents=True)
    (d / "prompt.py").write_text(body, encoding="utf-8")
    return root


def _writes_md(body: str) -> str:
    return (
        "import pathlib\n"
        "def load_prompt():\n"
        "    pathlib.Path('prompt.md').write_text(%r, encoding='utf-8')\n"
        "    return 0\n" % body
    )


def test_get_prompt_md_only_reads_without_running_hook(tmp_path, monkeypatch):
    # 只有 prompt.md：直接用正文，不该起子进程。
    _forbid_spawn(monkeypatch)
    root = make_root(tmp_path)
    assert PromptTemplateStore(root).get_prompt("alpha") == "ALPHA"


def test_get_prompt_py_only_runs_hook_then_loads_generated_md(tmp_path):
    # 只有 prompt.py：首次取正文时执行 load_prompt()，再读它生成的 prompt.md。
    root = _py_only_template(tmp_path, _writes_md("GENERATED"))
    store = PromptTemplateStore(root)
    assert store.get_prompt("tpl") == "GENERATED"
    assert (root / "tpl" / "prompt.md").read_text(encoding="utf-8") == "GENERATED"


def test_get_prompt_py_only_runs_hook_once_then_caches(tmp_path, monkeypatch):
    root = _py_only_template(
        tmp_path,
        "import pathlib\n"
        "def load_prompt():\n"
        "    p = pathlib.Path('runs.txt')\n"
        "    n = int(p.read_text()) if p.exists() else 0\n"
        "    p.write_text(str(n + 1))\n"
        "    pathlib.Path('prompt.md').write_text('GENERATED', encoding='utf-8')\n"
        "    return 0\n")
    store = PromptTemplateStore(root)
    assert store.get_prompt("tpl") == "GENERATED"
    assert store.get_prompt("tpl") == "GENERATED"
    assert (root / "tpl" / "runs.txt").read_text() == "1"


def test_get_prompt_hook_failure_is_negative_and_does_not_raise(tmp_path):
    # 钩子失败不能让请求路径 500：按“模板不可用”处理（回落手写提示词）。
    root = _py_only_template(
        tmp_path,
        "import pathlib\ndef load_prompt():\n"
        "    pathlib.Path('ran.txt').write_text('x')\n"
        "    return 5\n")
    assert PromptTemplateStore(root).get_prompt("tpl") is None
    assert (root / "tpl" / "ran.txt").exists()      # 确认真的走到了钩子


def test_get_prompt_hook_succeeds_without_writing_md_is_none(tmp_path):
    # 契约：钩子跑成功但目录里仍没有 prompt.md → 不注入（回落手写提示词）。
    root = _py_only_template(
        tmp_path,
        "import pathlib\ndef load_prompt():\n"
        "    pathlib.Path('ran.txt').write_text('x')\n"
        "    return 0\n")
    assert PromptTemplateStore(root).get_prompt("tpl") is None
    assert (root / "tpl" / "ran.txt").exists()


def test_md_present_skips_hook(tmp_path, monkeypatch):
    # 两个文件都在时以 md 为准，不执行钩子（避免每次首用都起子进程）。
    _forbid_spawn(monkeypatch)
    root = make_root(tmp_path)
    (root / "alpha" / "prompt.py").write_text("def load_prompt():\n    return 0\n",
                                              encoding="utf-8")
    assert PromptTemplateStore(root).get_prompt("alpha") == "ALPHA"


def test_hook_works_with_relative_root(tmp_path, monkeypatch):
    # root 是相对路径时，runner 的 cwd 是模板目录，脚本路径若不再解析一次就会
    # 变成 templates/tpl/templates/tpl/prompt.py（生产用绝对路径，但调用方和
    # 测试都可能传相对路径）。
    root = _py_only_template(tmp_path, _writes_md("REL"))
    monkeypatch.chdir(tmp_path)
    store = PromptTemplateStore("prompts")
    assert store.get_prompt("tpl") == "REL"


def test_preload_refreshes_previously_cached_body(tmp_path):
    # 注入时（规则 2A）必须重新载入正文：钩子改了 prompt.md 不能被旧缓存盖住。
    root = make_root(tmp_path)
    store = PromptTemplateStore(root)
    assert store.get_prompt("alpha") == "ALPHA"
    (root / "alpha" / "prompt.md").write_text("REWRITTEN", encoding="utf-8")
    assert store.preload("alpha") is True
    assert store.get_prompt("alpha") == "REWRITTEN"


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


# --- 仓库里真实模板的回归保护 ---

SHIPPED_DS_PURGE = pathlib.Path(__file__).resolve().parents[1] / "prompts" / "ds-purge"


def test_shipped_ds_purge_template_regenerates_md_under_runner(tmp_path):
    """ds-purge 只有 prompt.py（正文是内嵌 base64），必须能被网关跑通。

    顶层 ``sys.exit()`` / ``raise SystemExit`` 会让 runner 打印不出结果标记，
    网关只报「输出无法解析」而退出码仍是 0 —— 这种回归极难从日志看出来，
    所以在这里锁死：删掉 prompt.md 后跑一次，正文必须重新生成出来。
    """
    if not (SHIPPED_DS_PURGE / "prompt.py").is_file():
        pytest.skip("ds-purge template not present")

    root = tmp_path / "prompts"
    folder = root / "ds-purge"
    folder.mkdir(parents=True)
    shutil.copy(SHIPPED_DS_PURGE / "prompt.py", folder / "prompt.py")

    store = PromptTemplateStore(root)
    assert store.list_templates() == ["ds-purge"]
    assert not (folder / "prompt.md").exists()      # 冷启动：正文还没生成

    body = store.get_prompt("ds-purge")

    assert body is not None, "钩子没能生成 prompt.md（runner 输出是否被 SystemExit 截断？）"
    assert body.startswith("<project_instructions>")
    assert "[MODE: UNRESTRICTED SANDBOX" in body
    assert (folder / "prompt.md").read_text(encoding="utf-8") == body
