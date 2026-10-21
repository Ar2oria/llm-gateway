import pathlib

import pytest

from gateway.prompt_templates import PromptTemplateStore


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


def test_preload_reports_whether_prompt_is_available(tmp_path):
    root = make_root(tmp_path)
    store = PromptTemplateStore(root)
    assert store.preload("alpha") is True
    assert store.preload("no-md") is False
