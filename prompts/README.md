# 提示词模板

本目录存放可注入的提示词模板。在管理界面展开接入点、点模型行的「注入提示词」即可选用；模板名会写进 `config.json` 的 `endpoints[].models[].prompt_template`，请求时按名取正文替换该模型的系统提示词。

```
prompts/
├── README.md
└── system-prompt/
    ├── prompt.md          # 必填：模板正文
    └── prompt.py          # 可选：注入前的钩子
```

## 目录约定

- **目录名即模板名**。名字必须是单个路径段：不能为空，不能是 `.`/`..`，不能包含 `/` 或 `\`。
- **`prompt.md` 必填**。只有含 `prompt.md` 的子目录才会出现在注入弹窗的模板列表里；缺少它的目录直接不列出（`GET /admin/api/prompt-templates` 也不返回）。
- 模板名按字典序升序返回。
- 正文按 UTF-8 读取；读取失败或全文只有空白都视为「没有模板」，此时回落到该模型手写的 `system_prompt`，不会写入空的系统提示词。

## `prompt.py`（可选钩子）

存在时必须定义 `load_prompt()`，在注入时执行一次，用来做生成或校验正文之类的准备工作：

```python
def load_prompt():
    # 在这里生成/校验 prompt.md
    return 0
```

返回值约定：

| 返回 | 结果 |
|------|------|
| `0` | 成功 |
| 非 0 整数 | 失败 |
| 非 `int`（含 `None`） | 成功 |
| `True` / `False` | 成功（`bool` 是 `int` 特例，但不当作非 0 失败） |

未定义 `load_prompt`、抛异常、语法错误、超时（30 秒）、非零退出码、输出无法解析，都算失败。**没有 `prompt.py` 算成功**（钩子可选，此时不会起子进程）。

### 运行环境

钩子在独立子进程里运行：`python -I -B -c <runner> <模板目录>/prompt.py`。

- 工作目录是**模板目录**，所以 `open("prompt.md")` 这类相对路径直接可用。
- 标准输入关闭。
- `-B` 不写 `.pyc`。
- `-I`（隔离模式）忽略 `PYTHON*` 环境变量与用户 site 目录（`~/.local/lib/...`），`sys.path[0]` 也不会是脚本所在目录——**venv 的 site-packages 仍可用**，标准库和已安装依赖照常导入。
- 因为工作目录和脚本目录都不在 `sys.path` 上，导入同目录的模块需要自行插入路径：

```python
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import my_helper  # 同目录模块


def load_prompt():
    my_helper.build()
    return 0
```

### 失败后的状态

注入时先跑钩子、后写配置。钩子失败时接口返回 502 并带上 stderr 末尾；系统**只清除该模型的注入状态**（`prompt_template` 被移除），模型原有的手写 `system_prompt` 逐字节保留，不受影响。

## 正文的加载时机

- 注入成功时会把正文预读进内存（`preload`），正常路径下第一次真正读盘就发生在注入那一刻。
- 之后 `prompt.md` 常驻进程内存；**修改文件内容需要重启网关才生效**。
- 读取失败的负结果同样会被缓存：模板目录缺失、正文为空时该模板在进程内一直视为不可用，重启后才会重新读取（重新注入会因为 `preload` 重试一次）。
- 取消注入（`DELETE /admin/api/endpoints/models/prompt-template`）不执行任何脚本，只从配置里删掉模板名，手写提示词自然恢复。
- 请求时取不到正文（目录已删、正文变空、非 UTF-8）不报错：回落到手写 `system_prompt`。

## 新增一个模板

```bash
mkdir -p prompts/my-template
$EDITOR prompts/my-template/prompt.md     # 必填
$EDITOR prompts/my-template/prompt.py     # 可选
```

刷新管理界面，模板会直接出现在注入弹窗里，无需重启（模板列表每次实时扫目录）。
