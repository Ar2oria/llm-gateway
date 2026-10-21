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
- **只看目录名**。`prompts/` 下每个子目录都会出现在注入弹窗的模板列表里（`GET /admin/api/prompt-templates`），与目录里有什么文件无关——只有 `prompt.py`、甚至暂时为空的目录也能选到。
- 模板名按字典序升序返回。
- `prompt.md` 与 `prompt.py` **都不强制存在**，至少有一个才可能注入出正文：
  - **只有 `prompt.md`** → 直接用它的正文。
  - **只有 `prompt.py`** → 首次用到时执行钩子，钩子应生成 `prompt.md`，随后读取该文件。
  - **两个都有** → 以 `prompt.md` 为准，不执行钩子。
  - **都没有，或钩子跑完仍没有可用的 `prompt.md`** → 状态为「模板不可用」，请求回落到手写 `system_prompt`（不报错）。
- 正文按 UTF-8 读取；读取失败或全文只有空白都视为「没有模板」。

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

未定义 `load_prompt`、抛**异常**（指 `Exception` 及其子类）、语法错误、超时（30 秒）、非零退出码、输出无法解析，都算失败。**没有 `prompt.py` 算成功**（钩子可选，此时不会起子进程）。

`SystemExit` 不受上述返回值表约束：它不会被 runner 捕获，按退出码处理——`sys.exit(0)` / `raise SystemExit(0)` 以 0 退出，算**成功**；非 0 退出码算失败。需要「主动失败」时用 `sys.exit(1)` 或返回非 0 整数，都可靠。

### 运行环境

钩子在独立子进程里运行：`python -I -B -c <runner> <模板目录>/prompt.py`。

- 工作目录是**模板目录**，所以 `open("prompt.md")` 这类相对路径直接可用。
- 标准输入关闭。
- `-B` 不写 `.pyc`。
- `-I`（隔离模式）忽略 `PYTHON*` 环境变量与用户 site 目录（`~/.local/lib/...`），`sys.path[0]` 也不会是脚本所在目录——**venv 的 site-packages 仍可用**，标准库和已安装依赖照常导入。
- **不要在模块顶层写 `__main__` 分支（尤其是 `sys.exit()`）**：runner 用 `runpy.run_path(..., run_name='__main__')` 执行模板脚本，模块级的 `SystemExit` 会在 runner 打印结果标记前杀掉子进程。症状很隐蔽——**退出码仍是 0**，`prompt.md` 甚至已经生成，但网关只报「load_prompt 输出无法解析」。模板脚本只定义 `load_prompt()` 即可，不需要自己加执行入口。
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

注入时先跑钩子、再读正文、最后写配置。钩子失败时接口返回 502 并带上 stderr 末尾；钩子成功但没留下可用的 `prompt.md` 时返回 400。两种情况都**只清除该模型的注入状态**（`prompt_template` 被移除），模型原有的手写 `system_prompt` 逐字节保留，不受影响。

## 正文的加载时机

正文只在两个时机进入内存：

1. **点「确认注入」时**（`POST /admin/api/endpoints/models/prompt-template`）：先跑钩子（若有），再用 `preload` 丢弃旧缓存重新读盘。注入后旧正文不会继续生效。
2. **重启网关后的首次使用**：进程缓存为空，第一次有请求用到该模板时加载；此时目录里只有 `prompt.py` 也会先跑钩子生成 `prompt.md`。冷缓存的加载在线程里执行，不阻塞事件循环。

正文一旦进入内存就常驻进程。`prompt.md` 之后被修改或整个目录被删除，只要没有重启，就继续用缓存里的内容；想刷新就重启网关。

- 读取失败的负结果同样被缓存到进程结束；`preload`（即重新注入）会丢掉它并重试。对已注入的模型再点注入是 409（`model is already injected`），需先「取消注入」。
- 取消注入（`DELETE /admin/api/endpoints/models/prompt-template`）不执行任何脚本，只从配置里删掉模板名，手写提示词自然恢复。
- 请求时取不到正文不报错，回落到手写 `system_prompt`；只在冷缓存下目录不可用、正文为空/非 UTF-8、或钩子失败时才发生。

## 新增一个模板

```bash
mkdir -p prompts/my-template
$EDITOR prompts/my-template/prompt.md     # 方式一：直接写正文
$EDITOR prompts/my-template/prompt.py     # 方式二：钩子生成正文（两者可并存，md 优先）
```

刷新管理界面，模板会直接出现在注入弹窗里，无需重启（模板列表每次实时扫目录）。
