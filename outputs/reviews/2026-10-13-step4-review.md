# Code Review — B-MVP Step 4（cleanup / chat / cli）+ 相关文件

> 2026-10-13 · 审查范围：`cli.py`、`cleanup.py`、`chat.py`，以及它们调用到的 `model_providers/`、`ingest/`、`config.py`
> 工具：`ruff`（F/B/UP/SIM 规则，22 处报错）、`import-linter`（3/3 kept）、人工逐行审读。
> 严重度：🔴 运行即失败 · 🟠 行为错误/隐藏 bug · 🟡 设计与工程标准 · 🟢 风格

## 总评

**结构是对的**：分层约束全部通过，Pydantic、`match`、原子写、`.replace()` 代替 `.format()`、chat 的回滚思路都用得到位。
问题集中在两类：
1. **`cli.py` 还是草稿状态**，没有 import、`session_dir` 未定义、有两个分支未接通——跑不起来。
2. **异常处理的"包装"在跨模块时互相打架**：下层把具体异常统一包成 `RuntimeError`，上层就再也认不出它是什么了。这是本次最值得学的一课（§2）。

另一个系统性观察：C# 编译器会拦下的错误（未定义名称、未 import），Python 要到**运行到那一行**才报。
22 处里有 20 处是这一类。§5 的建议能把它们变回"编辑时就看到红线"。

---

## 1. 🔴 运行即失败

| # | 位置 | 问题 | 修法 |
|---|---|---|---|
| 1 | `cli.py` 顶部 | 没有 import：`sys`、`getpass`、`load_dotenv`、`load_settings`、`ProviderSettings`、`FileCredentialStore`、`create_provider`、`ProviderError`、`MissingCredentialError`、`ingest_file`、`UnsupportedFileError`、`FfmpegNotFoundError`、`clean_transcript`、`ChatSession` | 见 §4 参考结构 |
| 2 | `cli.py:110` | `session_dir` 从未定义；`_make_session_dir` 写了但没调用 | 见 §4 |
| 3 | `cli.py:125` | `最外层异常边界` 前面少了 `#`。**不是 SyntaxError**——Python 3 允许中文做变量名，所以它被当成一个表达式，运行到这里抛 `NameError` | 加 `#` |
| 4 | `cli.py:110` | 不给路径参数时 `args.source` 是 `None`，`_read_path_interactively()` 没被调用 → `None.is_file()` 崩 | `path = args.source or _read_path_interactively()` |
| 5 | `cleanup.py:12` | `from coachagent.model_providers import BadRequestError`——该包没有导出这个名字 → **import cleanup 时就 `ImportError`** | 删掉，见 §2 |
| 6 | `model_providers/__init__.py` | `__all__` 里写了 `"MissingCredentialsError"`：①拼写多了个 s（类名是 `MissingCredentialError`）②没有对应的 import。`__all__` 只是"声明"，不会帮你 import | `from .credentials import MissingCredentialError`，并修正 `__all__` 里的拼写 |

## 2. 🟠 行为错误（这一节最重要）

### 2.1 异常被层层包装，导致 REPL 的保护失效
调用链：`cli` REPL → `ChatSession.ask` → `OpenAIProvider.chat`

- `OpenAIProvider.chat` 把 `openai.APIError` 转成 `ProviderError` ✅（adapter 该做的事）
- `ChatSession.ask` 又把**所有**异常包成 `RuntimeError` ❌
- `cli` 的 REPL 写的是 `except ProviderError` → **永远匹配不上**

后果：问答中任何一次网络抖动、限流，整个程序崩溃，对话历史全丢。你本来设计的"出错不退出 REPL"实际不生效。

**原则**：只在**跨越抽象边界**时转换异常类型（SDK 异常 → 我们的 `ProviderError`，这是 adapter 的职责），
同一层内部不要再包装。只想做清理（回滚）时，用 bare `raise` 原样往上抛：

```python
# chat.py —— ≈ C#: catch { rollback; throw; }   （注意是 throw; 不是 throw ex;）
self._messages.append(Message(role="user", content=question))
try:
    result = self._provider.chat(self._messages)
except BaseException:          # 连 Ctrl+C 中断请求也回滚
    self._messages.pop()
    raise                      # 原样重抛，类型、traceback 都不变
self._messages.append(Message(role="assistant", content=result.text))
return result.text
```
`try/except/else` 也行；关键是 `raise` 不带参数。

### 2.2 `clean_transcript` 可能静默返回 `None`
```python
except BadRequestError as e:
    if "temperature" in str(e):
        return provider.ask(...)
    # ← 条件不成立时：没有 return、没有 raise → 函数返回 None
```
签名写的是 `-> str`，实际可能返回 `None`，下游 `write_text(None)` 才崩，离出错点很远。
**规则**：`except` 里处理不了的，必须 `raise`。

而且这个分支**永远走不到**：`OpenAIProvider` 已经把 `BadRequestError`（`APIError` 的子类）转成了 `ProviderError`。

**修法**：temperature 兼容性是**某个 provider 的怪癖**，属于 adapter 的职责，不该泄漏到 cleanup（应用层）。
`cleanup.py` 删掉整个 try/except，就一行：

```python
return provider.ask(transcript.to_markdown(), system=CLEANUP_SYSTEM_PROMPT, temperature=0.1)
```
让 `ProviderError` 原样传到 cli 处理。在 `openai_provider.py` 里改：

```python
kwargs = {"model": self.model, "input": user_and_assistant}
if instructions:
    kwargs["instructions"] = instructions
if temperature is not None:
    kwargs["temperature"] = temperature     # None 时不传 —— SDK 默认值是 omit（不发送），
resp = self._client.responses.create(**kwargs)  # 你现在显式传 None 会发送 "temperature": null
```
（真遇到拒绝 temperature 的模型，再在这里 catch `BadRequestError` → 去掉 temperature 重试一次；仍然只在 adapter 里。）

### 2.3 其他
| # | 位置 | 问题 | 修法 |
|---|---|---|---|
| a | `config.py:68` | `except pydantic.ValidationError` 但没 `import pydantic` → 平时不触发，**只有 settings.json 写坏时**才 `NameError`，把真正的错误盖掉。典型"潜伏 bug"，测试/静态检查才抓得到 | `from pydantic import ValidationError` |
| b | `cli.py` | `--from-json`、`--whisper-model` 定义了但没接通；`settings.whisper_model` 也没传给 `WhisperLocalTranscriber` | 见 §4 |
| c | `cli.py` | 首次运行不写 `settings.json`——Q4 "确认 default" 的落地点缺失 | `if not SETTINGS_FILE.exists(): save_settings(settings); print(...)` |
| d | `cli.py` | **Q6 防泄露检查（`git check-ignore`）没实现**。会议数据相关，这条是 guardrail，不是 nice-to-have | 见 §4 `_ensure_ignored` |
| e | `cli.py` REPL | `input()` 遇到 Ctrl+Z/Ctrl+D 抛 `EOFError` → 崩 | `except (EOFError, KeyboardInterrupt): break` |
| f | `ingest/service.py` | `on_progress("extracting audio…")` 写在 `extract_audio()` **之后**，进度提示晚了一步 | 挪到调用之前 |
| g | `ingest/models.py` | `text` property 里 `self.segments.sort(...)` 会**修改对象本身**——getter 带副作用 | 改用 `sorted(...)`，和 `to_markdown` 一致 |
| h | `cleanup.py` prompt | 三引号字符串每行带 4 空格缩进，原样发给模型；且没说"只输出正文" → 模型常会加"以下是整理后的内容："前言写进 cleaned.md | 去掉缩进（或 `textwrap.dedent`）；加一条"只输出整理后的记录，不要任何前言或说明" |

## 3. 🟡 设计与工程标准

1. **异常边界放错位置**：现在的 try/except 写在 `if __name__ == "__main__":` 下。将来用 `[project.scripts]` 做 `coach` 命令时，入口直接调 `main()`，这层保护就被绕过。
   工程惯例：`def main(argv: list[str] | None = None) -> int`，返回退出码，异常边界在 `main` 内部；底部只写 `raise SystemExit(main())`。
   `argv` 参数让测试可以直接 `main(["x.mp4", "--model", "foo"])`，不用伪造命令行。
2. **`source` 和 `--from-json` 互斥**，应该让 argparse 来保证（`add_mutually_exclusive_group`），而不是运行时自己判断。
3. **argparse 没有 `description` / `help`**：`--help` 是 CLI 的文档，写上几乎零成本。
4. **`main()` 太长**（参数 + 组装 + 三个步骤 + REPL）。拆成 `_build_parser()`、`_build_provider(args, settings)`、`_repl(session)`，`main` 只剩编排——和 `ingest_file` 是同一个思路。
5. **函数内 import 用多了**：`service.py`、`openai_provider.py` 里对**本包模块**（`.detect`、`.base`）也做了函数内 import。只有重量级的第三方包（`whisper`/torch、`openai`）才值得 lazy；本地模块放顶部，读者一眼看到依赖。
6. **复合语句写在一行**：`if q in {...}: break`、`try: print(...)` —— PEP 8 不推荐，`ruff` 默认规则也会报 E701/E702。

## 4. `cli.py` 参考结构（重点讲 argparse）

argparse 的心智模型 ≈ C# 的 `System.CommandLine`：**先声明参数 → `parse_args()` → 得到一个对象**，每个参数是它的一个属性（`--from-json` 自动变成 `args.from_json`，横线转下划线）。

```python
import argparse
import subprocess
import sys
from getpass import getpass
from pathlib import Path

from dotenv import load_dotenv

from coachagent.chat import ChatSession
from coachagent.cleanup import clean_transcript
from coachagent.config import SETTINGS_FILE, FileCredentialStore, ProviderSettings, Settings, load_settings, save_settings
from coachagent.ingest import FfmpegNotFoundError, Transcript, UnsupportedFileError, WhisperLocalTranscriber, ingest_file
from coachagent.model_providers import LLMProvider, MissingCredentialError, ProviderError, create_provider


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="coachagent",
        description="Transcribe a meeting recording, clean it with an LLM, then ask questions about it.",
    )
    src = p.add_mutually_exclusive_group()          # 二选一：要么给媒体文件，要么给已有 json
    src.add_argument("source", nargs="?", type=Path, help="audio/video file (prompted if omitted)")
    src.add_argument("--from-json", type=Path, metavar="JSON", help="reuse a raw.transcript.json, skip Whisper")
    p.add_argument("--provider", help="LLM provider (default: settings.json)")
    p.add_argument("--model", help="override model name")
    p.add_argument("--whisper-model", help="tiny/base/small/medium (default: settings.json)")
    return p
#   nargs="?"  → 位置参数可省略（省略时值为 None）
#   type=Path  → argparse 自动把字符串转成 Path
#   没写 default → 默认 None；"None 表示用户没指定，交给下一级配置决定"，正好配合 §3 的优先级链


def _ensure_ignored(path: Path) -> None:
    """Q6 guardrail：产出目录若在 git repo 内却没被 ignore，拒绝继续。"""
    r = subprocess.run(["git", "check-ignore", "-q", str(path)], capture_output=True)
    # returncode: 0 = 被 ignore ✓ ；1 = 在 repo 内但没被 ignore ✗ ；128 = 不在 git repo（放行）
    if r.returncode == 1:
        raise RuntimeError(f"refusing to write meeting data to non-ignored path: {path}")


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)       # argv=None → 读 sys.argv[1:]
    try:
        load_dotenv()
        settings = load_settings()
        if not SETTINGS_FILE.exists():
            save_settings(settings)
            print(f"wrote default settings: {SETTINGS_FILE}")

        provider = _build_provider(args, settings)   # 先建：缺 key 立刻失败
        print(f"provider: {provider.name} / {provider.model}")

        # --- 拿到 transcript：两条路 ---
        if args.from_json:
            transcript = Transcript.model_validate_json(args.from_json.read_text(encoding="utf-8"))
            session_dir = args.from_json.parent
        else:
            path = args.source or _read_path_interactively()
            session_dir = _make_session_dir(path, settings.sessions_dir or Path("coach/sessions"))
            _ensure_ignored(session_dir)
            transcriber = WhisperLocalTranscriber(args.whisper_model or settings.whisper_model)
            transcript = ingest_file(path, work_dir=session_dir, transcriber=transcriber, on_progress=print)

        cleaned = clean_transcript(transcript, provider)
        (session_dir / "cleaned.md").write_text(cleaned, encoding="utf-8")
        print(f"✓ {session_dir}")

        _repl(ChatSession(provider, context=cleaned))
        return 0

    except (UnsupportedFileError, FileNotFoundError, FfmpegNotFoundError,
            MissingCredentialError, ProviderError) as e:
        print(f"[error] {e}", file=sys.stderr)     # 错误信息走 stderr
        return 1
    except KeyboardInterrupt:
        print("\nbye")
        return 130                                 # Unix 惯例：128 + SIGINT(2)
    # 其他异常不 catch —— 开发期要看到完整 traceback


def _repl(session: ChatSession) -> None:
    while True:
        try:
            q = input("问> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q in {"", "exit", "quit"}:
            break
        try:
            print(session.ask(q))
        except ProviderError as e:                 # 依赖 §2.1 修好之后才生效
            print(f"[error] {e}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
```
`_make_session_dir`、`_read_path_interactively` 你已经写好了。`_build_provider(args, settings)` 就是把你现在 99–107 行
（`name = ...` 到 `create_provider(...)`）原样抽成一个函数。（argparse 部分已在 sandbox 实测：互斥、省略、覆盖都符合预期。）

**argparse 自测**（不跑 Whisper、不调 API，只看参数解析）：
```powershell
uv run python -m coachagent.cli --help
uv run python -m coachagent.cli a.mp4 --from-json b.json     # 应报 "not allowed with argument"
```

## 5. 🟢 让 Python 也有"编译器红线"（建议）

本次 22 处里 20 处是"未定义名称"——C# 编译不过的那种。Python 里两样工具补上：

| 工具 | 作用 | 怎么用 |
|---|---|---|
| **Ruff**（dev 依赖 + VS Code 扩展 `charliermarsh.ruff`） | 秒级 lint：未定义名、未用 import、`raise` 遗漏… | `uv add --dev ruff`；`uv run ruff check coachagent` |
| **Pylance 类型检查** | 编辑器里实时红线（≈ C# 的编译错误提示） | VS Code 设置 `"python.analysis.typeCheckingMode": "standard"` |

建议的提交前三连：`uv run ruff check coachagent` → `uv run lint-imports` → 跑 §5.1 的冒烟命令（smoke test）。

## 6. 修复清单（按顺序）

- [ ] §1 全部 6 条（之后 `ruff check` 应为 0 error）
- [ ] §2.1 chat.py 改 bare `raise`
- [ ] §2.2 cleanup.py 删 try/except；openai_provider.py 改 kwargs
- [ ] §2.3 a–h
- [ ] cli.py 按 §4 重构（`main() -> int`、互斥组、`_ensure_ignored`、`--from-json`、`--whisper-model`）
- [ ] `uv run python -m coachagent.cli --help` 能看到完整帮助
- [ ] 端到端：mp4 → cleaned.md → 问"有几个不同的人"
