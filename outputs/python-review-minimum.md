# Review Agent 代码的 Python 最小范围（基于 coachagent/ 现状，2026-10-13）

> 目标：能 review 本 repo 这类 LLM/Agent 代码，不是"学完 Python"。
> 每一项都指向 `coachagent/` 里真实出现的代码；没出现的不列（原则 2）。
> 读者背景：.NET 10 年，Python 停在 3.7。C# 对照只在有帮助时给。

---

## 0. 一句话结论

当前代码里 review 真正需要的只有 **4 块，约 25 个点**：
**(A) 3.8→3.13 的新语法**、**(B) Python 特有的运行时陷阱**、**(C) Pydantic v2 + typing 的建模方式**、**(D) LLM 调用层的 review 视角**。
asyncio、decorator 原理、generator、metaclass、packaging 现在都**不在范围内**——B4 之前代码里不会出现。

---

## A. 新语法（3.8 → 3.13，你缺的那 4 年）

| # | 知识点 | repo 出处 | review 时看什么 |
|---|---|---|---|
| A1 | `X \| None` 代替 `Optional[X]`；`list[str]` / `dict[str, T]` 内置泛型 | 到处 | 只要看懂即可；`from __future__ import annotations` 让注解延迟求值（3.13 下仍常见） |
| A2 | walrus `:=` | `credentials.py::resolve_api_key` | 只在"赋值+判断"合一时用；`if key := ...` 后 key 在外层作用域仍可见 |
| A3 | `match` / `case`，含 guard `case x if ...` 和 `case _` | `detect.py`、`ingest/service.py` | 有没有 `case _` 兜底；guard 写法是否比 if/elif 更清楚（不一定） |
| A4 | keyword-only 参数 `def f(a, *, b)` | `ingest_file`、`create_provider`、`ask` | 公共 API 的可选参数应放在 `*` 后面，防止位置传参误用 |
| A5 | `StrEnum`（3.11+） | `ingest/models.py::MediaKind` | 成员即 str，JSON 序列化免转换 |
| A6 | 类型别名 `X = Callable[[str], None]`（3.12 可写 `type X = ...`） | `ProgressFn`、`PromptFn` | 回调签名是否用别名说清楚 |
| A7 | `collections.abc.Callable` 而非 `typing.Callable` | `service.py`、`credentials.py` | 风格项，不阻塞 |

## B. Python 运行时陷阱（.NET 背景最容易漏）

| # | 知识点 | repo 出处 | review 时看什么 |
|---|---|---|---|
| B1 | **名字在运行时才解析**：没 import 的名字只有执行到那行才 `NameError` | `config.py::load_settings` 的 `except pydantic.ValidationError` | `except` 子句里的名字也一样——只有异常真发生时才炸。**没有编译器帮你**，要靠 pyright/ruff 或测试 |
| B2 | `__init__.py` re-export + `__all__` | `ingest/__init__.py`、`model_providers/__init__.py` | `__all__` 里每个名字必须真的被 import；调用方 `from pkg import X` 的 X 必须在 `__init__` 里 |
| B3 | 函数没走到 `return` → **隐式返回 `None`** | `cleanup.py::clean_transcript` | 每条 `except` 分支要么 return 要么 raise；类型注解 `-> str` 运行时不检查 |
| B4 | `raise X from e` 异常链 ≈ InnerException | `openai_provider.py`、`chat.py` | 包装异常时**别丢类型**：`except Exception → RuntimeError` 会让上层 `except ProviderError` 失效 |
| B5 | 可变默认值：`= []` 错，`Field(default_factory=list)` 对 | `ingest/models.py` | 函数参数 / Pydantic / dataclass 都适用 |
| B6 | `@property` 不应有副作用；`list.sort()` 原地改 vs `sorted()` 返回新列表 | `Transcript.text` | getter 里改状态 = 读一次属性就改了数据 |
| B7 | 函数内 lazy import | `transcribers.py`（whisper）、`registry.py`（openai） | 只对**重**依赖（torch/SDK）有意义；stdlib 和本包模块应放顶部 |
| B8 | 文件 IO 必写 `encoding="utf-8"` | 所有 `write_text` / `read_text` | Windows 默认 cp1252/GBK，中文会炸 |
| B9 | 原子写：写 `.tmp` 再 `Path.replace()` | `config.py` | 配置/凭据写盘是否会写一半 |

## C. 建模：Pydantic v2 + typing

| # | 知识点 | repo 出处 | review 时看什么 |
|---|---|---|---|
| C1 | `BaseModel` 字段即校验；`model_validate` / `model_validate_json` / `model_dump` / `model_dump_json` | `models.py`、`config.py`、`openai_provider.py` | v1 的 `.dict()` `.parse_obj()` 已废弃；外部输入（文件、LLM 输出、whisper dict）进系统必须过 `model_validate` |
| C2 | 默认忽略多余字段（`extra="ignore"`） | `TranscriptSegment.model_validate(seg)` | 想严格就 `model_config = ConfigDict(extra="forbid")`——**B5 让 LLM 填 schema 时会用到** |
| C3 | `Literal["a","b"]` 运行时由 Pydantic 校验 | `base.py::Role` | B3 加 tool calling 时 `"tool"` 角色要扩进来 |
| C4 | `@dataclass(frozen=True)` vs `BaseModel` | `ProviderSpec` vs `Message` | 代码内常量用 dataclass；跨边界的数据用 Pydantic |
| C5 | `Protocol`（结构化）vs `ABC`（名义继承） | `Transcriber`/`CredentialStore` vs `LLMProvider` | Protocol 运行时**不检查**，签名不符只有类型检查器能发现；ABC 实例化时就报 `TypeError`。要共享实现（`ask()`）才选 ABC |
| C6 | 依赖注入靠参数：`transcriber=None`、`store=`、`prompt=`、`on_progress=` | `service.py`、`registry.py` | 库代码里不 `print`/`input`/`load_dotenv`，副作用全交给 cli（Composition Root） |

## D. LLM / Agent 调用层的 review 视角（这是 AI 特有的部分）

| # | 看什么 | repo 出处 |
|---|---|---|
| D1 | **Provider 抽象**：统一 `Message` → adapter 翻译 → 统一 `ChatResult`；上层不 import SDK | `base.py`、`openai_provider.py`，由 import-linter 契约守住 |
| D2 | **无状态 provider + 有状态 session**；失败时回滚刚 append 的 user 消息 | `chat.py::ChatSession.ask` |
| D3 | **Prompt 结构**：指令与材料分离（`<transcript>` XML 标签）；用户内容含 `{}` 时别用 `str.format` | `chat.py`、`cleanup.py` |
| D4 | **模型参数兼容性**：reasoning 模型拒绝 `temperature`；`None` 时最好**不传**而非传 null | `openai_provider.py`、`cleanup.py` |
| D5 | **错误分类**：auth / rate-limit / bad-request / connection 要区分（重试与否不同），统一包成自家异常但保留子类型信息 | `openai_provider.py`（目前全压成一个 `ProviderError`） |
| D6 | **Context 预算**：context stuffing + 无限增长的 history，何时会超 context window | `chat.py`（暂可接受，B7 RAG 再解决） |
| D7 | **Secrets**：key 不 print/log，只露后 4 位；解析链优先级 | `credentials.py::mask_key`、`resolve_api_key` |
| D8 | **可观测性**：usage（tokens）是否被记录 | `ChatResult.usage`（有字段，没人用） |
| D9 | **数据外发**：哪些步骤把会议文本发到云端 | cleanup / chat → OpenAI；纯本地需 Ollama `base_url` |

B3/B4 开始后再加：tool schema（JSON Schema 由 Pydantic 生成）、tool 结果喂回、`max_steps` 终止、tool 异常不应炸掉 loop。

---

## 用这份清单 review 当前代码：实测发现

在 sandbox 里 import / 调用验证过（Python 3.10 + StrEnum shim，逻辑与 3.13 一致）。

| 严重度 | 位置 | 问题 | 对应条目 |
|---|---|---|---|
| 🔴 阻塞 | `cleanup.py` | `from coachagent.model_providers import BadRequestError` → **ImportError**，模块根本 import 不了 | B2 |
| 🔴 逻辑 | `cleanup.py` | 即使导入成功也接不到：`OpenAIProvider` 已把 `openai.BadRequestError` 包成 `ProviderError`；且 "temperature" 不在消息里时该分支**返回 None** | B3, B4, D5 |
| 🔴 阻塞 | `model_providers/__init__.py` | `__all__` 写了 `"MissingCredentialsError"`（多一个 s，也没 import）→ `from ... import *` **AttributeError**；cli 也拿不到这个异常 | B2 |
| 🔴 潜伏 | `config.py::load_settings` | `except pydantic.ValidationError` 但只 import 了 `BaseModel, Field` → settings.json 一被改坏就 **NameError**，掩盖真实错误 | B1 |
| 🟡 | `chat.py::ask` | `except Exception` 包成 `RuntimeError` → cli 计划的 `except ProviderError` 抓不到，REPL 会退出 | B4 |
| 🟡 | `ingest/models.py::Transcript.text` | property 里 `self.segments.sort()` 原地改数据（已验证）；`to_markdown` 用 `sorted()`，两处不一致 | B6 |
| 🟡 | `openai_provider.py` | `temperature=None` 显式传给 SDK（会序列化为 null）；无 system 时传 `instructions=""`。建议参数为 None 就不放进 kwargs | D4 |
| 🟢 | `ingest/service.py` | `on_progress("extracting audio…")` 在抽取**之后**才调用 | — |
| 🟢 | `audio.py`、`credentials.py`、`service.py` | stdlib / 本包模块 lazy import，注释自己也说该挪顶部 | B7 |
| 🟢 | `hello.py`、`client.py` | 旧入口，库函数里 `load_dotenv()`，违反 C6；B-MVP 完成后可删 | C6 |
| 🟢 | 全局 | 没有任何测试；上面 4 个 🔴 都是"跑一次 import 就能发现"的级别 | — |

**观察**：4 个 🔴 有 3 个属于 B1/B2/B3——"静态语言里编译器替你挡掉的那类错"。这是你从 .NET 过来最该补的一个习惯：
**每改完一个模块，至少跑一次 `uv run python -c "import coachagent.xxx"`，并在 VS Code 打开 Pylance `typeCheckingMode: "basic"`**（零依赖，能当场标红 B1/B2/B3）。

---

## 明确不在最小范围（遇到再学）

asyncio（B? UI/并发时）· decorator 原理 · generator / `yield`（stream 时）· context manager 自写 · metaclass · packaging / `[build-system]` · `pydantic-settings` · pytest fixture 进阶（B4 补最小测试时再学基础）。
