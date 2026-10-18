# Python 补习清单（由 B-MVP 实际错误归纳）

> 2026-10-17。只列**你真的踩过**的点，每条指回 `coachagent/` 里出过事的位置。按优先级排序。
> 读法：每个主题先看"C# 直觉为什么会骗你"，再看官方文档对应小节（都不长）。

---

## P1 · 模块与 import 系统 —— 本次错误最多的一类

**踩过的坑**
- `cli.py`：`from config import ...`、`from registry import ...`、`from credentials import getpass`（裸模块名 / 来源搞错）
- `cleanup.py`：从 `model_providers` import 一个它没导出的 `BadRequestError`
- `model_providers/__init__.py`：`__all__` 里写了 `MissingCredentialsError`，却没 import（还拼错）

**C# 直觉为什么会骗你**
- C# 的 `using` 按 namespace 找、由编译器在**所有引用的 assembly** 里解析；Python 的 import 是**运行时按 `sys.path` 逐个目录找文件**。
  `sys.path[0]` 取决于启动方式：`python -m coachagent.cli` → repo 根；`python coachagent/cli.py` → `coachagent/` 目录。
  所以 `from config import ...` 在后一种方式下"碰巧能跑"，在前一种下失败——同一行代码，结果取决于怎么启动。
- `__all__` **只是声明**（影响 `from x import *` 和工具提示），不会替你 import 任何东西。≈ 文档，不是代码。

**要掌握**
- 三种写法：绝对 `from coachagent.ingest import X`、相对 `from .detect import X`（只在包内）、裸名 `from config import X`（**本项目禁用**）
- 包 = 有 `__init__.py` 的目录；`__init__.py` 在第一次 import 包时执行；re-export 就是在它里面 import
- `python -m pkg.mod` vs `python path/to/mod.py` 的区别
- 读：[The import system §5.1–5.4](https://docs.python.org/3/reference/import.html)、[Modules tutorial §6.1–6.4](https://docs.python.org/3/tutorial/modules.html)

## P2 · 异常处理

**踩过的坑**
- `chat.py`：把所有异常包成 `RuntimeError`，使 REPL 的 `except ProviderError` 失效（已修 ✅）
- `cleanup.py`：`except` 分支条件不成立时既不 return 也不 raise → 函数静默返回 `None`
- `cleanup.py`：catch 一个永远不会到达的异常（下层早已转换）
- 自评：新语言里容易忘记 try/except

**C# 直觉为什么会骗你**
- 规则和 C# 基本一样（`raise` ≈ `throw;`，`raise X from e` ≈ 带 `InnerException`），差别在于：**Python 函数没有"所有路径必须返回"的编译检查**——漏掉的路径默认返回 `None`（pyright 能查，见 P6）。
- `except Exception` 不包括 `KeyboardInterrupt`（它继承自 `BaseException`），这点和 C# 不同。

**要掌握**
- `try / except / else / finally` 四段各自何时执行
- bare `raise` vs `raise X from e` vs `raise X`（丢失原因）
- 在哪一层转换异常：**只在跨抽象边界时**（SDK → `ProviderError` 在 adapter 里）
- 读：[Errors and Exceptions §8.3–8.7](https://docs.python.org/3/tutorial/errors.html)

## P3 · 名字解析与作用域（"没有编译期"）

**踩过的坑**
- `cli.py`：`session_dir` 未定义；`最外层异常边界` 漏 `#` 变成一个"表达式"
- `config.py`：`except pydantic.ValidationError` 却没 import `pydantic`——**只有设置文件损坏时才炸**

**C# 直觉为什么会骗你**
- Python 只在**执行到那一行**时才查名字。没走到的 `except` 分支里写错名字，平时完全看不出来。
- 作用域是 **LEGB**（Local → Enclosing → Global → Builtin），**没有块级作用域**：`if/for/try` 里赋值的变量，块外照样可用。
- Python 3 允许 Unicode 标识符，所以中文行不是 SyntaxError。

**要掌握**
- LEGB；`global` / `nonlocal` 何时需要（极少需要——需要时通常是设计问题）
- 读：[Python Scopes and Namespaces §9.2](https://docs.python.org/3/tutorial/classes.html#python-scopes-and-namespaces)

## P4 · 函数签名与"缺省值"语义

**踩过的坑**
- `openai_provider.py`：`temperature=None` 被显式发送（SDK 的"不传"是 `omit` 哨兵值，不是 `None`）
- `cli.py`：回写 `args.source`、`settings.sessions_dir` 来表达"用默认值"（已修 ✅）

**要掌握**
- `*` 之后是 keyword-only 参数（`ingest_file(path, *, work_dir, ...)`）
- `a or b` 是 **truthiness** 判断，不是 C# 的 `??`：`""`、`0`、`[]`、`Path("")` 都会落到 `b`
- 可变默认值陷阱（`def f(x=[])`）——你在 Pydantic 里用 `Field(default_factory=list)` 已经避开了
- 构造 kwargs 字典再 `f(**kwargs)`：表达"有值才传"
- 读：[More on Defining Functions §4.9](https://docs.python.org/3/tutorial/controlflow.html#more-on-defining-functions)

## P5 · 可变性与副作用

**踩过的坑**
- `ingest/models.py`：`text` property 里 `self.segments.sort(...)`——getter 修改了对象
- 两次 "目录不存在"：`audio.py`（ffmpeg 输出）、`config.py`（auth.json）——写文件前没 `mkdir`

**要掌握**
- `list.sort()`（原地，返回 `None`） vs `sorted()`（返回新列表）——同类：`list.reverse()` vs `reversed()`
- 返回内部 list 时给副本（你在 `ChatSession.history` 做对了）
- `Path.mkdir(parents=True, exist_ok=True)` 作为"写文件前的固定动作"

## P6 · 类型系统（pyright 这次额外抓到的）

**现存问题**（当前代码，pyright standard 报错，运行时不炸）
1. `ingest/transcribers.py`：Protocol 声明 `name: str`（**可读写属性**），实现用的是 `@property`（**只读**）→ 不兼容。
   连带 `service.py` 两条报错。修法：Protocol 里也写成 property：
   ```python
   class Transcriber(Protocol):
       @property
       def name(self) -> str: ...
       def transcribe(self, audio: Path) -> Transcript: ...
   ```
2. `model_providers/openai_provider.py`：`[m.model_dump() for m in ...]` 的类型是 `list[dict[str, Any]]`，SDK 要求 `ResponseInputParam`。
   运行没问题，但类型对不上。修法（已在 sandbox 用 pyright 验证为 0 error）：
   ```python
   from typing import TYPE_CHECKING
   if TYPE_CHECKING:                                   # 只给类型检查器看，运行时不 import → 不破坏 lazy import
       from openai.types.responses import ResponseInputParam
   ...
   user_and_assistant: ResponseInputParam = [
       {"role": m.role, "content": m.content} for m in messages if m.role != "system"
   ]
   ```

**要掌握**
- Protocol 里"属性" vs "property" 的区别
- `X | None` 与 **narrowing**（`if x is None: ...` 之后类型检查器知道 x 不是 None）
- 泛型 **invariance**：`list[子类]` 不能赋给 `list[父类]`（和 C# 的 `List<T>` 一样；`Sequence` 才是协变的，≈ `IReadOnlyList<out T>`）
- `TypedDict`：OpenAI SDK 的参数类型大量是它（≈ 只有形状、没有行为的 DTO）
- `TYPE_CHECKING`
- 读：[typing — Protocols](https://typing.python.org/en/latest/spec/protocol.html)、[mypy cheat sheet](https://mypy.readthedocs.io/en/stable/cheat_sheet_py3.html)（规则与 pyright 基本一致）

---

## 附：哪个工具能抓哪类错（实测）

在 sandbox 里把你这次的错误重新注入，用 ruff / pyright standard / import-linter 分别跑：

| 错误 | ruff | pyright standard | import-linter | 只有运行时 |
|---|:-:|:-:|:-:|:-:|
| `session_dir` 未定义、中文漏 `#` | ✅ | ✅ | | |
| 调用少参数（`ingest_file(path)` 漏 `work_dir`） | | ✅ | | |
| except 分支漏 return（隐式 None） | | ✅ | | |
| import 不存在的名字（`BadRequestError`） | | ✅ | | |
| `__all__` 写了不存在的名字 | | ✅（warning） | | |
| `from registry import`（找不到模块） | | ✅ | | |
| **`from config import`（裸名，但同目录恰有 config.py）** | | ❌ | ❌ | ✅ |
| 跨层 import | | | ✅ | |

最后一行是三者的共同盲区：pyright 会**按导入文件所在目录**解析裸名，于是 `coachagent/config.py` 被"找到"了；但 `python -m` 运行时找不到。
最便宜的补丁是一个 import 冒烟测试（smoke test）：`uv run python -c "import coachagent.cli"`，或写成 pytest 的一条测试。
