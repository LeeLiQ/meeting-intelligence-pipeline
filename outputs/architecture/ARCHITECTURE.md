# coachagent 架构图（v2，2026-10-16）

> 旧版（3 张 Mermaid）备份在 `_scratch/ARCHITECTURE.md.20261016-150043.bak`。
> 新版用 **Archify** 生成：可交互的单文件 HTML（缩放、点节点高亮路径、暗/亮主题、导出 PNG/SVG）。
> 双击 HTML 用浏览器打开即可，不需要任何扩展。

## 按这个顺序读（≈ 读一个 ASP.NET 项目的顺序）

| # | 文件 | 回答的问题 | C# 对照 |
|---|---|---|---|
| 1 | `01-overview.html` | 入口在哪？它按什么顺序调用了谁？ | 打开 `Program.cs` |
| 2 | `03-runtime.html` | 一次运行里，调用和返回值怎么来回？ | 在 `Program.cs` 打断点单步 |
| 3 | `02-packages.html` | 每个包对外暴露什么、内部怎么分文件？ | 看每个 `.csproj` 的 public 类型 |

旧图难读的根因：**把三个层级画在了同一张图里**（模块依赖 + 所有 class/函数 + Protocol 线），
等于把 Solution Explorer 全部展开再加 Class View。新版每张图只回答一个问题，并且永远从 `cli.main()` 读起。

## 读 Python 代码时的心智映射

| Python | 当成 C# 的什么 |
|---|---|
| `cli.main()` + `if __name__ == "__main__"` | `Program.cs` / `Main()`（Composition Root，手工 new + 传参，没有 DI 容器） |
| 包（有 `__init__.py` 的目录） | 一个 project / assembly |
| `__init__.py` 里的 import + `__all__` | 该 assembly 的 `public` 类型 |
| 只有函数、没有 class 的 `.py` | `static class` |
| `Protocol` | `interface`，但**不需要写 `: IFoo`**，签名对上就算实现（图里虚线） |
| `ABC` + `@abstractmethod` | `abstract class` |
| Pydantic `BaseModel` / `@dataclass(frozen=True)` | DTO / `record` |

## 修改 / 重新生成

JSON 源在 `src/`。改 JSON 后在本机重渲染（需 Node ≥ 18 + 本机 Chrome；archify 装好后路径见下）：

```powershell
node <archify>\bin\archify.mjs finalize architecture outputs\architecture\src\01-overview.architecture.json outputs\architecture\01-overview.html --quality showcase --json
node <archify>\bin\archify.mjs finalize sequence     outputs\architecture\src\03-runtime.sequence.json     outputs\architecture\03-runtime.html     --quality showcase --json
```

或者直接在 Claude Code / Cowork 里说"用 archify 更新 01-overview，加上 B3 的 tools/"。

**注意**：图是手工（agent）对照 2026-10-16 的代码写的 JSON，不是从代码自动解析。Archify 的校验只保证"图没画坏"（布局、标签、连线），
**不保证图和代码一致**。代码改了图不会自动跟着变。

## 自动生成（随代码更新，对照用）

需要"代码此刻真实长什么样"时用 pyreverse（pylint 自带），`uvx` 临时拉起，不进项目依赖：

```powershell
uvx --from pylint pyreverse -o mmd -p coachagent -d outputs\architecture --ignore=transcribe.py,hello.py,client.py coachagent
```

它看不见：Protocol 的结构化实现、顶层函数、函数内 lazy import。所以它只适合做"有没有漏画"的对照，不适合用来理解。

## 让结构被强制（已加，2026-10-12）

`pyproject.toml` 的 `[tool.importlinter]`：layers `cli > cleanup|chat > ingest|model_providers|config`；
ingest 禁 `openai`/`google`；model_providers 禁 `whisper`。运行 `uv run lint-imports`。
新增模块（如 B3 `tools/`）时加进 layers，同时更新 `src/01-overview.architecture.json`。
