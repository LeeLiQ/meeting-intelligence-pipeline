# Python 项目脚手架 —— 给 .NET 工程师的对照表

> 2026-10-17。每一行都对到本 repo 的现状：✅ 已有 · ⚠️ 有但不完整 · ❌ 没有。
> 一句话总结：**.NET 里一个 SDK + Visual Studio 打包给你的东西，Python 里是 6–8 个独立工具拼起来的。**
> 拼法没有唯一标准，下面是 2025–2026 年主流、且本 repo 已经选定的那一套（以 `uv` + `ruff` + `pyright` 为核心）。

---

## 1. 心智模型：最大的三个差异

1. **没有编译这一步。** C# 先编译（Roslyn 把所有类型错误拦在运行前）再运行；Python 直接解释执行，
   写错的名字要**运行到那一行**才报错。所以 Python 必须靠**外部工具**补回"编译期检查"：
   类型检查器（pyright）≈ Roslyn 的语义检查，linter（ruff）≈ analyzers。**它们不拦运行**——有红线也照样能跑。
2. **SDK 和 runtime 是同一个东西。** 装一个 CPython 解释器 = 既能"编译"（自动生成 `.pyc` 字节码，≈ IL）又能运行。
3. **没有"官方套装"。** 包管理、虚拟环境、lint、格式化、类型检查、测试各自独立选型，配置汇总在 `pyproject.toml`。

## 2. 对照表

### 运行时与依赖

| .NET | Python | 本 repo | 说明 |
|---|---|---|---|
| .NET SDK / runtime | **CPython 解释器** | ✅ 3.13 | 由 uv 安装管理：`uv python install 3.13` |
| `global.json`（锁 SDK 版本） | **`.python-version`** + `requires-python` | ✅ `3.13` | uv 读它决定用哪个解释器 |
| `.csproj` | **`pyproject.toml`** | ✅ | 依赖、元数据、**以及所有工具的配置**（`[tool.xxx]` 段）——还兼任 `.editorconfig`、`Directory.Build.props` 的角色 |
| NuGet / nuget.org | **pip / PyPI** | ✅ | |
| `dotnet` CLI（restore/build/run/add package） | **`uv`** | ✅ | `uv add pkg` ≈ `dotnet add package`；`uv sync` ≈ `restore`；`uv run x` ≈ `dotnet run` |
| `packages.lock.json` | **`uv.lock`** | ✅ | 要进 git |
| `bin/` `obj/` + 每项目独立依赖 | **`.venv/`（虚拟环境）** | ✅ | 每个项目一份隔离的解释器 + 包目录；**不进 git**。Python 默认装到全局——venv 就是为了避免这个 |
| `<PackageReference PrivateAssets="all">`（只开发用） | **`[dependency-groups] dev`** | ✅ | `uv add --dev` |
| user-secrets / `appsettings.json` | `.env` + `python-dotenv`；本 repo 另有 `~/.coachagent/` | ✅ | 配置项变多时：`pydantic-settings` ≈ `IOptions<T>` 绑定 |

### 编辑器与"实时报错"

| .NET | Python | 本 repo | 说明 |
|---|---|---|---|
| Visual Studio / Rider | **VS Code** + 扩展 | ⚠️ | 也可用 PyCharm（≈ Rider，开箱即用但较重） |
| Roslyn IntelliSense + 编译错误红线 | **Pylance**（扩展 `ms-python.vscode-pylance`，引擎是 pyright） | ⚠️ | 需选对解释器（`.venv`），并设 `diagnosticMode: workspace` 才会检查未打开的文件 |
| 编译器（命令行） | **pyright**（`uv run pyright`） | ✅ `[tool.pyright]` standard | 与 Pylance 同引擎、读同一配置 → 编辑器和命令行结果一致 |
| Roslyn analyzers / StyleCop | **ruff**（`uv run ruff check`） | ⚠️ 用的是默认规则 | 速度极快（Rust 写的）；规则集按字母前缀开关：F=错误、B=常见 bug、UP=新语法、I=import 排序… |
| `dotnet format` / CSharpier | **ruff format**（`uv run ruff format`） | ❌ 未启用 | ≈ black 格式化器，零配置 |
| ArchUnitNET / NetArchTest；project reference 约束 | **import-linter**（`uv run lint-imports`） | ✅ 4 条契约 | |
| VS 调试器（F5） | **debugpy**（扩展 `ms-python.debugpy`）+ `.vscode/launch.json` | ❌ | 可直接以 `-m coachagent.cli` 方式断点调试 |

### 测试

| .NET | Python（pytest） | 本 repo |
|---|---|---|
| xUnit / NUnit | **pytest** | ✅ `tests/test_smoke.py` |
| `[Fact]` | 普通函数 `def test_xxx():` —— 名字以 `test_` 开头即被发现 | ✅ |
| `[Theory]` + `[InlineData]` | `@pytest.mark.parametrize` | ✅ 已在用 |
| `Assert.Equal(a, b)` | 直接写 `assert a == b`（pytest 会自动展开显示两边的值） | ✅ |
| 构造函数 / `IClassFixture` / `IDisposable` | **fixture**（`@pytest.fixture` + `yield`），按参数名注入 | — |
| `Assert.Throws<T>` | `with pytest.raises(T):` | — |
| 临时目录 | 内置 fixture `tmp_path`（直接写参数名就拿到一个 `Path`） | — |
| Moq / NSubstitute | `unittest.mock`（stdlib）/ `pytest-mock`。**本项目更常用"手写假实现"**：`LLMProvider` 是 ABC、`Transcriber`/`CredentialStore` 是 Protocol，写一个 `FakeProvider` 比 mock 清楚 | — |
| coverlet（覆盖率） | `pytest-cov` | ❌ |
| VS Test Explorer | VS Code **Testing** 面板（`"python.testing.pytestEnabled": true`） | ❌ |
| `dotnet test` | `uv run pytest -q` | ✅（需 `[tool.pytest.ini_options] pythonpath`，已配） |

### 流程

| .NET | Python | 本 repo |
|---|---|---|
| 构建失败即阻止 | **没有自动阻止**——要么手动跑，要么挂 git hook | ⚠️ 手动 |
| Husky.Net（git hooks） | **pre-commit** 框架（`.pre-commit-config.yaml`） | ❌ |
| GitHub Actions / Azure Pipelines | 同样用 GitHub Actions（`astral-sh/setup-uv`） | ❌ |
| `.nupkg` 打包发布 | wheel（`uv build`）→ PyPI；需要 `[build-system]` | ❌ 不需要（app 模式，不发包） |

## 3. 本 repo 的日常循环

```powershell
# 写代码时：编辑器里 Pylance 实时红线（≈ VS 里边写边编译）

# 提交前（≈ 一次 dotnet build + dotnet test）：
uv run ruff check coachagent tests     # lint
uv run pyright                         # 类型检查（"编译"）
uv run lint-imports                    # 架构约束
uv run pytest -q                       # 测试

# 运行：
uv run python -m coachagent.cli <file>
```

为什么 Python 里这四步都得自己跑：没有一个"build"把它们串起来。常见做法是挂成 **pre-commit hook**（commit 时自动跑，失败就不让提交）
或做成 VS Code task / 一个 `check.ps1` 脚本。

## 4. 还缺的东西（按性价比排序）

| # | 加什么 | 效果 | 成本 |
|---|---|---|---|
| 1 | `.vscode/settings.json` + `extensions.json`（进 git） | 解释器、diagnosticMode、pytest 面板、ruff 作为格式化器全部固定下来；换机器打开即一致 ≈ 把 VS 的 solution 设置提交进 repo | 2 个小文件 |
| 2 | `.vscode/launch.json` | F5 断点调试 `coachagent.cli`（含 `--from-json` 一份配置） | 1 个小文件 |
| 3 | `[tool.ruff]` 明确规则集 + 启用 `ruff format` | 把 review 时用的 F/B/UP/SIM/I 规则固定下来，而不是靠记 | pyproject 几行 |
| 4 | pre-commit hook | 上面四步在 commit 时自动跑 | 1 个 yaml + `uv add --dev pre-commit` |
| 5 | pytest-cov | 覆盖率报告 | B4 写真正的单元测试时再加 |
| 6 | GitHub Actions | 远端也跑一遍 | 推到 GitHub 后再说 |
