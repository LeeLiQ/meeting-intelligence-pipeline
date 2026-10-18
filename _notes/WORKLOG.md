
## 2026-07-03 — 项目重启为 v3（Meeting Coach，双线并行）
**决策：**
- 目标重定义：Line A 信息抽取能力训练（事件驱动，我先做、agent 批改）+ Line B 从零建 agent（pull 式 backlog）。双线并行，共享素材。
- 旧学习体系（learning/、progresses/）归档到 _archive/；旧 pipeline 代码冻结为只读参考库；新代码将从零写在 coachagent/。
**改动：**
- 新建 PLAN.md（v3 设计）、BACKLOG.md（B1–B6 + 后备）、coach/templates/meeting-template-v1.md、coach/error-log.md。
- 重写 CLAUDE.md 到 v3；.gitignore 增加 coach/recordings/、coach/sessions/（公司会议数据不进 git）。
- 清理了一个卡住的 .git/index.lock（本次 git mv 失败的残留）。
**未决项：**
- 模板 v1 未经真实 audio 验证——第一轮 Line A 后按需改版。
- coachagent/ 尚未创建，等 B1 被取。
- learning/ 下文件原本就未被 git 追踪，归档用的是普通 mv，git 历史无变化。
- 2026-07-03（续）：README.md 写为 v3 项目 intro（是什么/为什么这样设计/怎么执行/进化机制/目录/边界）。

## 2026-08-17 — B1 落点澄清：不跑 uv init
**决策：**
- `coachagent/` 直接建在 repo root 下，与旧 pipeline 共享同一个 pyproject / uv.lock / .venv，
  **不**执行 `uv init`。理由：root 已是 uv project。实测两种跑法——
  `uv init`（root）→ 报错 `Project is already initialized`，不覆盖也不新建；
  `uv init coachagent` → 不报错，但会在 root pyproject 追加 `[tool.uv.workspace] members`
  并在子目录生成第二个 pyproject，把单项目拆成 workspace（两套依赖声明），与 PLAN 的
  "共享环境、旧 pipeline 只读"意图冲突。
- B1 无需 `uv add`：pydantic、python-dotenv 已在 root dependencies 里。
- `uv run python -m coachagent.hello` 能找到模块，是因为 root pyproject 没有 `[build-system]`
  （uv 的 app 模式，非 package），`uv run` 以项目根为 cwd，cwd 在 `sys.path` 上，无需 install。
**改动：**
- BACKLOG.md B1 条目改写：删去"`uv init` 建包"的措辞，补"注意"行记录上述两个陷阱。
**未决项：**
- `coachagent/` 仍未创建，等 B1 被取。

## 2026-09-06 — secrets 存放策略（决策：维持现状）
**背景：** VS Code 弹 `python.terminal.useEnvFile` 通知，引出"key 该不该从 .env 搬到 Windows 环境变量"的讨论。
**结论：**
- Windows 环境变量（User/System 皆然）是注册表明文，不加密；搬过去只是换暴露面形状，不是安全升级。若真要搬，只用 User 级。
- 本地开发单 principal，`.env` + gitignore 已足够；残余风险只有"进 git / 被同步"，gitignore 已覆盖。补偿控制是 key 可随时 deactivate/rotate。
- 因此**不引入 `keyring`/DPAPI**。生产环境的对应物是 AWS Parameter Store / Secrets Manager（那里的价值在多 principal + audit + rotation，本地不成立）。
**未决项：**
- `python.terminal.useEnvFile` 那条通知本身还没处理（选 A 开注入 or 选 B 置空 `python.envFile` 改代码里 `load_dotenv()`）。B 系列真正跑起代码时再定。

## 2026-09-07 — Line B 意义复核
- 疑问：旧 pipeline 都在，从零写 `coachagent/` 意义何在？
- 检查：grep `helper/`、`main.py`，无 `tool_calls`/`tools=`/`tool_choice`/`max_steps`；LLM 调用只有单发 `response_format`（structured output）。
- 结论：旧代码是 pipeline，没有 agent loop / tool calling。Line B 的不可替代部分从 B3 开始；B1 只是地基，B2（Whisper 包装）与旧 stage 重复。
- 未决：B2 是砍掉还是改成"import 旧 Whisper stage"，待 Qian 决定。下一步建议直取 B3。
- 决定：B2 改为"import 旧 Whisper stage"而非重写，BACKLOG 已更新。理由（Qian）：昨天 B1 单次调用就撞上 OpenAI SDK 升级，旧方法不可用——这类漂移不跑不知道。
- B1 撞到的 SDK 变动：`client.chat.completions.create` → `client.responses.create`。旧 `helper/llm/openai_provider.py:93` 仍用 chat.completions（非 structured 路径），structured 路径（:51）已是 `responses.parse`。预测：B2 只 import Whisper stage 时不会触发它；但旧 tests 里凡走非 structured 路径的可能已 fail——B2 开始时先跑一遍旧 tests 验证。
- Qian 提出更大的疑问：目录里绝大多数东西是否都该 archive、从零开始？请求重新讨论项目目的。
- 跨 session 检索（子 agent）：转型需求原始讨论在 "AI Agent learning project"（local_86a5298e）；"Pi Agent 学习平台评估"（local_a5e142ef）把目标细化为 短期=懂 agent harness 怎么建、长期=站到上游（requirements/spec/verification/taste）。原讨论中 Qian 已表态"愿意 archive 旧 pipeline 从零重建"，v3 选的是折中（原地冻结）。
- 待决：目录重整方案，等 Qian 回答"07-03 以来 Line A 跑过几轮"。
- Qian 反馈：Line A 只做过第 1 步（3 次，breadcrumbs → 整理稿，不在 repo 内），第 2–4 步从未做；coach/recordings 与 coach/sessions 皆空。判断：Line A 断在素材（无录音/transcript），不在 agent。
- 产出 `outputs/project-brief-2026-09-07.md`：整个 repo 的思路/现状/推理链/待决项，供与"转型路径" project 对照。待决 A–D 未决。
- 收到转型 project 回函 `_notes/reply-brief-2026-09-07.md`：portfolio 为主、coach 为副；A 素材断点不成立（录音存在且已授权，待放入 coach/sessions/）；B 目录重整同意；C 砍 B2 直取 B3；PLAN v4 由转型 project 写入。
- 阻塞发现：git 工作区 54 项未提交（最后 commit f2ece91 早于 v3 重启），CLAUDE.md/PLAN/BACKLOG/coach/coachagent 全是 untracked，helper/ 亦有未提交修改。sandbox 无法写 .git（index.lock 权限）。目录重整前必须先由 Qian 本机 commit 快照。
- Qian 已 commit v3 快照 5811700。目录重整完成：helper/ main.py tests/ prompts/ samples/ → _archive/pipeline-v1/；Whisper 重写为 coachagent/transcribe.py（未对真实录音跑过）；BACKLOG：B1 打勾、B2 取消、B3 置顶并加 SDK 注意；CLAUDE.md 两处路径更新。
- 待 Qian 本机执行：`git tag pipeline-v1-final 5811700`，然后 `git add -A && git commit -m "archive pipeline-v1; add coachagent.transcribe; cancel B2"`。sandbox 无法删 __pycache__/pytest-cache 残留（gitignored，可手删）。
- 待 Qian：录音放入 coach/sessions/<date>-<slug>/；找一段公共会议录音作 demo 素材。PLAN v4 由转型 project 写入，本侧未动 PLAN.md。

## 2026-10-11 — 新协作模式 + MVP skeleton r1
**决策：**
- 协作模式变更（Qian 提出）：Qian 说目标 → 双方各自出思路/结构（Claude 出带注释的代码骨架，注明 lib/package）→ 对比迭代若干轮 → 确认后 Qian 亲手写。
- 第一个目标：workflow（非 agent）—— mp4 路径 → 抽音频 → Whisper transcript → 一次 AI 清理 → 终端问答（验收题："有几个不同的人"）。
- 骨架设计：`ingest/` 做成 package（内部不 print/input，进度走 callback，便于将来接 UI）；`model_providers/` 参考 Pi：统一 Message + ABC `LLMProvider.chat()` + 每家 adapter + `create_provider()` 工厂 + key 解析链（env → .env → getpass）。对话历史放 client 端 `ChatSession`，不用 OpenAI `previous_response_id`。
- `Transcriber` 用 Protocol、`LLMProvider` 用 ABC（后者有共享方法 `ask()`）。
**改动：**
- 新建 `outputs/skeleton/r1/`（README + coachagent 骨架，只有签名/注释/`...`），未动 `coachagent/` 本体。
**未决项（见 r1 README 第 5 节 Q1–Q6）：**
- Q1 Whisper 不做 speaker diarization → "几个人"只能 LLM 推断；r1 是否接受（备选 pyannote/whisperx 或云端 diarize）。
- Q2 清理边界；Q3 不做 chunking；Q4 key 是否写回 .env；Q5 迁移现有 transcribe.py/client.py/hello.py；Q6 产出落 coach/sessions/。
- 顺带发现：pyproject 的 `google-generativeai` 已 deprecated，加 Gemini 时换 `google-genai`。

## 2026-10-11（续）— skeleton r2
**决策（Qian 答 r1 Q1–Q6）：** 接受推断式"几个人"；清理一次调用不总结；不做 chunking；key 写回且换更好的存储并确定 default；先 commit 再迁移；产出不泄露即可。
**r2 新决策（Claude 拍板，待 Qian 审）：**
- 新增 `config.py`：`~/.coachagent/auth.json`（secrets）+ `settings.json`（default provider/model 等），照 Pi。优先级 CLI > env/.env > settings.json > 内置。仍明文（延续 09-06）。
- `resolve_api_key` 注入 `prompt` 函数（模块内不 input）；`create_provider` 只收值不收 Settings；`CredentialStore` Protocol 在 model_providers，`FileCredentialStore` 在 config，cli 为 Composition Root。三个叶子模块互不 import。
- `_make_session_dir` 用 `git check-ignore` 防止产出落到未 ignore 路径。
- 回答"ffmpeg/whisper 算不算 tool"：不算——无决策空间即 workflow；入口变成对话 agent 后可包成 `transcribe_media` tool，实现即 `ingest_file`。
- B3 改为 `search_transcript`/`read_transcript_window`；`extract_meeting_notes` 并入 B5。BACKLOG 已插入 B-MVP 并改 B3。
**改动：** `outputs/skeleton/r2/`（含 tools/ 预告）；BACKLOG.md。
**未决项：**
- Qian 本机：删 `.git/index.lock`（本次 sandbox `git status` 残留）→ commit 快照 → 开写 MVP。
- `_archive/` 下 git 显示 " M" 均为 CRLF 差异（--ignore-cr-at-eol 无 diff），可选 `core.autocrlf true` + `add --renormalize`。
- 告知 Qian：cleanup/问答会把 transcript 文本发到 OpenAI API；纯本地方案为 Ollama（base_url 已留口）。

## 2026-10-11（续）— B-MVP Step 1 通过
- Qian 已开写 `coachagent/`（骨架拷入后填实现）。Step 1（detect + audio）手跑通过。
- 踩坑：输出目录不存在 → ffmpeg 退出码 4294967294（-2/ENOENT）。修复：`out_dir.mkdir(parents=True, exist_ok=True)`；并建议把 CalledProcessError 的 stderr 尾部包进异常。
- Qian 自评：新语言环境下容易忘记 try/except（非阻塞，熟悉后补）。
- r2 README 新增 §5.1：Step 1–4 的 smoke test 命令（PowerShell，`$mp4` 变量）。

## 2026-10-12 — B-MVP Step 2–3 完成，进入 Step 4
- Step 3 通过：create_provider → ask 能回话（key 来自 .env）。registry 的 openai 条目由 Qian 取消注释（骨架本就留空）。
- config.py 修正：FileCredentialStore.set() 补 mkdir；save_settings 加 indent=2；json import 挪顶部；load_settings 改 catch ValidationError。
- **Loose end：** auth.json 写回分支未实测。因为 .env 的 OPENAI_API_KEY 优先级更高，prompt → store.set 这条路走不到。
  测法：当前终端 `Remove-Item Env:OPENAI_API_KEY`（若 `Test-Path` 为 True），去掉命令里的 load_dotenv，跑 r2 README §5.1 Step 3。
  期望：提示输入 → 生成 ~/.coachagent/auth.json → 再跑不提示。

## 2026-10-12（续）— 架构可视化
- Qian 需求：Python 代码零碎，看不清 class/函数关系。
- 产出 `outputs/architecture/ARCHITECTURE.md`：模块依赖 flowchart、类与函数 classDiagram（`<<functions>>` 框 ≈ C# static class；虚线 = Protocol 结构化实现）、cli 运行 sequenceDiagram。三张 Mermaid 已用 mermaid@11 parse 校验通过。
- 自动生成：`uvx --from pylint pyreverse -o mmd ...` 已在本 repo 验证可跑；局限：看不见 Protocol 结构化实现、顶层函数。
- 可选未加：`import-linter` 强制模块边界；显式继承 Protocol 让关系可见。
- 发现：`ingest/__init__.py`、`model_providers/__init__.py` re-export 仍是注释 → cleanup.py/chat.py 的 `from coachagent.ingest import ...` 会 ImportError（Step 4a 前要补）。~~registry 顶层 import openai_provider~~（更正：实为函数内 lazy import，Claude 误读）。

## 2026-10-12（续）— import-linter 约束
- 应 Qian 要求加模块边界约束：pyproject.toml 追加 `[tool.importlinter]` 3 条契约（layers: cli > cleanup|chat > ingest|model_providers|config；ingest 禁 openai/google；model_providers 禁 whisper）。备份：`_scratch/pyproject.toml.<ts>.bak`。
- 验证：当前代码 3 kept；在 /tmp 副本注入违规（含函数内 lazy import）→ 正确报 2 broken 并给出行号。
- **待 Qian 本机**：`uv add --dev import-linter`（sandbox 不能动 Windows .venv/uv.lock），之后 `uv run lint-imports`。
- Review `__init__.py`：结构正确。建议 model_providers 再导出 `MissingCredentialError`（cli 要 catch）。~~registry 顶层 import~~ 更正：该 import 本就在 `_build_openai` 内，是 Claude 读 ast.walk 扫描结果时误判，无需改动。
- ARCHITECTURE.md §5 更新为"已加"。新增模块（B3 tools/）需加入 layers。

## 2026-10-13 — Python review 最小范围
- 应 Qian 要求，基于 coachagent/ 现状整理 review Agent 代码所需的 Python 最小范围：`outputs/python-review-minimum.md`（A 新语法 / B 运行时陷阱 / C Pydantic+typing / D LLM 调用层，每项指向 repo 出处）。
- 顺带用清单 review 现有代码，sandbox 实测确认 4 个 🔴：cleanup.py import 不存在的 BadRequestError（ImportError）且该分支会返回 None；model_providers `__all__` 拼错 MissingCredentialsError；config.py `pydantic.ValidationError` 未 import（NameError）。
- 未改代码（留给 Qian 亲手修，B-MVP 协作模式）。建议：开 Pylance basic type checking + 每模块跑一次 import smoke。

## 2026-10-13 — Step 4 code review
- 产出 `outputs/reviews/2026-10-13-step4-review.md`。ruff 22 处报错（20 处未定义名），import-linter 3/3 kept。
- 🔴 阻塞：cli.py 缺全部 import、session_dir 未定义、125 行漏 `#`（中文被当标识符 → NameError）、无路径参数时未调交互输入；cleanup.py import 不存在的 `BadRequestError`；model_providers `__all__` 拼写 `MissingCredentialsError` 且未 import。
- 🟠 关键行为 bug：ChatSession 把所有异常包成 RuntimeError → cli REPL 的 `except ProviderError` 永不命中（"出错不退出 REPL"失效）；clean_transcript 的 except 分支可能隐式返回 None。原则：只在跨抽象边界转换异常，回滚用 bare `raise`；temperature 兼容属 adapter 职责（kwargs 省略 None——SDK 默认 omit，显式 None 会发 null）。
- 其他：config.py `pydantic.ValidationError` 未 import（潜伏）；--from-json/--whisper-model 未接通；未写默认 settings.json；Q6 `git check-ignore` 未实现；EOFError；progress 顺序；property 副作用；prompt 缩进与前言。
- 给出 cli.py 参考结构（`main(argv) -> int`、互斥组、`_ensure_ignored`、`_repl`），argparse 部分 sandbox 实测通过。
- 建议：`uv add --dev ruff` + Pylance standard 类型检查，弥补 Python 无编译期检查。

## 2026-10-16 — 架构图重做（Mermaid → Archify）
- Qian 反馈：10-12 的 Mermaid 图仍难懂、看不清。根因诊断：一张 classDiagram 混了三个层级（模块依赖 + 全部 class/函数 + Protocol 线），没有"从入口读起"的路径。
- 调研 Archify（tt-a1i/archify，MIT，v3.1）：agent 写 typed JSON → Node 渲染成单文件交互 HTML（缩放/路径高亮/暗亮主题/导出 PNG·SVG），带 schema + 布局 + 真浏览器校验。
  适合：入口/架构/时序这种"讲给人看"的图。局限：不从代码自动解析，校验只保证图没画坏、不保证与代码一致；代码改了要手动更新 JSON。
- 在 sandbox 实跑 archify（git clone + @sparticuz/chromium 当 ARCHIFY_CHROME），3 张图 finalize 全部 4 gates pass，并截图人工看过（亮/暗）。
- 产出 `outputs/architecture/`：`01-overview.html`（cli.main ≈ Program.cs，①→⑤ 调用顺序 + DI）、`02-packages.html`（包 ≈ csproj，__init__ ≈ public 表面，Protocol 虚线）、`03-runtime.html`（时序）；JSON 源在 `src/`；ARCHITECTURE.md 改为阅读顺序 + C# 映射表。旧版备份 `_scratch/ARCHITECTURE.md.20261016-150043.bak`。
- 布局取舍：01 图把 ffmpeg/Whisper 并进 ingest_file 的 sublabel（否则 ②③ 行序要颠倒才能不交叉）。
- 未决：Qian 本机若要自己改图需 `npx skills add tt-a1i/archify -g`（Claude Code）；B3 加 tools/ 时同步更新 src/01-overview。

## 2026-10-17 — Step 4 修复复查
- chat.py 已改为 `except BaseException: pop(); raise` ✅；config.py ValidationError 已修 ✅；import-linter 3/3 kept。
- ruff 仅剩 session_dir（F821 ×3），但 cli.py 23–27 行 import 指向不存在的模块（`config`/`credentials`/`registry` 裸模块名、`getpass` 来自 stdlib、`load_dotenv` 来自 dotenv、`ProviderError` 经 chat 导入）→ 运行时 ModuleNotFoundError/ImportError。ruff 不做 import 解析，import-linter 只看 coachagent.* 内部，两者都盲区。sandbox 网络不通，未能实跑验证，由静态核对确认。
- 未改：cleanup.py 仍 import BadRequestError；model_providers `__all__` 仍 `MissingCredentialsError` 且未 import；cli 外层异常边界被整体删除（非阻塞）。
- 讲解 session_dir：main() 局部变量；Python 无块作用域（if/else 内赋值、块外可用）。
- 未改 review 文档（Qian 要求）。

## 2026-10-17（续）— 复查 #2
- ruff：All checks passed。cli import 全部改正；cleanup.py 精简为一行调用；model_providers 导出 MissingCredentialError ✅。session_dir 已定义。
- Qian 自加第 4 条契约 "Only model_providers may talk to LLM SDKs" → BROKEN：forbidden 契约默认检查**传递依赖**（cli→model_providers→openai_provider→openai）。意图是只禁直接 import → 需 `allow_indirect_imports = true`。
- 剩余（非阻塞首跑）：cli 直接改写 args.source / settings.sessions_dir（建议局部变量）；--from-json、--whisper-model 未接通；外层异常边界、Q6 check-ignore、默认 settings.json、EOFError 未做；openai_provider 仍显式传 temperature=None（若 OPENAI_MODEL 为 reasoning 模型，cleanup 的 0.1 会被拒 → 首跑最可能的失败点）。
- 2026-10-17：Claude 代改 cli.py 两处：args.source/settings.sessions_dir 改为局部变量 source/sessions_root；REPL 复合语句拆行。ruff 通过。
