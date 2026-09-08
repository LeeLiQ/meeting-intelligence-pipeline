# BACKLOG — Line B（pull 式）

> 规则：每项 ≤30 min、独立可完成、有验收。有精力就取一个：对 Claude 说 **"取 B\<N\>"**。
> 做完打勾并在条目下写一行收获。Claude 负责补充新条目，保持待取项 ≥3 个。

- [ ] **B1 — 从零搭包**：在 repo root 直接建 `coachagent/` 包（**不跑 `uv init`**——root 已是 uv project），
  写一个能跑的 hello-LLM 脚本（从 `.env` 读 key）。
  - 目的：现代 Python 工程起步（uv、pyproject、包布局）。
  - 验收：`uv run python -m coachagent.hello` 打印一条模型回复。
  - 概念：uv workflow / src layout。参考：`helper/llm/` 的 provider 接口（看思路，不复用代码）。
  - 注意：`uv init` 在 root 会直接报错拒绝（pyproject 已存在）；`uv init coachagent` 则会把 root 的
    pyproject 改成 `[tool.uv.workspace]` 并生成第二个 pyproject —— 两者都不要。pydantic 与
    python-dotenv root pyproject 里已有，B1 无需 `uv add`。

- [ ] **B2 — transcribe CLI（改：复用不重写）**：`coachagent transcribe <audio>` 直接 **import** 旧 pipeline 的
  Whisper stage，不重新包 Whisper。（2026-09-07 决定：旧 stage 与新写会完全重复，Line B 的不可替代部分从 B3 开始。）
  - 目的：跨包 import 与 uv 的包解析；顺带确认旧代码在当前依赖版本下还能跑（昨天 B1 已撞过一次 OpenAI SDK 升级，
    这类"不跑不知道"的漂移正是复用旧代码的价值）。
  - 验收：对一段真实 recording 产出 transcript（落在 `coach/sessions/`，不进 git）；`coachagent` 里不出现第二份 Whisper 调用代码。
  - 参考：`helper/pipeline/stages.py` 的 Whisper stage；注意 `helper/` 只读，不为了 import 方便去改它。

- [ ] **B3 — 第一次 tool call**：定义 `extract_meeting_notes` 的 tool schema，让模型返回 tool-call 请求，打印其 args。
  - 目的：tool/function calling 解剖——只到"模型提议调用"，不执行。
  - 验收：能用自己的话说清 tool schema 的三要素，以及模型*何时决定*调用。
  - 概念：tool calling vs structured output（v2 的 confusion pair #1/#3）。

- [ ] **B4 — 最小 agent loop**：执行 tool → 结果喂回 → final answer 或 `max_steps` 停。
  - 目的：agent loop + **stop conditions**（v2 记录的头号薄弱点）。
  - 验收：能指出所选 stop condition 防住哪个失败模式（如无限 tool 调用）。

- [ ] **B5 — Schema 化模板**：把会议模板定义成 Pydantic schema，agent 从 transcript 填出合法实例并渲染成 `agent.md`。
  - 目的：schema 驱动抽取；模板与代码同源，模板升版即 schema 升版。
  - 验收：对一段 transcript 产出通过校验的实例。
  - 概念：structured output / Pydantic v2。

- [ ] **B6 — diff 批改器**：对比 `mine.md` 与 `agent.md`，产出"漏项 / 误解 / 结构差异"报告。
  - 目的：闭合 Line A 循环——从此批改自动化。
  - 验收：对一次真实 session 跑出 `diff.md`，且我认可其中至少一条批改。

**后备（B6 之后再展开）：** B7 RAG over `coach/sessions/`（embedding、vector store、chunking）
→ B8 error-pattern memory（跨 session 记住我的薄弱点，针对性出题）→ B9 eval + tracing。
