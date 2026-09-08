# BACKLOG — Line B（pull 式）

> 规则：每项 ≤30 min、独立可完成、有验收。有精力就取一个：对 Claude 说 **"取 B\<N\>"**。
> 做完打勾并在条目下写一行收获。Claude 负责补充新条目，保持待取项 ≥3 个。

> **2026-09-07 重排**（依转型 project 回函）：旧 pipeline 已归档到 `_archive/pipeline-v1/`；Whisper 已抄成
> `coachagent/transcribe.py`，B2 因此取消；**下一条直取 B3**。节奏 checkpoint（非日程）：约每周 1 条，10 月底到 B6。

- [x] **B1 — 从零搭包**：在 repo root 直接建 `coachagent/` 包（**不跑 `uv init`**——root 已是 uv project），
  写一个能跑的 hello-LLM 脚本（从 `.env` 读 key）。
  - 目的：现代 Python 工程起步（uv、pyproject、包布局）。
  - 验收：`uv run python -m coachagent.hello` 打印一条模型回复。
  - 概念：uv workflow / src layout。参考：`helper/llm/` 的 provider 接口（看思路，不复用代码）。
  - 注意：`uv init` 在 root 会直接报错拒绝（pyproject 已存在）；`uv init coachagent` 则会把 root 的
    pyproject 改成 `[tool.uv.workspace]` 并生成第二个 pyproject —— 两者都不要。pydantic 与
    python-dotenv root pyproject 里已有，B1 无需 `uv add`。

- [x] ~~**B2 — transcribe CLI**~~ 取消（2026-09-07）：Whisper 无 agent 学习价值，直接由 Claude 从 v1 抄成
  `coachagent/transcribe.py`。用法：`uv run python -m coachagent.transcribe <audio> [-o out.md] [--model base]`。
  首次对真实录音跑通时顺手验证，不单独占一条。

- [ ] **B3 — 第一次 tool call**：定义 `extract_meeting_notes` 的 tool schema，让模型返回 tool-call 请求，打印其 args。
  - 目的：tool/function calling 解剖——只到"模型提议调用"，不执行。
  - 验收：能用自己的话说清 tool schema 的三要素，以及模型*何时决定*调用。
  - 概念：tool calling vs structured output（v2 的 confusion pair #1/#3）。对照物：
    `_archive/pipeline-v1/helper/pipeline/stages.py:407` 的 `response_format=` 就是 structured output——B3 做的是另一件事。
  - SDK 注意：用 `client.responses.create(..., tools=[...])`，不用 `chat.completions`（B1 已撞过废弃）。

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
