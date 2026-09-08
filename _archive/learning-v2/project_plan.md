# 05 — Project Arc: Pipeline → Memory-Augmented Agent

> The project *is* this repo. You don't build a throwaway demo; you evolve the
> Meeting Intelligence Pipeline through five milestones. Each milestone teaches a
> named slice of the map by **shipping a code change you can run and show**.
>
> Cadence: a few focused sessions per week (no daily clock). A milestone is "done"
> when its acceptance criteria pass *and* you've passed the examiner test for it.

---

## The through-line in one paragraph

Today `PipelineRunner` runs stages in a fixed order. We will (1) teach the model to
*choose* which stage to run by exposing stages as **tools** and writing an **agent
loop**; (2) re-express that loop on a **framework** to see the patterns named; (3)
give the agent a **retrieval tool** over the corpus of past meeting artifacts; (4)
turn that retrieval into a real **memory layer** (short-term + long-term, with
compaction); and (5) add **evaluation, guardrails, and tracing**, then write it up.
Agents and RAG/memory get learned in the order you'd actually build them.

---

## Milestone 1 — Single tool-calling agent (raw loop)

**Concepts (map §1):** tool/function calling, tool schema design, agent loop, stop
conditions, ReAct, structured-output-vs-tool-calling.

**Build:**
- Extend `helper/llm/base.py::LLMProvider.generate()` (or add `generate_with_tools`)
  to accept a tool list and return tool-call requests. Implement for one provider first.
- Wrap **one** existing capability as a tool — start with `ExtractionStage`'s work as
  `extract_semantics(transcript) -> SemanticPayload`.
- Write a minimal loop in a new `helper/agent/loop.py`: send transcript + tool schema →
  if the model asks for a tool, execute it, feed the result back → stop on final answer
  or `max_steps`.
- Keep the existing pipeline untouched and working; the agent is a new entrypoint
  (e.g. `python -m helper.agent` or a `--agent` flag).

**Acceptance criteria:**
- Given a transcript, the agent *decides* to call `extract_semantics`, you execute it,
  and it produces the same validated `SemanticPayload` the pipeline would.
- You can articulate, in `notes/`, the exact stop condition you implemented and one
  failure mode (e.g. infinite tool-calling) and how `max_steps` guards it.

**Confusion pair resolved:** #1 function-calling vs. tool-use, #3 structured-output vs.
tool-calling. **Examiner test:** Milestone-1 review before moving on.

---

## Milestone 2 — Multi-tool agent on a framework

**Concepts (map §1):** orchestration patterns, state/context passing, planning vs.
reactive, framework-vs-SDK-vs-raw-loop.

**Build:**
- Expose 2–3 stages as tools: `normalize`, `extract_semantics`, `write_worksheet`.
- Port the M1 loop to **LangGraph** (nodes + a `State` — maps directly onto your
  `PipelineStage` Protocol and `PipelineContext`) **and/or** the **OpenAI Agents SDK**
  (agents + handoffs). Pick one to go deep; skim the other.
- Write a 1-page compare-and-contrast: raw loop vs. framework — what the framework gave
  you (state mgmt, retries, tracing) and what it hid.

**Acceptance criteria:**
- The agent handles a goal that needs ≥2 tools in a model-decided order (e.g. "normalize
  this, then extract, but skip the worksheet if it's not a requirements discussion" —
  reuse `detect_conflicts`/readiness logic as the decision signal).
- Your compare-and-contrast names which `PipelineContext` field maps to which framework
  state concept.

**Confusion pair resolved:** #2 pipeline-vs-agent, #10 framework-vs-SDK-vs-raw-loop.
**Examiner test:** Milestone-2 review.

---

## Milestone 3 — Retrieval as a tool (RAG)

**Concepts (map §2):** why-RAG, embeddings, vector store + similarity search, chunking,
indexing-vs-query pipeline, grounding, top-k/threshold/MMR.

**Build:**
- New package `helper/memory/`. An **indexer**: walk past artifacts
  (`*.normalized.md`, `*.extracted.json`, `*.worksheet.json`), chunk them (chunk *per
  requirement / per decision / per meeting* — your data is pre-structured, use it),
  embed, and store. Start with a **local** vector store (e.g. Chroma, FAISS, or
  sqlite-vec) and a local or API embedder — your call; keep it behind a small interface
  so it's swappable like `LLMProvider`.
- A **retriever**: `search_past_meetings(query, k) -> list[chunk]`.
- Register it as a new agent tool. Now the agent can pull prior context on demand.

**Acceptance criteria:**
- Index ≥3 past meeting outputs. Ask "what did we decide about X before?" and the agent
  retrieves the right chunk and grounds its answer in it (cite the source artifact).
- In `notes/`, state your chunking choice and *why*, and show one query where top-k=2
  vs. top-k=5 changes the answer.

**Confusion pair resolved:** #4 RAG-vs-long-context, #6 embeddings-vs-fine-tuning,
#7 semantic-vs-keyword, #9 chunk-vs-context-window. **Examiner test:** Milestone-3 review.

---

## Milestone 4 — A real memory layer

**Concepts (map §2):** memory types (short/long, semantic/episodic), memory operations
(write/retrieve/update/forget), **compaction/summarization**, context-window budgeting.

**Build:**
- **Short-term:** a per-run scratchpad the agent reads/writes within a session.
- **Long-term:** promote durable facts (decisions, requirements, owners) into the store
  with metadata (meeting date, source). Add a `remember(fact)` and a `forget`/supersede
  path so a later meeting can overturn an earlier decision.
- **Compaction:** when context gets large, summarize old turns/meetings rather than
  dropping them. This directly answers your `InterpretationStage` token-cost TODO.
- Cross-meeting use case: "has this requirement come up before?" → dedup / flag
  conflicts across meetings (a smarter `detect_conflicts` that spans history).

**Acceptance criteria:**
- Feed two meetings where the second changes a decision from the first; the agent
  surfaces the prior decision *and* notes it was superseded.
- You can explain (notes/) the difference between what lives in short-term vs. long-term
  memory and your compaction trigger.

**Confusion pair resolved:** #5 RAG-vs-memory, #8 short-vs-long-term. **Examiner test:**
Milestone-4 review (this is the natural "beginner → intermediate" checkpoint).

---

## Milestone 5 — Eval, guardrails, tracing, present

**Concepts (maps §1 & §2):** agent observability/tracing, cost/latency control, RAG eval
(recall@k, faithfulness/groundedness), guardrails/human-in-the-loop.

**Build:**
- Extend `llm_logger.py` from per-LLM-call to **per-agent-step + per-tool-call** tracing
  (step index, tool name, args hash, retrieved chunk ids).
- A tiny **eval set** from your own meetings: a handful of (question → expected source
  chunk / expected fact) pairs. Measure retrieval recall@k and answer groundedness.
- Promote your heuristic guards (`pipeline_guards.py`) into explicit agent guardrails
  (max steps, budget cap, "refuse if retrieval returns nothing relevant").
- Write `learning/05_writeup.md` + update repo `README.md`: what it does, the
  architecture diagram (you already keep Mermaid in `progresses/`), and a demo script.

**Acceptance criteria:**
- One command runs the agent end-to-end with tracing on; one command runs the eval and
  prints recall@k + a groundedness score.
- A reader who isn't you can run the demo from the README.

**Confusion pair resolved:** consolidates all. **Examiner test:** final milestone review
+ a full mock "explain this system to a senior engineer" oral.

---

## Definition of done (the whole arc)

The repo runs as both (a) the original deterministic pipeline and (b) a
memory-augmented agent that retrieves across past meetings, with tracing and a small
eval harness — and you can whiteboard every box without notes. Stretch sequel: expose
the tools over **MCP** so another agent can call your meeting-memory.

---

## Guardrails for the learning itself

- The existing pipeline must keep passing its full test suite at every milestone (don't break
  Stage 0 to build Stage 1 — add alongside).
- New code follows the patterns already in the repo: swap-able provider interfaces,
  versioned prompts, Pydantic schemas, JSONL logs.
- No company data or secrets in `learning/` or in any indexed corpus — use your own
  practice transcripts and the `samples/` file.

---

# 中文版

> 专业术语保留英文。

> 项目*就是*这个 repo。你不做一次性 demo；你把 Meeting Intelligence Pipeline 经由五个
> milestone 演进。每个 milestone 通过**交付一个你能运行、能展示的代码改动**来教会地图里
> 一个有名字的切片。
>
> 节奏：每周几次专注的 session（不按天打卡）。一个 milestone "完成"的标准是：它的验收
> 标准通过*并且*你通过了它的 examiner test。

## 一段话讲清这条主线
今天 `PipelineRunner` 以固定顺序运行 stage。我们将 (1) 通过把 stage 暴露为 **tools** 并
写一个 **agent loop**，教模型自己*选择*运行哪个 stage；(2) 在一个**框架**上重新表达这个
loop，让那些模式被点名；(3) 给 agent 一个对历史会议产物语料的 **retrieval tool**；(4) 把
那个 retrieval 变成一个真正的 **memory layer**（short-term + long-term，带 compaction）；
(5) 加上 **evaluation、guardrails、tracing**，然后写成文档。Agents 和 RAG/memory 会按你
真实构建的顺序被学会。

## Milestone 1 — 单 tool 的 tool-calling agent（raw loop）
**概念（地图 §1）：** tool/function calling、tool schema design、agent loop、stop
conditions、ReAct、structured-output-vs-tool-calling。
**构建：**
- 扩展 `helper/llm/base.py::LLMProvider.generate()`（或新增 `generate_with_tools`）以接受
  一个 tool 列表并返回 tool-call 请求。先为一个 provider 实现。
- 把**一个**现有能力包成 tool——先从 `ExtractionStage` 的工作开始，做成
  `extract_semantics(transcript) -> SemanticPayload`。
- 在新文件 `helper/agent/loop.py` 里写一个最小 loop：发送 transcript + tool schema → 若
  模型请求一个 tool，执行它并把结果喂回去 → 在 final answer 或 `max_steps` 时停止。
- 保持现有 pipeline 不动、可用；agent 是一个新 entrypoint（例如 `python -m helper.agent`
  或一个 `--agent` 标志）。
**验收标准：**
- 给定一段 transcript，agent *决定*调用 `extract_semantics`，你执行它，并产出与 pipeline
  相同的、通过校验的 `SemanticPayload`。
- 你能在 `notes/` 里讲清你实现的确切 stop condition，以及一个失败模式（例如无限 tool
  调用）和 `max_steps` 如何防住它。
**解决的 confusion pair：** #1 function-calling vs. tool-use，#3 structured-output vs.
tool-calling。**Examiner test：** 进入下一阶段前做 Milestone-1 review。

## Milestone 2 — 框架上的多 tool agent
**概念（地图 §1）：** orchestration patterns、state/context passing、planning vs.
reactive、framework-vs-SDK-vs-raw-loop。
**构建：**
- 把 2–3 个 stage 暴露为 tools：`normalize`、`extract_semantics`、`write_worksheet`。
- 把 M1 的 loop 移植到 **LangGraph**（nodes + 一个 `State`——直接对应你的 `PipelineStage`
  Protocol 和 `PipelineContext`）**和/或** **OpenAI Agents SDK**（agents + handoffs）。挑
  一个深入，另一个略读。
- 写一页 raw loop vs. 框架的对比——框架给了你什么（state 管理、retry、tracing）以及它隐藏
  了什么。
**验收标准：**
- agent 能处理一个需要由模型决定顺序、用到 ≥2 个 tool 的目标（例如"先 normalize，再
  extract，但如果这不是 requirements discussion 就跳过 worksheet"——把
  `detect_conflicts`/readiness 逻辑复用为决策信号）。
- 你的对比说明指出哪个 `PipelineContext` 字段对应框架里的哪个 state 概念。
**解决的 confusion pair：** #2 pipeline-vs-agent，#10 framework-vs-SDK-vs-raw-loop。
**Examiner test：** Milestone-2 review。

## Milestone 3 — 把 retrieval 作为 tool（RAG）
**概念（地图 §2）：** why-RAG、embeddings、vector store + similarity search、chunking、
indexing-vs-query pipeline、grounding、top-k/threshold/MMR。
**构建：**
- 新包 `helper/memory/`。一个 **indexer**：遍历过去的产物（`*.normalized.md`、
  `*.extracted.json`、`*.worksheet.json`），切块（按 *per requirement / per decision /
  per meeting* 切——你的数据已是结构化的，善用它）、embed、存储。先用**本地** vector store
  （如 Chroma、FAISS 或 sqlite-vec）和一个本地或 API embedder——你来定；把它藏在一个小接口
  后面，使其像 `LLMProvider` 一样可替换。
- 一个 **retriever**：`search_past_meetings(query, k) -> list[chunk]`。
- 把它注册为一个新的 agent tool。现在 agent 能按需拉取以往 context。
**验收标准：**
- 索引 ≥3 份过去的会议输出。问"我们以前关于 X 决定了什么？"，agent 取回正确的 chunk 并把
  答案建立在它之上（注明来源产物）。
- 在 `notes/` 里说明你的 chunking 选择*及原因*，并展示一个 top-k=2 与 top-k=5 会改变答案
  的查询。
**解决的 confusion pair：** #4 RAG-vs-long-context、#6 embeddings-vs-fine-tuning、
#7 semantic-vs-keyword、#9 chunk-vs-context-window。**Examiner test：** Milestone-3 review。

## Milestone 4 — 一个真正的 memory layer
**概念（地图 §2）：** memory types（short/long、semantic/episodic）、memory operations
（write/retrieve/update/forget）、**compaction/summarization**、context-window budgeting。
**构建：**
- **Short-term：** 一个 per-run scratchpad，agent 在一次 session 内读写它。
- **Long-term：** 把持久事实（decisions、requirements、owners）连同元数据（会议日期、来源）
  提升进 store。加一条 `remember(fact)` 和一条 `forget`/supersede 路径，让后来的会议能推翻
  早先的决定。
- **Compaction：** 当 context 变大时，对旧的轮次/会议做摘要而不是丢弃。这正好回答你
  `InterpretationStage` 的 token 成本 TODO。
- 跨会议用例："这个 requirement 以前出现过吗？" → 跨会议去重 / 标记冲突（一个能跨历史的、
  更聪明的 `detect_conflicts`）。
**验收标准：**
- 喂入两次会议，第二次改变了第一次的一个决定；agent 浮现出先前的决定*并且*指出它已被取代。
- 你能（在 notes/ 里）讲清 short-term 与 long-term memory 各自装什么，以及你的 compaction
  触发条件。
**解决的 confusion pair：** #5 RAG-vs-memory、#8 short-vs-long-term。**Examiner test：**
Milestone-4 review（这是自然的"入门 → 进阶"检查点）。

## Milestone 5 — eval、guardrails、tracing、展示
**概念（地图 §1 与 §2）：** agent observability/tracing、cost/latency control、RAG eval
（recall@k、faithfulness/groundedness）、guardrails/human-in-the-loop。
**构建：**
- 把 `llm_logger.py` 从"每次 LLM 调用"扩展到 **每个 agent step + 每次 tool call** 的
  tracing（step 序号、tool 名、args hash、取回的 chunk id）。
- 用你自己的会议建一个很小的 **eval 集**：几条 (问题 → 期望的来源 chunk / 期望的事实) 对。
  测 retrieval recall@k 和答案 groundedness。
- 把你的启发式 guards（`pipeline_guards.py`）提升为显式的 agent guardrails（max steps、
  预算上限、"若 retrieval 返回不相关内容则拒答"）。
- 写 `learning/05_writeup.md` + 更新 repo `README.md`：它做什么、架构图（你已在
  `progresses/` 里保留 Mermaid）、以及一个 demo 脚本。
**验收标准：**
- 一条命令开着 tracing 端到端运行 agent；一条命令运行 eval 并打印 recall@k + 一个
  groundedness 分数。
- 一个不是你的人能照着 README 跑通 demo。
**解决的 confusion pair：** 全部收束。**Examiner test：** 最终 milestone review + 一次完整
的"向资深工程师讲清这个系统"的口试。

## 完成的定义（整条弧线）
这个 repo 同时能作为 (a) 原来的确定性 pipeline 和 (b) 一个带 tracing、带小型 eval harness、
能跨历史会议做 retrieval 的 memory-augmented agent 运行——而且你能不看笔记白板出每一个方块。
延伸续集：通过 **MCP** 把 tool 暴露出去，让另一个 agent 能调用你的 meeting-memory。

## 学习本身的 guardrails
- 现有 pipeline 必须在每个 milestone 都保持其完整 test suite 通过（不要为了建 Stage 1 而
  破坏 Stage 0——要在旁边添加）。
- 新代码沿用 repo 已有模式：可替换的 provider 接口、versioned prompts、Pydantic schema、
  JSONL 日志。
- `learning/` 或任何被索引的语料中不放公司数据或 secret——用你自己的练习 transcript 和
  `samples/` 文件。
