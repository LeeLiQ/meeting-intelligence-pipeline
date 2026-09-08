# 01 — Knowledge Map: AI Agents + RAG / Memory

> The map's job is to tell you **what to skip**. You are an experienced engineer, so
> this map omits general software engineering and anchors every AI concept to a
> concrete place in *this* repo or a milestone in `project_plan.md`. Learn the
> "must-learn 20%" first; the "skip for now 60%" is listed at the bottom on purpose.

---

## 0. What these two fields actually solve

**Agents.** Your pipeline today is a *fixed* sequence: `main.py` wires stages, and
`PipelineRunner.run()` executes them in the same order every time. That is the
correct design when *you* know the steps at code-time. An **agent** is what you
reach for when the steps depend on the input and must be decided *at runtime by the
model*: "given this transcript, decide whether to extract requirements, search past
meetings, or just summarize." Agents trade determinism for adaptability. The whole
field is about controlling that trade-off (loops, tools, stop conditions, guardrails).

**RAG / memory.** An LLM is stateless and has a bounded context window. RAG
(Retrieval-Augmented Generation) is the technique of *fetching the relevant external
text and putting it into the prompt* so the model answers from real data instead of
its weights. "Memory" is the larger system that decides *what to remember, when to
retrieve it, and how to compact it* across turns and across meetings. You already do
a primitive, hardcoded version of RAG: `InterpretationStage` stuffs the extracted
JSON **plus** the transcript into the prompt to ground the PRD (see the `TODO` in
`stages.py` that worries about the token cost of doing this — that TODO *is* the
RAG-vs-long-context trade-off).

These two fields meet in the project arc: an **agent** that uses **retrieval/memory**
as one of its tools.

---

## 1. Core concepts — Agents (the must-learn 20%)

Each line: concept — one-sentence definition — **where it lives for you**.

1. **Pipeline vs. agent** — static, developer-decided control flow vs. dynamic,
   model-decided control flow. → You already wrote this distinction in
   `progresses/week1-summary.md`; it is the bridge concept. Note: `notes/pipeline-vs-agent.md`.
2. **Tool / function calling** — you expose a function with a JSON schema; the model
   emits a structured *request* to call it; **your code** executes it and returns the
   result. → Milestone 1. Your `LLMProvider.generate()` will grow a `tools=` path.
3. **Tool schema design** — the description and parameter schema are *prompt
   engineering*, not just typing. → You already write schemas: `SemanticPayload`,
   `RequirementWorksheet` (Pydantic). Tool schemas are the same skill pointed at actions.
4. **Structured output vs. tool calling** — both use a schema; structured output
   constrains the *answer's shape*, tool calling chooses an *action*. → You use
   structured output today in `RequirementWorksheetStage` (`response_format=`). This is
   the closest thing you already have to tool calling — start here.
5. **The agent loop** — perceive → decide → act → observe → repeat until done. →
   Milestone 1 replaces `PipelineRunner`'s fixed `for stage in stages` with a loop
   where the model picks the next action.
6. **Stop conditions** — how the loop knows it's finished (final answer, max steps,
   budget, guard). → Your week1 notes flagged "can't explain stop conditions" — this is
   a named weak spot to close in Milestone 1.
7. **ReAct (reason + act)** — the canonical loop: the model writes a thought, picks a
   tool, sees the result, repeats. → The mental model for Milestone 1's raw loop.
8. **Orchestration patterns** — single agent + tools vs. multi-agent handoffs vs.
   graph of nodes. → Milestone 2. Your `PipelineContext` ≈ LangGraph `State`; your
   `PipelineStage` Protocol ≈ a LangGraph node. The jump is small *by design* (see the
   docstring in `stages.py` that already name-checks LangGraph/Haystack/Agents SDK).
9. **State / context passing** — what flows between loop iterations. → You have this:
   `PipelineContext` is a mutable state bag. Agents formalize it.
10. **Planning vs. reactive** — plan-then-execute vs. decide-one-step-at-a-time. →
    A Milestone 2 comparison exercise.
11. **Guardrails / validation / human-in-the-loop** — cheap deterministic checks
    around expensive model calls. → You already do this: `pipeline_guards.py`
    (`check_transcript_quality`, `detect_conflicts`). Reframe these as agent guardrails.
12. **Agent observability / tracing** — per-step spans, token/latency/tool-call logs.
    → You already have `llm_logger.py` (JSONL per call). Milestone 5 extends it from
    "per LLM call" to "per agent step + per tool call."
13. **Cost & latency control** — step caps, model routing, parallel tool calls. → Your
    `LLMFactory` already routes by model name; your logger already tracks tokens/latency.

If you only learn 1–7 deeply, you can build a working agent. 8–13 make it *good*.

---

## 2. Core concepts — RAG / Memory (the must-learn 20%)

1. **Why RAG exists** — stateless model + bounded context → fetch relevant text at
   query time. → Milestone 3. The honest framing for your "search past meetings" tool.
2. **Embeddings** — text → a vector, so "similar meaning" ≈ "near in space." → Milestone
   3. New dependency (an embedding model); mirrors your local-Whisper philosophy if you
   pick a local embedder.
3. **Vector store + similarity search** — store vectors, retrieve top-k by cosine /
   ANN. → Milestone 3. A new `helper/memory/` package; start with a local store.
4. **Chunking** — how you split a document before embedding (size, overlap, boundaries).
   → Milestone 3. Your artifacts chunk naturally: per-requirement, per-decision,
   per-meeting — better than blind 500-token windows.
5. **Indexing pipeline vs. query pipeline** — *write path* (embed + store) vs. *read
   path* (embed query → retrieve → inject). → Milestone 3. The write path is literally a
   new pipeline stage; you already know how to add stages.
6. **Grounding / context injection** — putting retrieved text into the prompt to reduce
   hallucination. → You already do the hardcoded version in `InterpretationStage`.
7. **Top-k, similarity threshold, MMR** — knobs that trade recall vs. noise vs.
   redundancy. → Milestone 3 tuning exercise.
8. **Hybrid search (vector + keyword/BM25)** — combine semantic and exact-match. →
   Skip at first; revisit if pure-vector recall disappoints.
9. **Reranking** — a second model reorders candidates for precision. → Skip-for-now;
   know it exists.
10. **Memory types** — short-term (this session's scratchpad) vs. long-term (the vector
    store); semantic (facts) vs. episodic (what happened when). → Milestone 4, the
    concept that distinguishes "RAG" from "memory."
11. **Memory operations** — write, retrieve, update, forget, **compact/summarize**. →
    Milestone 4. Compaction is the one engineers underrate.
12. **RAG eval** — retrieval quality (recall@k, is the right chunk in the top-k?) vs.
    generation quality (faithfulness/groundedness: did the answer stick to retrieved
    text?). → Milestone 5. You can build a tiny labeled set from your own past meetings.
13. **Context-window management** — budgeting tokens across system prompt, retrieved
    context, history, and output. → Milestone 4–5; connects back to the cost concern in
    your `InterpretationStage` TODO.

---

## 3. The 10 confusion pairs to nail (tailored to you)

These are the distinctions where "I thought I understood" breaks. Write a note when
you can state each in your own words *and* point to where it bites in this repo.

1. **Function calling vs. tool use** — the model *proposes* a call; *you* execute it.
   (You flagged this exact confusion in week1.)
2. **Pipeline vs. agent** — who decides the next step: the code, or the model.
3. **Structured output vs. tool calling** — shape-the-answer vs. choose-an-action;
   both are "JSON schema sent to the model." (You only have the first today.)
4. **RAG vs. long context** — retrieve the relevant 2KB vs. dump the whole 50KB. (Your
   `InterpretationStage` TODO is this debate, live, in your code.)
5. **RAG vs. memory** — RAG is a *retrieval technique*; memory is a *system* (with
   write/forget/compact policies) that may use RAG underneath.
6. **Embeddings vs. fine-tuning** — retrieve external facts at runtime vs. change the
   model's weights. (Beginners reach for fine-tuning when they need RAG.)
7. **Semantic search vs. keyword search** — meaning-similarity vs. exact-token match.
8. **Short-term vs. long-term memory** — scratchpad for this run vs. durable store
   across runs/meetings.
9. **Chunk vs. context window** — how you *split for storage* vs. how much the model
   can *read at once*. Independent knobs people conflate.
10. **Orchestration framework vs. provider SDK vs. raw loop** — LangGraph (graph/state)
    vs. OpenAI Agents SDK (provider-native agents/handoffs) vs. a hand-written `while`
    loop. You'll build raw first (M1) precisely so the frameworks (M2) demystify.

---

## 4. Beginner → project-ready stages (mapped to milestones)

- **Stage 0 — you are here.** Deterministic linear pipeline, structured output
  (`response_format`), file-based prompt versioning, JSONL observability, Pydantic
  schemas, 90+ passing tests. Strong foundation; nothing agentic yet.
- **Stage 1 — single tool-calling agent.** Extend `LLMProvider.generate()` to support
  tools; wrap one existing stage as a tool; hand-write the loop. *(Milestone 1)*
- **Stage 2 — multi-tool agent on a framework.** Your stages become a toolbox; port the
  loop to LangGraph and/or the OpenAI Agents SDK; compare to the raw loop. *(Milestone 2)*
- **Stage 3 — retrieval as a tool.** Index past meeting artifacts; add
  `search_past_meetings(query)` so the agent can pull prior context. *(Milestone 3)*
- **Stage 4 — a real memory layer.** Short-term scratchpad + long-term store; answer
  "has this requirement/decision come up before?"; add compaction. *(Milestone 4)*
- **Stage 5 — eval, guardrails, tracing, present.** Tiny eval set, groundedness check,
  per-step tracing, write-up. *(Milestone 5)*

---

## 5. Skip-for-now (the 60% — deliberately not learning yet)

Knowing these exist is enough; do **not** rabbit-hole until after Milestone 4.

- Model training / fine-tuning / RLHF / PEFT-LoRA (you need *retrieval*, not training).
- GraphRAG, knowledge graphs, multi-vector / late-interaction (ColBERT).
- Reranker models, query rewriting, HyDE, advanced retrieval research.
- Large-scale / distributed vector DBs, sharding, billion-scale ANN tuning.
- Complex multi-agent societies, debate, role-play swarms.
- Agentic RL, tree-of-thoughts, MCTS-style planning.
- Prompt-compression research, KV-cache tricks, speculative decoding.
- Voice/multimodal agents, browser/computer-use agents.

**Come-back-to-after-the-project (the last 20%):** hybrid search + reranking (improve
M3), evaluation frameworks (deepen M5), one production vector DB, and MCP (exposing
your tools to other agents — a natural sequel once M1–M2 make "tool" concrete).

---

## 6. How to use this map

- Treat §1–§2 as the backlog of concepts; **max 3 per session** (CLAUDE.md rule).
- When you learn one, write `notes/<concept>.md` and check off the confusion pair in §3
  it resolves.
- Every concept should get *used* in the milestone it's anchored to — that's the
  "prove it by shipping" contract. If a concept never shows up in `05_project`-equivalent
  code (this repo), you haven't learned it yet.

---

# 中文版

> 专业术语保留英文。地图的作用是告诉你**该跳过什么**。你是资深工程师，所以本地图省略
> 通用软件工程，并把每个 AI 概念锚定到*本* repo 的具体位置或 `project_plan.md` 的某个
> milestone。先学"必学的 20%"；底部那份"暂时跳过的 60%"是故意列出来的。

## 0. 这两个领域到底解决什么问题

**Agents。** 你今天的 pipeline 是一个*固定*序列：`main.py` 装配 stage，
`PipelineRunner.run()` 每次都以同样顺序执行。当*你*在写代码时就知道步骤，这是正确的设计。
**agent** 则是当步骤取决于输入、必须*由模型在运行时决定*时才用的东西："给定这段
transcript，决定是抽取 requirements、检索过去的会议、还是只做摘要。" Agent 用确定性换取
适应性。整个领域就是在控制这个取舍（loop、tools、stop conditions、guardrails）。

**RAG / memory。** LLM 是无状态的，且 context window 有上限。RAG
（Retrieval-Augmented Generation）就是*把相关的外部文本取出来放进 prompt* 的技术，让模型
基于真实数据而非权重作答。"Memory" 是更大的系统，决定*记什么、何时取回、如何在多轮与多个
会议间压缩*。你已经做了一个原始的、写死的 RAG 版本：`InterpretationStage` 把抽取出的
JSON **加上** transcript 一起塞进 prompt 来给 PRD 提供依据（见 `stages.py` 里那个担心这样
做 token 成本的 `TODO`——那个 TODO *就是* RAG-vs-long-context 的取舍）。

这两个领域在项目弧线里汇合：一个把 retrieval/memory 当作其中一个 tool 的 **agent**。

## 1. 核心概念 — Agents（必学的 20%）
每行：概念 — 一句话定义 — **它在你这儿的位置**。
1. **Pipeline vs. agent** — 静态、由开发者决定的控制流 vs. 动态、由模型决定的控制流。→
   你已经在 `progresses/week1-summary.md` 写过这个区分；它是桥接概念。见
   `notes/pipeline-vs-agent.md`。
2. **Tool / function calling** — 你暴露一个带 JSON schema 的函数；模型发出一个结构化的
   *请求*去调用它；**你的代码**执行它并返回结果。→ Milestone 1。你的
   `LLMProvider.generate()` 会长出一条 `tools=` 路径。
3. **Tool schema design** — 描述和参数 schema 是 *prompt engineering*，不只是类型标注。→
   你已经在写 schema：`SemanticPayload`、`RequirementWorksheet`（Pydantic）。tool schema
   是同一种技能，只是指向"动作"。
4. **Structured output vs. tool calling** — 两者都用 schema；structured output 约束*答案
   的形状*，tool calling 选择一个*动作*。→ 你今天已经在 `RequirementWorksheetStage` 用了
   structured output（`response_format=`）。这是你已有的、最接近 tool calling 的东西——从
   这里开始。
5. **The agent loop** — perceive → decide → act → observe → 循环直到完成。→ Milestone 1
   把 `PipelineRunner` 固定的 `for stage in stages` 换成一个由模型挑选下一步动作的 loop。
6. **Stop conditions** — loop 如何知道自己结束了（final answer、max steps、预算、guard）。
   → 你 week1 的笔记标注过"讲不清 stop conditions"——这是 Milestone 1 要补的明确薄弱点。
7. **ReAct (reason + act)** — 经典 loop：模型写下一个想法、挑一个 tool、看结果、再循环。→
   Milestone 1 raw loop 的心智模型。
8. **Orchestration patterns** — 单 agent + tools vs. 多 agent handoff vs. 节点图。→
   Milestone 2。你的 `PipelineContext` ≈ LangGraph `State`；你的 `PipelineStage` Protocol
   ≈ 一个 LangGraph node。这一跳*被刻意设计得很小*（见 `stages.py` 里已经点名
   LangGraph/Haystack/Agents SDK 的 docstring）。
9. **State / context passing** — 在 loop 各轮之间流动的东西。→ 你已经有了：
   `PipelineContext` 是一个可变 state bag。Agent 把它正式化。
10. **Planning vs. reactive** — 先规划再执行 vs. 一次决定一步。→ Milestone 2 的对比练习。
11. **Guardrails / validation / human-in-the-loop** — 在昂贵的模型调用周围加便宜的确定性
    检查。→ 你已经在做：`pipeline_guards.py`（`check_transcript_quality`、
    `detect_conflicts`）。把它们重新理解为 agent guardrails。
12. **Agent observability / tracing** — 逐步 span、token/latency/tool-call 日志。→ 你已经
    有 `llm_logger.py`（每次调用一行 JSONL）。Milestone 5 把它从"每次 LLM 调用"扩展到
    "每个 agent step + 每次 tool call"。
13. **Cost & latency control** — step 上限、模型路由、并行 tool 调用。→ 你的 `LLMFactory`
    已经按模型名路由；你的 logger 已经记录 token/latency。

如果你只把 1–7 学透，就能搭出一个能用的 agent。8–13 让它*变好*。

## 2. 核心概念 — RAG / Memory（必学的 20%）
1. **Why RAG exists** — 无状态模型 + 有上限的 context → 在查询时取回相关文本。→
   Milestone 3。给你的 "search past meetings" tool 一个诚实的定位。
2. **Embeddings** — 文本 → 一个向量，于是"语义相近" ≈ "空间上相邻"。→ Milestone 3。新依赖
   （一个 embedding 模型）；如果你选本地 embedder，就呼应了你本地 Whisper 的理念。
3. **Vector store + similarity search** — 存向量，用 cosine / ANN 取回 top-k。→
   Milestone 3。一个新的 `helper/memory/` 包；先从本地 store 开始。
4. **Chunking** — embedding 之前怎么切文档（大小、重叠、边界）。→ Milestone 3。你的产物
   天然可切：per-requirement、per-decision、per-meeting——比无脑的 500-token 窗口更好。
5. **Indexing pipeline vs. query pipeline** — *写路径*（embed + store）vs. *读路径*
   （embed query → retrieve → 注入）。→ Milestone 3。写路径其实就是一个新的 pipeline
   stage；你已经会加 stage 了。
6. **Grounding / context injection** — 把取回的文本放进 prompt 以减少幻觉。→ 你已经在
   `InterpretationStage` 做了写死的版本。
7. **Top-k、similarity threshold、MMR** — 在 recall vs. 噪声 vs. 冗余之间取舍的旋钮。→
   Milestone 3 的调参练习。
8. **Hybrid search (vector + keyword/BM25)** — 结合语义与精确匹配。→ 一开始跳过；若纯向量
   recall 不理想再回来。
9. **Reranking** — 用第二个模型为候选重排以提精度。→ 暂时跳过；知道它存在即可。
10. **Memory types** — short-term（本次 session 的 scratchpad）vs. long-term（vector
    store）；semantic（事实）vs. episodic（何时发生了什么）。→ Milestone 4，区分 "RAG" 与
    "memory" 的那个概念。
11. **Memory operations** — write、retrieve、update、forget、**compact/summarize**。→
    Milestone 4。compaction 是工程师最被低估的一个。
12. **RAG eval** — retrieval 质量（recall@k：对的 chunk 是否在 top-k 里）vs. 生成质量
    （faithfulness/groundedness：答案是否贴着取回的文本）。→ Milestone 5。你可以用自己过去
    的会议建一个很小的标注集。
13. **Context-window management** — 在 system prompt、取回的 context、history 和输出之间
    分配 token 预算。→ Milestone 4–5；回接到你 `InterpretationStage` TODO 里的成本顾虑。

## 3. 必须吃透的 10 个 confusion pair（为你定制）
这些是"我以为我懂了"会崩掉的区分点。当你能用自己的话说清每一个*并且*能指出它在本 repo
哪里咬人时，就写一篇 note。
1. **Function calling vs. tool use** — 模型*提议*一次调用；*你*来执行。（你在 week1 标过
   这个困惑。）
2. **Pipeline vs. agent** — 谁决定下一步：代码，还是模型。
3. **Structured output vs. tool calling** — 塑造答案 vs. 选择动作；两者都是"发给模型的 JSON
   schema"。（你今天只有前者。）
4. **RAG vs. long context** — 取回相关的 2KB vs. 把整份 50KB 全塞进去。（你的
   `InterpretationStage` TODO 就是这场争论，活在你代码里。）
5. **RAG vs. memory** — RAG 是一种*检索技术*；memory 是一个*系统*（带 write/forget/compact
   策略），其底层可能用到 RAG。
6. **Embeddings vs. fine-tuning** — 运行时取回外部事实 vs. 改变模型权重。（新手在需要 RAG
   时去碰 fine-tuning。）
7. **Semantic search vs. keyword search** — 语义相似 vs. 精确 token 匹配。
8. **Short-term vs. long-term memory** — 本次运行的 scratchpad vs. 跨运行/跨会议的持久 store。
9. **Chunk vs. context window** — *为存储而切*的方式 vs. 模型*一次能读*多少。两个被混为一谈
   的独立旋钮。
10. **Orchestration framework vs. provider SDK vs. raw loop** — LangGraph（图/状态）vs.
    OpenAI Agents SDK（provider 原生 agents/handoffs）vs. 手写 `while` loop。你会先搭 raw
    （M1），正是为了让框架（M2）变得不再神秘。

## 4. 从入门到能做项目的阶段（映射到 milestone）
- **Stage 0 — 你在这里。** 确定性线性 pipeline、structured output（`response_format`）、
  文件式 prompt versioning、JSONL observability、Pydantic schema、90+ 个通过的测试。底子很
  扎实；但还没有任何 agentic 成分。
- **Stage 1 — 单 tool 的 tool-calling agent。** 扩展 `LLMProvider.generate()` 支持 tools；
  把一个现有 stage 包成 tool；手写 loop。*(Milestone 1)*
- **Stage 2 — 框架上的多 tool agent。** 你的 stage 变成一个工具箱；把 loop 移植到 LangGraph
  和/或 OpenAI Agents SDK；与 raw loop 对比。*(Milestone 2)*
- **Stage 3 — 把 retrieval 作为 tool。** 索引过去的会议产物；加 `search_past_meetings(query)`
  让 agent 能拉取以往 context。*(Milestone 3)*
- **Stage 4 — 一个真正的 memory layer。** short-term scratchpad + long-term store；回答
  "这个 requirement/decision 以前出现过吗？"；加 compaction。*(Milestone 4)*
- **Stage 5 — eval、guardrails、tracing、展示。** 小 eval 集、groundedness 检查、逐步
  tracing、写成文档。*(Milestone 5)*

## 5. 暂时跳过（这 60% — 故意现在不学）
知道它们存在就够了；在 Milestone 4 之前**不要**钻进去。
- 模型训练 / fine-tuning / RLHF / PEFT-LoRA（你需要的是 *retrieval*，不是训练）。
- GraphRAG、知识图谱、multi-vector / late-interaction（ColBERT）。
- Reranker 模型、query rewriting、HyDE、进阶 retrieval 研究。
- 大规模 / 分布式 vector DB、sharding、十亿级 ANN 调优。
- 复杂的 multi-agent 社会、debate、角色扮演 swarm。
- Agentic RL、tree-of-thoughts、MCTS 式 planning。
- prompt 压缩研究、KV-cache 技巧、speculative decoding。
- 语音/多模态 agent、浏览器/computer-use agent。

**做完项目后再回来（最后的 20%）：** hybrid search + reranking（改进 M3）、evaluation
框架（深化 M5）、一个生产级 vector DB，以及 MCP（把你的 tool 暴露给其他 agent——一旦 M1–M2
把 "tool" 讲明白，这是自然的续集）。

## 6. 如何使用本地图
- 把 §1–§2 当作概念待办；**每次 session 最多 3 个**（CLAUDE.md 规则）。
- 每学会一个，就写 `notes/<concept>.md`，并勾掉它在 §3 里解决的那个 confusion pair。
- 每个概念都应在它锚定的 milestone 里被*用上*——这就是"靠交付来证明"的契约。如果一个概念
  从未出现在 `05_project` 等价物（本 repo）的代码里，你就还没学会它。
