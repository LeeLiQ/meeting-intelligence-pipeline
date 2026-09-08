# Learning Progress

> This is the engine of the loop. Claude reads it at the start of every session to
> decide what to teach next, and updates it at the end of every session. Newest entry
> on top. Keep it honest — empty/weak sections are signals, not failures.

---

## You are here (Stage 0 — baseline, before any agent/RAG work)

**What already exists (don't re-teach):** deterministic linear pipeline
(`PipelineRunner` over `PipelineStage` Protocol), Strategy/Factory LLM providers,
versioned file prompts, JSONL observability (`llm_logger.py`), Pydantic schemas
(`SemanticPayload`, `RequirementWorksheet`), provider-native structured output in
`RequirementWorksheetStage`, 90+ passing pytest tests.

**Closest things to the new material I already have:**
- Structured output (`response_format=`) → one step from tool calling.
- `PipelineContext` (mutable state bag) → one step from agent/graph state.
- `pipeline_guards.py` (quality gate, conflict detection) → agent guardrails.
- `InterpretationStage` stuffing JSON+transcript into the prompt → hardcoded RAG.

## Known weak spots (carried over from week-1 notes — close these in M1/M3)
- Can't clearly explain **stop conditions** in an agent loop. *(target: Milestone 1)*
- Haven't implemented even a single **tool call** in code. *(target: Milestone 1)*
- Confused **Tool Use vs. Function Calling** triggers. *(error type: concept gap;
  target: Milestone 1, confusion pair #1)*

## Milestone tracker
- [ ] **M1** — Single tool-calling agent (raw loop)        ← next
- [ ] **M2** — Multi-tool agent on a framework
- [ ] **M3** — Retrieval as a tool (RAG)
- [ ] **M4** — A real memory layer
- [ ] **M5** — Eval, guardrails, tracing, present

## Next session target — Session 1 (kicks off Milestone 1)
- **Concepts (max 3):** (1) function-calling vs. tool-use; (2) anatomy of a tool schema;
  (3) the minimal agent loop + stop conditions.
- **Worked example:** trace one tool-call round-trip by hand (model proposes →
  we execute → we return → model continues).
- **Exercise:** `learning/exercises/session-01-first-tool-call.md`.
- **Deliverable:** extend `helper/llm/base.py` so one provider can return a tool-call
  request, and call it once from a throwaway script (real loop comes next session).
- **Acceptance:** I can state the stop condition I chose and one failure mode it guards;
  the script prints a model-decided call to `extract_semantics` with valid args.

---

## Session log

<!-- Copy this block for each new session; newest on top. -->

### Session N — YYYY-MM-DD — (milestone)
**Learned:**
-
**Got wrong (+ error type):**
-
**Weak spots now:**
-
**Next session target:**
-

---

# 中文版

> 专业术语保留英文。这是循环的引擎：Claude 在每次 session 开始时读它来决定接下来教什么，
> 并在每次 session 结束时更新它。最新条目放最上面。空的/薄弱的小节是信号，不是失败。
> 注：日常更新请直接编辑上面的英文部分（那是真正被读取的活日志）；本中文版仅作对照。

## 你在这里（Stage 0 — 基线，尚未做任何 agent/RAG 工作）
**已经存在的东西（不要重复教）：** 确定性线性 pipeline（`PipelineRunner` 跑
`PipelineStage` Protocol）、Strategy/Factory 的 LLM provider、versioned 文件 prompt、
JSONL observability（`llm_logger.py`）、Pydantic schema（`SemanticPayload`、
`RequirementWorksheet`）、`RequirementWorksheetStage` 里 provider 原生的 structured
output、90+ 个通过的 pytest 测试。

**我已有的、最接近新材料的东西：**
- Structured output（`response_format=`）→ 离 tool calling 一步之遥。
- `PipelineContext`（可变 state bag）→ 离 agent/graph state 一步之遥。
- `pipeline_guards.py`（quality gate、conflict detection）→ agent guardrails。
- `InterpretationStage` 把 JSON+transcript 塞进 prompt → 写死的 RAG。

## 已知薄弱点（从 week-1 笔记带过来——在 M1/M3 补上）
- 讲不清 agent loop 里的 **stop conditions**。*(目标：Milestone 1)*
- 还没在代码里实现过哪怕一次 **tool call**。*(目标：Milestone 1)*
- 混淆 **Tool Use vs. Function Calling** 的触发。*(错误类型：concept gap；目标：
  Milestone 1，confusion pair #1)*

## Milestone 追踪
（用上方英文的勾选框：M1 单 tool 的 tool-calling agent ← 下一个；M2 框架上的多 tool
agent；M3 retrieval 作为 tool（RAG）；M4 真正的 memory layer；M5 eval/guardrails/
tracing/展示。）

## 下一次 session 目标 — Session 1（启动 Milestone 1）
- **概念（最多 3 个）：**(1) function-calling vs. tool-use；(2) 一个 tool schema 的解剖；
  (3) 最小 agent loop + stop conditions。
- **Worked example：** 手动走一遍一次 tool-call 的往返（模型提议 → 我们执行 → 我们返回 →
  模型继续）。
- **Exercise：** `learning/exercises/session-01-first-tool-call.md`。
- **Deliverable：** 扩展 `helper/llm/base.py`，让一个 provider 能返回一个 tool-call 请求，
  并从一个一次性脚本里调用它一次（真正的 loop 下次再做）。
- **验收：** 我能讲清我选的 stop condition 和它防住的一个失败模式；脚本打印出一个由模型
  决定的、对 `extract_semantics` 的调用且参数合法。

## Session 日志的写法
用上方英文模板：`Session N — 日期 —（milestone）`，分四节：Learned / Got wrong（+错误
类型）/ Weak spots now / Next session target。最新放最上面。
