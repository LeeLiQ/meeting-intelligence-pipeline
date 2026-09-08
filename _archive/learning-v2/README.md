# Learning System — Design & How to Use It

> 🟢 **Tired, or don't know where to start? Read only `learning/START_HERE.md`.** This
> file is the full design — reference material, not required reading.

This turns the article's method ("learn any field by making it a project that runs")
into a system wrapped around **this** repo. The key adaptation: you already shipped the
hard part. The article's payoff is "prove you learned by building a project in
`05_project/`". You have a real, well-architected pipeline (Stage 0). So the project
isn't a throwaway — **the project is evolving this repo into a memory-augmented agent**,
and that single arc teaches both target fields (AI Agents *and* RAG/memory) in the order
you'd actually build them.

## The idea in one line
Don't restart with AI every session. Open the repo and say:
**"Based on learning/progress.md, plan today's session."** — your notes compound, your
blind spots shrink, and the deliverable is the repo itself.

## How the article maps onto this repo

| Article piece | Here |
|---|---|
| `CLAUDE.md` (rules loaded every session) | `/CLAUDE.md` — tuned for an experienced eng learning agents+RAG |
| `progress.md` (the loop's engine) | `learning/progress.md` — seeded with "you are here" + Session 1 |
| `01_map.md` (draw the map first) | `learning/01_map.md` — concepts anchored to your actual files |
| `05_project/` (build a small project) | **this whole repo** → `learning/project_plan.md` (M1–M5) |
| `02_sources/`, `03_notes/`, `04_exercises/`, `06_reviews/` | `learning/{sources.md,notes/,exercises/,reviews/}` |
| Examiner (Step 5) | `@examiner` subagent + `reviews/_milestone-test-template.md` |
| Reusable Skill (Step 6) | `learning/claude-extras/skills/domain-learning-master/` |
| Subagents / hooks (advanced) | `learning/claude-extras/` (optional; install when justified) |

Two deliberate deviations from the article: (1) learning artifacts live under
`learning/` instead of the repo root, so the real codebase stays clean; (2) your
existing `progresses/` week-summaries are kept as historical context, while
`learning/progress.md` is the new living log.

## Files created
```
CLAUDE.md                      # session rules (loaded every session)
learning/
├── README.md                  # this file — the design + how-to
├── 00_goal.md                 # goal, success definition, constraints
├── 01_map.md                  # knowledge map: agents + RAG/memory (what to skip)
├── project_plan.md            # the 5-milestone arc (pipeline → memory agent)
├── progress.md                # living log — the engine of the loop
├── sources.md                 # short curated reading list
├── notes/pipeline-vs-agent.md # exemplar concept note (the format to copy)
├── exercises/session-01-first-tool-call.md
├── reviews/_milestone-test-template.md
└── claude-extras/             # optional: Skill + examiner subagent (+ install README)
```

## The five milestones (full detail in `project_plan.md`)
1. **Single tool-calling agent (raw loop)** — tools, the loop, stop conditions.
2. **Multi-tool agent on a framework** — LangGraph / OpenAI Agents SDK; your
   `PipelineContext` ≈ framework `State`.
3. **Retrieval as a tool (RAG)** — embeddings, vector store, chunking, grounding.
4. **A real memory layer** — short/long-term, episodic/semantic, compaction.
5. **Eval, guardrails, tracing, present** — recall@k, groundedness, per-step tracing.

Stage 0 (the current pipeline + its 90+ tests) stays green the whole way; agent/memory
features are added alongside, not by rewriting working stages.

## Start your first session now
1. (Optional) install the add-ons: see `learning/claude-extras/README.md`.
2. Say: **"Based on learning/progress.md, plan today's session."**
3. That kicks off Session 1 → Milestone 1, closing your carried-over weak spots
   (stop conditions, first tool call, tool-use vs. function-calling). Exercise is
   pre-seeded at `learning/exercises/session-01-first-tool-call.md`.
4. After a milestone, run `@examiner` (or "test me on Milestone 1").

## Why this will stick where "explain AI agents to me" didn't
Every concept in `01_map.md` is anchored to a line you can open, and every milestone
forces you to *use* the concept in runnable code with acceptance criteria. You can't
fool the examiner with fluent-but-vague answers — it grounds questions in the code you
actually wrote.

---

# 中文版

> 专业术语保留英文。

> 🟢 **累了、或不知从哪开始？只看 `learning/START_HERE.md`。** 本文件是完整设计，属参考
> 资料，不是必读。

这套系统把文章里的方法（"通过把任何领域变成一个能运行的项目来学习它"）落地成围绕**本
repo** 的一套系统。关键适配点：你已经把最难的部分做完了。文章的最终回报是"通过在
`05_project/` 里做项目来证明你学会了"——而你已经有一个真实、架构良好的 pipeline
（Stage 0）。所以这个项目不是一次性的——**这个项目就是把本 repo 演进成一个
memory-augmented agent**，而这一条主线会按你真实构建的顺序，同时教会你两个目标领域
（AI Agents *和* RAG/memory）。

## 一句话概括
不要每次 session 都从头跟 AI 开始。打开 repo 然后说：
**"Based on learning/progress.md, plan today's session."**——你的笔记会复利累积，盲区会
缩小，而 deliverable 就是 repo 本身。

## 文章方法如何映射到本 repo

| 文章中的部分 | 这里对应 |
|---|---|
| `CLAUDE.md`（每次 session 加载的规则） | `/CLAUDE.md` — 为"学 agents+RAG 的资深工程师"定制 |
| `progress.md`（循环的引擎） | `learning/progress.md` — 已用"you are here"+ Session 1 预置 |
| `01_map.md`（先画地图） | `learning/01_map.md` — 概念锚定到你真实的文件 |
| `05_project/`（做个小项目） | **整个 repo** → `learning/project_plan.md`（M1–M5） |
| `02_sources/`、`03_notes/`、`04_exercises/`、`06_reviews/` | `learning/{sources.md,notes/,exercises/,reviews/}` |
| Examiner（第 5 步） | `@examiner` subagent + `reviews/_milestone-test-template.md` |
| 可复用 Skill（第 6 步） | `learning/claude-extras/skills/domain-learning-master/` |
| Subagents / hooks（进阶） | `learning/claude-extras/`（可选；需要时再装） |

与文章的两处刻意不同：(1) 学习产物放在 `learning/` 下而不是 repo 根目录，让真实代码库
保持干净；(2) 你已有的 `progresses/` 周总结保留作为历史背景，而 `learning/progress.md`
是新的活日志。

## 已创建的文件
```
CLAUDE.md                      # session 规则（每次 session 加载）
learning/
├── README.md                  # 本文件 — 设计 + 使用说明
├── 00_goal.md                 # 目标、成功定义、约束
├── 01_map.md                  # 知识地图：agents + RAG/memory（学什么/跳过什么）
├── project_plan.md            # 5 个 milestone 的弧线（pipeline → memory agent）
├── progress.md                # 活日志 — 循环的引擎
├── sources.md                 # 精简的阅读清单
├── notes/pipeline-vs-agent.md # 示范用的 concept note（可照此格式）
├── exercises/session-01-first-tool-call.md
├── reviews/_milestone-test-template.md
└── claude-extras/             # 可选：Skill + examiner subagent（含安装 README）
```

## 五个 milestone（完整细节见 `project_plan.md`）
1. **单 tool 的 tool-calling agent（raw loop）** — tools、loop、stop conditions。
2. **框架上的多 tool agent** — LangGraph / OpenAI Agents SDK；你的 `PipelineContext`
   ≈ 框架的 `State`。
3. **把 retrieval 作为一个 tool（RAG）** — embeddings、vector store、chunking、grounding。
4. **一个真正的 memory layer** — short/long-term、episodic/semantic、compaction。
5. **Eval、guardrails、tracing、展示** — recall@k、groundedness、逐步 tracing。

Stage 0（当前 pipeline + 它的 90+ 测试）全程保持通过；agent/memory 功能在旁边添加，而
不是重写已能运行的 stage。

## 现在就开始你的第一个 session
1.（可选）安装 add-ons：见 `learning/claude-extras/README.md`。
2. 说：**"Based on learning/progress.md, plan today's session."**
3. 这会启动 Session 1 → Milestone 1，补上你遗留的薄弱点（stop conditions、第一个 tool
   call、tool-use vs. function-calling）。Exercise 已预置在
   `learning/exercises/session-01-first-tool-call.md`。
4. 一个 milestone 之后，运行 `@examiner`（或说 "test me on Milestone 1"）。

## 为什么这次能记住，而"给我讲讲 AI agent"不行
`01_map.md` 里每个概念都锚定在你能打开的某一行代码上，每个 milestone 都强迫你在带验收
标准的可运行代码里真正用上这个概念。你没法用流畅但空洞的回答骗过 examiner——它会把问题
锚定在你*真正写过*的代码上。
