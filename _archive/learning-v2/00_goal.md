# 00 — Goal

## What I'm learning
Two fields, learned together as one project:
1. **AI Agents** — tools, the agent loop, orchestration, guardrails.
2. **RAG / memory layers** — embeddings, retrieval, and a real memory system.

## Why this repo
I already have a well-architected linear pipeline (the Meeting Intelligence Pipeline,
"Stage 0"). Instead of a toy demo, I'll *evolve it* into a memory-augmented agent. The
codebase already contains the bridge concepts (structured output, a state bag, swappable
providers, observability), so each new AI concept attaches to something I built.

## Definition of success (the deliverable)
This repo runs as both (a) the original deterministic pipeline and (b) an agent that
decides which tools to run and retrieves across past meetings, with tracing and a small
eval harness — and I can whiteboard every component without notes. See
`project_plan.md` (Milestones 1–5).

## Cadence
A few focused sessions per week. Plan per session (`progress.md`-driven), milestone
tests after each milestone (not on a weekly clock).

## Constraints
- Keep Stage 0 (pipeline + full test suite) green throughout.
- No company data or secrets in `learning/` or any retrieval corpus.
- Follow existing repo patterns; justify new dependencies.

## How I drive it each session
Open the repo and say: **"Based on learning/progress.md, plan today's session."**

---

# 中文版

> 专业术语保留英文。

## 我在学什么
两个领域，作为一个项目一起学：
1. **AI Agents** — tools、agent loop、orchestration、guardrails。
2. **RAG / memory layer** — embeddings、retrieval，以及一个真正的 memory 系统。

## 为什么用这个 repo
我已经有一个架构良好的线性 pipeline（Meeting Intelligence Pipeline，即 "Stage 0"）。
与其做一次性 demo，不如把它*演进*成一个 memory-augmented agent。代码库里已经包含了
桥接概念（structured output、一个 state bag、可替换的 provider、observability），所以
每个新的 AI 概念都能挂到我已经建好的东西上。

## 成功的定义（deliverable）
这个 repo 同时能作为 (a) 原来的确定性 pipeline 和 (b) 一个能自己决定调用哪些 tool、
并跨历史会议做 retrieval 的 agent 运行，带 tracing 和一个小型 eval harness——而且我能
不看笔记把每个组件画在白板上。见 `project_plan.md`（Milestone 1–5）。

## 节奏
每周几次专注的 session。按 session 规划（以 `progress.md` 驱动），milestone test 在每个
milestone 之后做（不按周打卡）。

## 约束
- 全程保持 Stage 0（pipeline + 完整 test suite）通过。
- `learning/` 或任何检索语料中不放公司数据或 secret。
- 沿用 repo 已有模式；新依赖要说明理由。

## 我每次 session 如何驱动它
打开 repo 然后说：**"Based on learning/progress.md, plan today's session."**
