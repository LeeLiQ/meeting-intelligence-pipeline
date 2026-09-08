# Milestone Test — Template

> Run this **after** each milestone, before starting the next. Use the `@examiner`
> subagent (`.claude/agents/examiner.md`) or paste the prompt below. Copy this file to
> `reviews/M<N>-YYYY-MM-DD.md` and fill it in. The point is to find what you *thought*
> you understood but can't explain or apply.

## Examiner prompt (paste to kick it off)
```
Stop teaching new material. You are now the examiner for Milestone <N>.
Using learning/01_map.md, learning/project_plan.md (Milestone <N>), my notes/, and the
actual code I wrote this milestone, create a test with three parts:
  A. 4–6 multiple-choice / true-false on the concepts (target the confusion pairs).
  B. 2–3 "explain in your own words" prompts (I must connect each to a file in this repo).
  C. 1–2 scenario applications ("given X, what would the agent/retriever do, and why?").
Give me ALL questions first. I answer, THEN you grade.
For each wrong/weak answer, classify the error type (concept gap / can't apply /
can't articulate / knowledge confusion / missing background) and name the note or
map section to revisit. Write weak spots to learning/progress.md and save the full
record here.
```

## Record

**Milestone:** <N>  **Date:** YYYY-MM-DD  **Score:** _/_

### Part A — concepts
_(questions + my answers + grade)_

### Part B — explain in my own words
_(prompts + my answers + grade; each must cite a repo file)_

### Part C — scenario application
_(scenario + my answer + grade)_

### Errors found (with type)
- …

### Verdict
- [ ] Pass — proceed to Milestone <N+1>
- [ ] Patch first — weak spots written to `progress.md`; redo: …

---

# 中文版

> 专业术语保留英文。每完成一个 milestone 后、开始下一个之前运行本测试。用 `@examiner`
> subagent（`.claude/agents/examiner.md`）或粘贴下面的 prompt。把本文件复制为
> `reviews/M<N>-YYYY-MM-DD.md` 再填写。目的是找出你*以为*懂了、但讲不清或做不出的东西。

## Examiner prompt（粘贴以启动）
```
停止教新材料。你现在是 Milestone <N> 的 examiner。
依据 learning/01_map.md、learning/project_plan.md（Milestone <N>）、我的 notes/，以及我
这个 milestone 真正写的代码，出一套三部分的测试：
  A. 4–6 道针对概念的选择/判断题（瞄准那些 confusion pair）。
  B. 2–3 道"用自己的话解释"（我必须把每一题连到本 repo 里的某个文件）。
  C. 1–2 道场景应用（"给定 X，agent/retriever 会做什么、为什么？"）。
先把全部题目给我。我作答，然后你打分。
对每个错/弱的回答，给错误类型分类（concept gap / can't apply / can't articulate /
knowledge confusion / missing background），并指出该回看哪篇 note 或哪节 map。把薄弱点
写到 learning/progress.md，并把完整记录存到这里。
```

## 记录
**Milestone：** <N>  **日期：** YYYY-MM-DD  **得分：** _/_

### Part A — 概念
（题目 + 我的回答 + 评分）

### Part B — 用自己的话解释
（题目 + 我的回答 + 评分；每题须引用一个 repo 文件）

### Part C — 场景应用
（场景 + 我的回答 + 评分）

### 发现的错误（含类型）
- …

### 结论
- [ ] 通过 — 进入 Milestone <N+1>
- [ ] 先补 — 薄弱点已写入 `progress.md`；重做：…
