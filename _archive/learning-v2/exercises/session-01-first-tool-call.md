# Exercise — Session 1: Your first tool call

**Milestone:** 1 (single tool-calling agent). **Time:** ~45–60 min.
**Goal:** make the model *decide* to call one function, with valid arguments. No loop
yet — just one clean round-trip, so the mechanism stops being magic.

## Before you code — answer in writing (5 min)
1. In your own words: when the model "calls a tool," what does it actually return, and
   who runs the function? (Confusion pair #1.)
2. What are the minimum pieces a tool definition needs for the model to call it correctly?
3. What is the *stop condition* for a single-call interaction (vs. a multi-step loop)?

## Build (the deliverable)
1. Add a tools path to one provider. Extend `helper/llm/base.py::LLMProvider.generate()`
   (or add `generate_with_tools(...)`) to accept a list of tool schemas and return any
   tool-call request the model makes (name + parsed arguments) instead of only text.
   Implement it for **one** provider (whichever you know best); leave the other raising
   `NotImplementedError` for now.
2. Define one tool: `extract_semantics(transcript: str)` — reuse the existing
   `SemanticPayload` shape for its description/args. (Don't wire the real extraction yet;
   a stub that returns `{}` is fine — this exercise is about the *call*, not the work.)
3. In a throwaway script (`scratch_tool_call.py`, not committed), pass a short transcript
   + the tool schema and print: did the model ask to call the tool? With what arguments?

## Acceptance criteria
- [ ] Running the script prints a model-decided call to `extract_semantics` with
      arguments that validate against the tool's schema.
- [ ] You can explain the difference between the model *proposing* the call and your code
      *executing* it (point at the exact line where execution would happen).
- [ ] You wrote down the stop condition you used and one failure mode a real loop must
      guard (e.g. the model never stops calling tools).

## Update when done
- Add/revise `notes/tool-calling.md` (define it; resolve confusion pairs #1 and #3).
- Update `learning/progress.md` (Session log + tick the M1 weak spots you closed).
- Don't start the multi-step loop yet — that's Session 2.

---

# 中文版

> 专业术语保留英文。

**Milestone：** 1（单 tool 的 tool-calling agent）。**时长：** 约 45–60 分钟。
**目标：** 让模型*决定*调用一个函数，并给出合法参数。先不做 loop——只要一次干净的往返，
让这个机制不再神秘。

## 写代码之前——书面回答（5 分钟）
1. 用你自己的话：当模型"调用一个 tool"时，它实际返回的是什么，谁来运行这个函数？
   （confusion pair #1。）
2. 一个 tool 定义至少需要哪些部分，模型才能正确调用它？
3. 一次单次调用交互（相对于多步 loop）的 *stop condition* 是什么？

## 构建（deliverable）
1. 给一个 provider 加上 tools 路径。扩展
   `helper/llm/base.py::LLMProvider.generate()`（或新增 `generate_with_tools(...)`）以接受
   一个 tool schema 列表，并返回模型发出的任何 tool-call 请求（名字 + 解析后的参数），而不
   只是文本。先为**一个** provider 实现（你最熟的那个）；另一个暂时
   `raise NotImplementedError`。
2. 定义一个 tool：`extract_semantics(transcript: str)`——复用现有 `SemanticPayload` 的形状
   作为它的描述/参数。（先不接真正的抽取逻辑；返回 `{}` 的桩就行——这个练习关注的是*调用*，
   不是那项工作。）
3. 在一个一次性脚本（`scratch_tool_call.py`，不提交）里，传入一段简短 transcript + tool
   schema，并打印：模型是否请求调用该 tool？用了什么参数？

## 验收标准
- [ ] 运行脚本会打印出一个由模型决定的、对 `extract_semantics` 的调用，且参数能通过该 tool
      schema 的校验。
- [ ] 你能讲清模型*提议*调用与你的代码*执行*之间的区别（指出执行会发生的那一行）。
- [ ] 你写下了你用的 stop condition，以及一个真正的 loop 必须防住的失败模式（例如模型永远
      不停地调用 tool）。

## 完成后更新
- 新增/修订 `notes/tool-calling.md`（定义它；解决 confusion pair #1 和 #3）。
- 更新 `learning/progress.md`（Session 日志 + 勾掉你补上的 M1 薄弱点）。
- 先别开始多步 loop——那是 Session 2。
