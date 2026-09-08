# Note: Pipeline vs. Agent

> Exemplar note — this is the format for every `notes/<concept>.md`: define it in your
> own words, name the confusion it resolves, anchor it to this repo, give a tiny example,
> and end with a self-check you must pass. Seeded from `progresses/week1-summary.md`;
> revise it in your own words during Milestone 1.

## In one sentence
A **pipeline** has its control flow decided by *me at code-time*; an **agent** has its
control flow decided by *the model at run-time*.

## The distinction that matters
| | Pipeline | Agent |
|---|---|---|
| Who picks the next step | The code (fixed order) | The model (from context) |
| Tools | Functions called in hardcoded order | Functions the model may choose to call |
| Execution | Run once, start→finish | Loop until a stop condition |
| Predictability | High | Lower; needs guardrails |

## Where it lives in this repo
- **Pipeline (today):** `PipelineRunner.run()` — `for stage in self._stages` is the
  fixed control flow. `main.py` decides the order at code-time.
- **Agent (Milestone 1):** a loop where the model is handed tool schemas and chooses
  which to call; `PipelineContext` becomes the agent's state.

## Confusion pairs this resolves
- #2 pipeline-vs-agent (this note).
- Sets up #1 (function-calling vs. tool-use) — the *mechanism* an agent uses to act.

## Tiny example
Pipeline: `normalize() → extract() → worksheet()` — always, in that order.
Agent: "Here's a transcript and these tools (`normalize`, `extract`, `worksheet`).
Reach the goal." → model may skip `worksheet` if it judges the meeting isn't a
requirements discussion (reusing your `readiness` signal as the cue).

## When NOT to use an agent
If the steps never change, a pipeline is better: cheaper, deterministic, testable. Your
Stage 0 is *correct* as a pipeline. Add agency only where runtime decisions add value.

## Self-check (must pass before marking learned)
- [ ] I can point to the exact line that makes Stage 0 a pipeline, not an agent.
- [ ] I can name one decision in my own meetings worth handing to the model, and one
      that should stay hardcoded.
- [ ] I can state why "agent" implies "needs a stop condition" but "pipeline" doesn't.

---

# 中文版

> 专业术语保留英文。示范 note——这是每篇 `notes/<concept>.md` 的格式：用自己的话定义它、
> 点出它解决的 confusion、锚定到本 repo、给一个极小的例子、以一个你必须通过的 self-check
> 收尾。取材自 `progresses/week1-summary.md`；在 Milestone 1 期间用自己的话改写它。

## 一句话
**pipeline** 的控制流由*我在写代码时*决定；**agent** 的控制流由*模型在运行时*决定。

## 真正重要的区分
| | Pipeline | Agent |
|---|---|---|
| 谁选下一步 | 代码（固定顺序） | 模型（依据 context） |
| Tools | 按写死顺序调用的函数 | 模型可以选择调用的函数 |
| 执行 | 跑一次，从头到尾 | 循环直到满足 stop condition |
| 可预测性 | 高 | 较低；需要 guardrails |

## 它在本 repo 的位置
- **Pipeline（今天）：** `PipelineRunner.run()`——`for stage in self._stages` 就是固定
  控制流。`main.py` 在写代码时就决定了顺序。
- **Agent（Milestone 1）：** 一个把 tool schema 交给模型、由模型选择调用哪个的 loop；
  `PipelineContext` 成为 agent 的 state。

## 它解决的 confusion pair
- #2 pipeline-vs-agent（本篇）。
- 为 #1（function-calling vs. tool-use）做铺垫——agent 用来行动的*机制*。

## 极小的例子
Pipeline：`normalize() → extract() → worksheet()`——永远按这个顺序。
Agent："这是一段 transcript 和这些 tools（`normalize`、`extract`、`worksheet`）。达成
目标。" → 若模型判断这不是 requirements discussion，它可以跳过 `worksheet`（复用你的
`readiness` 信号作为线索）。

## 什么时候*不要*用 agent
如果步骤从不变化，pipeline 更好：更便宜、确定、可测试。你的 Stage 0 作为 pipeline 是*正确*
的。只在运行时决策能带来价值的地方才加入 agency。

## Self-check（标记为已学会之前必须通过）
- [ ] 我能指出哪一行让 Stage 0 是 pipeline 而非 agent。
- [ ] 我能说出我自己的会议里一个值得交给模型的决策，和一个应当写死的决策。
- [ ] 我能讲清为什么 "agent" 意味着"需要 stop condition"，而 "pipeline" 不需要。
