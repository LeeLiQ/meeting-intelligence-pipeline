# meeting-intelligence-pipeline — 项目思路整理（2026-09-07）

> 用途：拿到"转型路径"那个 project 里对照，找出两边脱节处。
> 只写事实与推理链，不写公司会议内容。

## 1. 这个 repo 存在的理由（来自 7 月初的转型讨论）

转型目标被切成两层：

- **短期**：弄懂一个 agent harness 是怎么建起来的（tool calling → agent loop → structured output → memory/RAG → eval）。
- **长期**：站到上游——requirements → spec → verification → taste，除了最后写代码以外的全部。写代码那层是 agent 正在替代的层。

痛点：长期 WFH 导致会议中信息抽取能力受损，business requirements → tech design 慢且不准。
这正是"长期"那层的入口能力，所以选了 **meeting coach** 作为 vehicle：一个题目同时喂两层。

## 2. v3 设计（PLAN.md，2026-07-03）

- **Line A（长期层，人是主角）**：有真实会议录音就"来一轮"：我先填模板 → agent（现阶段由 Claude 手动代替）独立抽取 → diff 批改 → 错误模式进 `coach/error-log.md`（attention lapse / concept gap / 结构化不足 / 术语不懂）。
- **Line B（短期层，pull 式 backlog）**：从零在 `coachagent/` 写 agent，每条 ≤30 min。学习弧：B1 搭包 → B2 transcribe → B3 tool call → B4 agent loop → B5 schema 化模板 → B6 diff 批改器 → B7 RAG → B8 error-pattern memory → B9 eval。
- **交汇点**：Line B 每建成一个能力，替换 Line A 的一个手动步骤。终点：一轮 Line A = 一条命令 + 我的 `mine.md`。
- v1/v2 失败原因：要求专门学习时间。v3 原则：不设日程、不欠账、事件驱动。

## 3. 当时的一个折中，现在看是错的

7 月讨论里我说过"愿意 archive 旧 pipeline 从零重建"。v3 实际落地是**原地冻结**（`helper/`、`main.py`、`tests/`、`prompts/` 只读保留）。
代价这两天显现：B1/B2 与旧代码重复、怕误触旧代码、分不清 response 来源。
今天 grep 确认：旧代码是纯 pipeline（单发 `response_format` structured output），**没有 tool calling、没有 agent loop**——Line B 的不可替代部分从 B3 才开始，B1 是地基，B2 是冗余。

## 4. 当前实际状态（事实）

| 项 | 状态 |
|---|---|
| Line B | B1 完成（`coachagent/hello.py` 能打印模型回复）；昨天撞到 OpenAI SDK 变动：`chat.completions.create` → `responses.create` |
| Line A 第 1 步 | 做了 3 次：会中记 breadcrumbs，会后按接近模板的格式重写。会议质量差，模板很难填满 |
| Line A 第 2–4 步 | **一次都没做**。原因是把"agent 步"误解为依赖 Line B；PLAN 原文是 Claude 手动代替 |
| `coach/recordings/` | 空 |
| `coach/sessions/` | 空（3 份整理稿不在 repo 内） |
| `coach/error-log.md` | 空表 |
| 模板 | 仍是 v1，没有因实战而升过版 |

## 5. 推理链：真正的瓶颈在哪

1. Line A 的价值全部来自 **diff**（我漏了什么 / 误解了什么），diff 需要同一段输入的第二份独立产出。
2. 第二份产出不需要 agent，Claude 手动就能做——但需要**录音或 transcript**作为共同输入。
3. 目前没有任何录音/transcript。所以 Line A 的闭环断在**素材**上，不在 agent、不在目录。
4. "会议质量差、填不满模板"这个感受本身无法判断是会议没内容还是我漏听——恰恰是最需要对照的情形。
5. 目录整理（archive 旧 pipeline）是对的、应该做，但它解决的是 Line B 的干扰，**不解决 Line A 的断点**。

## 6. 待决事项（需要在转型 project 里对照的）

- **A. 素材源**：那 3 次会议有录音/自动 transcript 吗？公司政策允许录吗？
  - 有 → 放进 `coach/sessions/<date>-<slug>/`，Line A 第 2–4 步立刻可跑。
  - 不能录 → Line A 的 ground truth 必须换（会后 JIRA ticket、正式 minutes、同事 notes），PLAN 需重写。
- **B. 目录重整**：`helper/`、`main.py`、`tests/`、`prompts/` → `_archive/pipeline-v1/` + git tag；Whisper 几十行抄进 `coachagent/transcribe.py`；repo 只剩 `coach/` 与 `coachagent/` 两个一等公民。B2 相应改写。
- **C. Line B 顺序**：B2 降级或砍掉，下一条直取 B3（第一件旧代码教不了的东西）。
- **D. 与转型路径的对齐点**（请在另一边核对）：
  - 长期层（上游）的训练在本 repo 里就是 Line A——它现在是零产出。转型路径那边是否还把"上游能力"寄托在这个 repo 上？
  - 短期层（harness）的训练是 Line B——进度 B1/9。那边的预期节奏是什么？
  - "Pi Agent 评估"里提过用 Pi 做 spec 写作陪练——与本 repo 的 Line A 是替代还是互补？
  - Eval 讨论里的冷启动方法（跑 5–10 次真实会议 → 错误聚类成 metrics）依赖 Line A 有产出，目前前提不成立。

## 7. 相关 session（本机可读）

- AI Agent learning project — `local_86a5298e`（转型需求原始讨论，产出 v3）
- Pi Agent 学习平台评估 — `local_a5e142ef`（短期/长期两层的来源）
- Eval 工程选题分析 — `local_d32bb06f`
- live-转型的路径指导和进度 — `cse_019E9BxaL8QbnUMfcikDoHFL`（云端 session，本机不可读）
