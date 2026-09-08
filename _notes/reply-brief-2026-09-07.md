---
title: 回函 · 转型路径 project → meeting-intelligence-pipeline project（2026-09-07）
created: 2026-09-07
type: handoff
用途: 回答 project-brief-2026-09-07 的待决事项 A–D，并给出两个 project 的分工。贴回 pipeline project 使用。不含任何公司会议内容。
---

# 回函：对 project-brief-2026-09-07 的答复

## 0. 先校准前提（这边 8 月以来的变化，那边还不知道）

- **验收者已定（2026-08-10）：外部面试官。** 目标 6–9 个月内拿到 AI engineer offer（上限 12 个月）。这条改变一切优先级。
- **会议管道的定位＝换工作的 portfolio 主件**，不是产品（会议 AI 红海，不做市场验收），也不再首先是 coach。
- **练兵场（vault 问答 agent）＝检索/eval 零件**，其评测集（10 题、带评分钥匙）已于 2026-08-22 定稿。
- **上游能力的训练已经在别的载体上发生**：template 练习（5 份理想输出倒推 spec，进度 3/5）、eval 出题（已完成）、工作中"承诺状态打标＋48 小时探针"。

## 1. 两边的"冲突"到底是什么

不是做什么的冲突，是**为什么**的冲突：v3 把管道当 **coach**（用户是学习者本人，价值＝我与机器抽取结果的 diff），这边把它当 **portfolio**（受众是面试官，价值＝可展示的工程判断）。

裁决：**portfolio 为主，coach 为副产品。** 理由是判官已定。但 coach 的核心机制不浪费——**Line A 的 diff 就是 eval**：人工理想输出 vs 机器输出，同一份 diff 读两个方向：读"我漏了什么"是 coach，读"管道错了什么"是 eval。机制保留，主消费者换成后者。这在面试里反而是好故事："我的评测集同时是我自己的训练反馈。"

## 2. 对待决事项的答复

**A. 素材源——断点不成立，那边的诊断基于过时信息。** Qian 2026-09-07 确认：录音**存在且已获授权**，只是尚未放进 repo 的目录（brief 生成时那边的讨论已让他困惑，未及纠正）。处理：
- 把已有录音放进 `coach/sessions/<date>-<slug>/`，Line A 第 2–4 步与 template 练习的"判分测试"即可运行，无需等待。
- **portfolio 仍不依赖公司录音**：公开 demo 与公开 repo 一律用公共录音（开源社区例会、市政会议等）。公司录音只喂私有实例，不出界。

**B. 目录重整——同意，照做。** 旧 pipeline 归档到 `_archive/pipeline-v1/` 打 tag；Whisper 那几十行抄进 `coachagent/transcribe.py`。这与这边计划里的"接管 Python 代码"一致：接管的意思不是维护旧代码，是读懂、把需要的那块提出来重写、其余归档。

**C. Line B 顺序——同意，B2 砍掉，直取 B3。** 旧代码教不了的东西从 tool calling 开始。

**D. 对齐点逐答：**
- *上游能力是否还寄托在本 repo？* 部分。上游训练的主载体是 template 练习和 eval 出题（都已在产出），本 repo 的 Line A 是可选加强，不是唯一依托。Line A 零产出不构成转型路径的断点。
- *Line B 的预期节奏？* 这边计划：管道完成体（template、eval、可点开的 demo、README、一篇写作物）落在 **2026-09 中 → 2026-11 中**。折算：B3–B6 是核心（tool call、loop、schema 化 template、diff 判分器），B7 RAG 与练兵场共用，B8 memory，B9 eval 已有方法论直接套。**大约每周 1 条 B**，10 月底到 B6，11 月中 B9。这是 checkpoint 不是日程——v3 的事件驱动、≤30 分钟、不欠账原则保留，v1/v2 的失败教训不能重犯。
- *Pi 与 Line A 替代还是互补？* Pi 作为"极简 harness 源码解剖"对 B3–B4 有参考价值（看一个最小 harness 怎么做 tool calling 和 loop），保留为读物；Pi 作为"spec 写作陪练"与 template 练习重复，**砍掉**。6–9 个月预算里只养一个上游训练载体。
- *Eval 冷启动依赖 Line A 有产出？* 前提部分成立：人工侧的 5 份理想输出已 3/5，template＋checklist 即将从中长出——这就是冷启动。缺的是机器侧，回到 A：用公共录音即可跑通。

## 3. 两个 project 的分工（防止再脱节）

- **转型路径 project（这边）**：管"为什么、验收标准、进度与里程碑"。真相源＝个人 vault `_notes/WORKLOG.md` 与 `outputs/00-转型索引.md`。
- **pipeline project（那边）**：管"怎么建"。真相源＝repo 的 PLAN.md，请升到 v4 反映本回函。
- 跨界物只有三样：template、checklist、脱敏/合成示例。原始会议笔记不过界。
- 双方各自的 handoff 文档是唯一桥梁；一处决定另一处照抄，不再两边各议一遍。

## 4. 下一条动作（那边）

1. 把已有录音放进 `coach/sessions/`；另找一段公共会议录音作为公开 demo 素材。
2. 目录重整（B）＋ B3 起步。
3. PLAN.md 升 v4：portfolio 为主、coach 为副、砍 B2 与 Pi 陪练。（v4 由转型路径 project 直接写入 repo——前提是 repo 文件夹已连接到该 project。）
