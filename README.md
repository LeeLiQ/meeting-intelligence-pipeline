# Meeting Coach — 项目介绍（v3）

> 给未来的我、和任何新开的 Claude session：这份文档讲清这个项目**是什么、为什么这样设计、怎么跑**。
> 分工：设计细节在 `PLAN.md`，可取任务在 `BACKLOG.md`，Claude 的工作规则在 `CLAUDE.md`。

## 这是什么

一个针对两个真实问题的自训练项目。

**问题一：** 常年 WFH 之后，我在 online meeting 中的信息抽取能力受损——听不住、听不进、
抓不准 requirements，business requirements → tech design 变得困难。

**问题二：** 我的技术栈（.NET 主力，Python 停在 3.7）与 AI 时代脱节。我的判断是：
未来有价值的角色不是纯写代码的 software engineer，而是能听懂需求、架构 agent 系统去实现
目标的 solution engineer——而信息抽取恰恰是这个角色的第一环。

**解法（一石三鸟）：** 建一个 meeting-coach agent。*用*它训练我的抽取能力（治问题一），
*建*它的过程学会 Agent 工程与现代 Python（治问题二）。

## 为什么是这个设计

前两版都失败了，教训只有一条：**任何要求"专门学习时间"的设计都会死。**
v1/v2 有精心设计的 milestones、session 制、验收标准——但一次 session 都没发生过
（存档见 `_archive/learning-v2/`）。v3 的三条设计原则，每条都针对这个失败：

**1. 事件驱动，不设日程。** 训练由真实 recording 触发：有会就练，没会不练，不欠账。
动机来自真实痛点，不靠自律。

**2. 我先做，agent 批改。** 要恢复的是*我*的能力，所以必须我先听、我先抽取；agent 是
examiner，不是替我干活的秘书。若反过来（agent 先做、我 review），练的是"验证"而非"抽取"，
正中痛点的下怀。

**3. Pull 式 backlog，30 分钟封顶。** Agent 工程的学习任务全在 `BACKLOG.md`，每项 ≤30 min、
独立可完成、有验收。有精力就取一个，没精力它就在那——没有进度条，不存在"落后"。

还有一个结构性决定：**双线并行，互不卡脖子。** Line A（能力训练）不等工具建好——agent
还不存在时，由 Claude 手动扮演它，训练照跑。Line B（工具建设）也不被 Line A 拖着——它沿
学习弧线（tool calling → agent loop → structured output → RAG/memory → eval）自己往前走。
两线共享素材（Line B 的语料就是 Line A 的产出），在终点汇合。

## 怎么执行

日常只有两个入口，对 Claude 说触发词即可：

**"来一轮"（附一段 audio）→ Line A 训练**，四步，约 30–45 min：

1. 我听 audio，边听边填 `coach/templates/` 最新模板 → `coach/sessions/<date>-<slug>/mine.md`
2. Claude/agent 独立处理同一段 audio → `agent.md`
3. Diff 批改 → `diff.md`；错误模式（去掉公司细节）追加 `coach/error-log.md`，分四类：
   attention lapse / concept gap / 结构化不足 / 术语不懂
4. 模板不好用就升版；流程哪步别扭，就往 `BACKLOG.md` 丢一条改进项

**"取 B\<N\>" → Line B mini-session**：按 `BACKLOG.md` 该条执行——一句话讲清概念动机 →
一起写代码 → 过验收 → 打勾并写一行收获。Claude 会要求我用自己的话复述，以验证真的学会了。

## 进化机制

没有固定课程表，系统靠三个反馈环自己长：

- **模板环** —— 每轮训练暴露模板缺陷 → 模板升版；B5 之后模板即 Pydantic schema，升版即代码升版。
- **错误环** —— `error-log.md` 积累我的错误模式 → 批改越来越对准我的薄弱点；B8 之后
  agent 记住这些模式并主动针对性出题。
- **工具环** —— Line B 每交付一个能力（transcribe → 抽取 → diff → memory），Line A 就少一个
  手动步骤。终点状态：一轮训练 = 一条命令 + 我的 `mine.md`。

## 目录

| 路径 | 是什么 |
|------|--------|
| `PLAN.md` | v3 设计细节 |
| `BACKLOG.md` | Line B 可取任务（pull 式） |
| `CLAUDE.md` | Claude 的工作规则与触发词 |
| `coach/` | 模板、error-log、本地 sessions（recordings/sessions 不进 git） |
| `coachagent/` | 新代码，从零构建（取 B1 时创建） |
| `helper/`、`main.py`、`tests/` | 旧 pipeline，冻结为只读参考库 |
| `_archive/` | v1/v2 学习体系存档 |
| `_notes/WORKLOG.md` | 工作日志（每次任务追加） |

## 边界

公司会议数据只留本地：`coach/recordings/`、`coach/sessions/` 已 gitignore，绝不进 git、
不进任何共享语料；能进 git 的只有模式与结构。明确不做的事：训练或微调模型、追求"完整课程
覆盖"、以及任何形式的打卡。
