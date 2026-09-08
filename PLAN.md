# PLAN v3 — Meeting Coach（双线并行）

> 2026-07-03 重启。v1/v2 失败的共同原因：要求"专门的学习时间"，而我缺的正是时间和动力。
> v3 绑定真实事件（meeting recordings），不设日程、不欠账。旧学习体系存档于 `_archive/learning-v2/`。

## 一句话目标

用一个 meeting-coach agent 恢复我的信息抽取能力（**Line A**），并通过从零构建它学会
Agent 工程与现代 Python（**Line B**）。两线并行，共享素材，互不卡脖子。

---

## Line A — 信息抽取训练（人是主角，事件驱动）

**触发：** 有新的 meeting recording，或想消化一段存量 audio。没 recording 不做，不欠账。

**一轮流程（约 30–45 min）：**

1. **我先做。** 听 audio，边听边填 `coach/templates/` 里的最新模板
   → 存为 `coach/sessions/<date>-<slug>/mine.md`。第一遍尽量不暂停；允许第二遍补。
2. **Agent 独立做。** agent（现阶段由 Claude 手动代替）处理同一段 audio → `agent.md`。
3. **Diff 批改。** 对比两份产出：我漏了什么、误解了什么、结构化差在哪 → `diff.md`；
   把**错误模式**（不含公司细节）追加到 `coach/error-log.md`，并分类：
   *attention lapse（走神漏听）/ concept gap（业务或技术概念不懂）/ 结构化不足（听到了但没归位）/ 术语不懂*。
4. **进化。** 模板不好用就改，版本号 +1；流程哪步别扭，改进项丢进 `BACKLOG.md`。

## Line B — Agent 工程 + Python refresh（pull 式 backlog）

- 所有任务在 `BACKLOG.md`，每个 **≤30 min**、独立可完成、写明目的/验收/教的概念。
- 有精力就取一个（对 Claude 说 *"取 B3"*），没精力它就在那。无提醒、无进度压力。
- 新代码**从零**写在 `coachagent/`（新包、新 entrypoint）。旧 pipeline（`helper/`、`main.py`）
  冻结为**只读参考库**：Whisper 接入、Pydantic schema、JSONL logging、provider 接口值得回头抄思路。
- 学习弧线保留 v2 骨架、砍掉 session 制：raw tool-calling loop → 多 tool → structured
  output → RAG/memory（语料就是 `coach/sessions/`）→ eval/tracing。
- Python refresh 不单列：uv、Pydantic v2、typing、pathlib、asyncio 等**遇到即点名**。

## 交汇点

Line B 每建成一个能力，就替换 Line A 中的一个手动步骤：
transcribe → 独立抽取 → diff 批改 → 错误模式记忆。
**终点状态：** 一轮 Line A = 一条命令 + 我的 `mine.md`；agent 记得我的历史错误模式并针对性追问。

## Guardrails

- **公司会议数据只留本地：** `coach/recordings/` 与 `coach/sessions/` 已 gitignore，
  绝不进 git、不进任何共享语料。`error-log.md` 与模板可进 git——只写模式与结构，不写公司细节。
- 旧 pipeline 的测试保持通过；不改 `helper/`，新代码全部在 `coachagent/`。
- `.env` 存 API key，已 gitignore，不回显。
