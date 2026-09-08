
## 2026-07-03 — 项目重启为 v3（Meeting Coach，双线并行）
**决策：**
- 目标重定义：Line A 信息抽取能力训练（事件驱动，我先做、agent 批改）+ Line B 从零建 agent（pull 式 backlog）。双线并行，共享素材。
- 旧学习体系（learning/、progresses/）归档到 _archive/；旧 pipeline 代码冻结为只读参考库；新代码将从零写在 coachagent/。
**改动：**
- 新建 PLAN.md（v3 设计）、BACKLOG.md（B1–B6 + 后备）、coach/templates/meeting-template-v1.md、coach/error-log.md。
- 重写 CLAUDE.md 到 v3；.gitignore 增加 coach/recordings/、coach/sessions/（公司会议数据不进 git）。
- 清理了一个卡住的 .git/index.lock（本次 git mv 失败的残留）。
**未决项：**
- 模板 v1 未经真实 audio 验证——第一轮 Line A 后按需改版。
- coachagent/ 尚未创建，等 B1 被取。
- learning/ 下文件原本就未被 git 追踪，归档用的是普通 mv，git 历史无变化。
- 2026-07-03（续）：README.md 写为 v3 项目 intro（是什么/为什么这样设计/怎么执行/进化机制/目录/边界）。

## 2026-08-17 — B1 落点澄清：不跑 uv init
**决策：**
- `coachagent/` 直接建在 repo root 下，与旧 pipeline 共享同一个 pyproject / uv.lock / .venv，
  **不**执行 `uv init`。理由：root 已是 uv project。实测两种跑法——
  `uv init`（root）→ 报错 `Project is already initialized`，不覆盖也不新建；
  `uv init coachagent` → 不报错，但会在 root pyproject 追加 `[tool.uv.workspace] members`
  并在子目录生成第二个 pyproject，把单项目拆成 workspace（两套依赖声明），与 PLAN 的
  "共享环境、旧 pipeline 只读"意图冲突。
- B1 无需 `uv add`：pydantic、python-dotenv 已在 root dependencies 里。
- `uv run python -m coachagent.hello` 能找到模块，是因为 root pyproject 没有 `[build-system]`
  （uv 的 app 模式，非 package），`uv run` 以项目根为 cwd，cwd 在 `sys.path` 上，无需 install。
**改动：**
- BACKLOG.md B1 条目改写：删去"`uv init` 建包"的措辞，补"注意"行记录上述两个陷阱。
**未决项：**
- `coachagent/` 仍未创建，等 B1 被取。

## 2026-09-06 — secrets 存放策略（决策：维持现状）
**背景：** VS Code 弹 `python.terminal.useEnvFile` 通知，引出"key 该不该从 .env 搬到 Windows 环境变量"的讨论。
**结论：**
- Windows 环境变量（User/System 皆然）是注册表明文，不加密；搬过去只是换暴露面形状，不是安全升级。若真要搬，只用 User 级。
- 本地开发单 principal，`.env` + gitignore 已足够；残余风险只有"进 git / 被同步"，gitignore 已覆盖。补偿控制是 key 可随时 deactivate/rotate。
- 因此**不引入 `keyring`/DPAPI**。生产环境的对应物是 AWS Parameter Store / Secrets Manager（那里的价值在多 principal + audit + rotation，本地不成立）。
**未决项：**
- `python.terminal.useEnvFile` 那条通知本身还没处理（选 A 开注入 or 选 B 置空 `python.envFile` 改代码里 `load_dotenv()`）。B 系列真正跑起代码时再定。

## 2026-09-07 — Line B 意义复核
- 疑问：旧 pipeline 都在，从零写 `coachagent/` 意义何在？
- 检查：grep `helper/`、`main.py`，无 `tool_calls`/`tools=`/`tool_choice`/`max_steps`；LLM 调用只有单发 `response_format`（structured output）。
- 结论：旧代码是 pipeline，没有 agent loop / tool calling。Line B 的不可替代部分从 B3 开始；B1 只是地基，B2（Whisper 包装）与旧 stage 重复。
- 未决：B2 是砍掉还是改成"import 旧 Whisper stage"，待 Qian 决定。下一步建议直取 B3。
- 决定：B2 改为"import 旧 Whisper stage"而非重写，BACKLOG 已更新。理由（Qian）：昨天 B1 单次调用就撞上 OpenAI SDK 升级，旧方法不可用——这类漂移不跑不知道。
- B1 撞到的 SDK 变动：`client.chat.completions.create` → `client.responses.create`。旧 `helper/llm/openai_provider.py:93` 仍用 chat.completions（非 structured 路径），structured 路径（:51）已是 `responses.parse`。预测：B2 只 import Whisper stage 时不会触发它；但旧 tests 里凡走非 structured 路径的可能已 fail——B2 开始时先跑一遍旧 tests 验证。
- Qian 提出更大的疑问：目录里绝大多数东西是否都该 archive、从零开始？请求重新讨论项目目的。
- 跨 session 检索（子 agent）：转型需求原始讨论在 "AI Agent learning project"（local_86a5298e）；"Pi Agent 学习平台评估"（local_a5e142ef）把目标细化为 短期=懂 agent harness 怎么建、长期=站到上游（requirements/spec/verification/taste）。原讨论中 Qian 已表态"愿意 archive 旧 pipeline 从零重建"，v3 选的是折中（原地冻结）。
- 待决：目录重整方案，等 Qian 回答"07-03 以来 Line A 跑过几轮"。
- Qian 反馈：Line A 只做过第 1 步（3 次，breadcrumbs → 整理稿，不在 repo 内），第 2–4 步从未做；coach/recordings 与 coach/sessions 皆空。判断：Line A 断在素材（无录音/transcript），不在 agent。
- 产出 `outputs/project-brief-2026-09-07.md`：整个 repo 的思路/现状/推理链/待决项，供与"转型路径" project 对照。待决 A–D 未决。
- 收到转型 project 回函 `_notes/reply-brief-2026-09-07.md`：portfolio 为主、coach 为副；A 素材断点不成立（录音存在且已授权，待放入 coach/sessions/）；B 目录重整同意；C 砍 B2 直取 B3；PLAN v4 由转型 project 写入。
- 阻塞发现：git 工作区 54 项未提交（最后 commit f2ece91 早于 v3 重启），CLAUDE.md/PLAN/BACKLOG/coach/coachagent 全是 untracked，helper/ 亦有未提交修改。sandbox 无法写 .git（index.lock 权限）。目录重整前必须先由 Qian 本机 commit 快照。
