# claude-extras — optional Claude Code add-ons (level-ups)

Staged here because `.claude/` is protected from automated edits. These are the
article's Step 6 (reusable Skill) and the examiner subagent. **Optional** — the core
loop (CLAUDE.md + learning/) works without them. Add when you want them.

## Install (copy into the repo's `.claude/`)
From the repo root:

```bash
mkdir -p .claude/skills .claude/agents
cp -r learning/claude-extras/skills/domain-learning-master .claude/skills/
cp    learning/claude-extras/agents/examiner.md            .claude/agents/
```

(An empty `.claude/skills/domain-learning-master/` and `.claude/agents/` may already
exist from setup — copying over them is fine.)

## What you get
- **`/domain-learning-master`** — reusable across *other* fields later: scaffolds a
  fresh learning repo and generates Day/Session-1 tasks. Not needed for *this* repo
  (already scaffolded); it's for the next field you pick up.
- **`@examiner`** — a grading subagent that runs in its own context (saves main-thread
  context) and quizzes you after each milestone, then writes weak spots to
  `learning/progress.md`. Invoke with `@examiner` or just "test me on Milestone N".

## Even-more-optional: hooks (hard enforcement)
The article suggests hooks for determinism. If you want them, add to your Claude Code
settings (not scaffolded here, since hooks run shell commands and deserve a manual look):
- **Stop hook** — block ending a session until `learning/progress.md` was modified.
- **SessionEnd hook** — snapshot `logs/` or commit progress when you close the session.

---

# 中文版

> 专业术语保留英文。

放在这里是因为 `.claude/` 不允许被自动编辑。这两样是文章的第 6 步（可复用 Skill）和
examiner subagent。**可选**——核心循环（CLAUDE.md + learning/）没有它们也能跑。想用时再加。

## 安装（复制进 repo 的 `.claude/`）
在 repo 根目录：

```bash
mkdir -p .claude/skills .claude/agents
cp -r learning/claude-extras/skills/domain-learning-master .claude/skills/
cp    learning/claude-extras/agents/examiner.md            .claude/agents/
```

（setup 可能已经建了空的 `.claude/skills/domain-learning-master/` 和 `.claude/agents/`——
覆盖复制没问题。）

## 你会得到什么
- **`/domain-learning-master`** — 以后用在*其他*领域时可复用：搭一个新的学习 repo 并生成
  Day/Session-1 任务。*本* repo 不需要它（已经搭好）；它是给你下一个领域用的。
- **`@examiner`** — 一个在自己 context 里运行的打分 subagent（节省主线程 context），每个
  milestone 后考你，然后把薄弱点写进 `learning/progress.md`。用 `@examiner` 调用，或直接说
  "test me on Milestone N"。

## 更可选：hooks（硬性强制）
文章建议用 hooks 来保证确定性。如果你想要，加到你的 Claude Code 设置里（这里没有预置，因为
hooks 会运行 shell 命令，值得手动过目）：
- **Stop hook** — 在 `learning/progress.md` 被修改之前，阻止结束一次 session。
- **SessionEnd hook** — 关闭 session 时快照 `logs/` 或提交 progress。
