# Skeleton r2 — MVP 定稿候选 + B3 tool call 预告

> 2026-10-11，第 2 轮。r1 的 Q1–Q6 已全部落地；r2 自己拍板了 2 个架构点（见 §2）。
> **若你认可，MVP 部分即可开写**（§5 是写代码的顺序）。`tools/` 是 B3 的，MVP 不用建。

## 1. r1 → r2 变化

| Q | 你的答复 | r2 落地 |
|---|---|---|
| Q1 diarization | 接受推断 | 不变：QA prompt 要求列依据 + 置信度；`segment.speaker` 预留 |
| Q2 清理边界 | 一次调用、不总结 | prompt 意图标为已确认；`ask()` 加 `temperature`，cleanup 用 0.1 |
| Q3 chunking | 不做 | 删除 warning 逻辑，明确"一次调用" |
| Q4 key 存储 | 写回，且要更好的方式 + default | **新增 `config.py`**：`~/.coachagent/auth.json` + `settings.json`（照 Pi），见 §3 |
| Q5 迁移 | 先 commit 再迁 | §6 给出命令 |
| Q6 不泄露 | — | `_make_session_dir` 加 `git check-ignore` 硬检查；另见 §3 末尾的 API 数据说明 |

## 2. 结构（r2 定稿）

```
coachagent/
├── cli.py                  Composition Root：唯一 print/input、唯一知道所有具体实现的地方
├── config.py               Settings + FileCredentialStore（~/.coachagent/）        ← r2 新增
├── cleanup.py              workflow 的 AI 节点（一次调用）
├── chat.py                 ChatSession：多轮历史
├── ingest/                 模块 1：输入预处理（models / detect / audio / transcribers / service）
├── model_providers/        模块 2：base(ABC) / credentials(Protocol+解析链) / openai_provider / registry
└── tools/                  B3 预告，MVP 不建                                       ← r2 新增
```

**依赖方向**（箭头 = import）——这是 r2 最想让你审的部分：

```
cli      → config, ingest, model_providers, cleanup, chat
cleanup  → ingest (Transcript), model_providers (LLMProvider)
chat     → model_providers
tools    → ingest                                  (B3)
config   → 无        model_providers → 无        ingest → 无      ← 三个叶子模块互不 import
```

r2 拍板的两点：
1. **`resolve_api_key` 接收注入的 `prompt` 函数**，模块内不 `input`/`getpass`。CLI 传 `getpass`，
   UI 传弹窗，测试传 `None`（非交互，缺 key 直接抛）。和 ingest 的 `on_progress` 同一原则。
2. **`create_provider` 只收值，不收 `Settings` 对象**。`CredentialStore` 是定义在 model_providers 里的
   Protocol，`FileCredentialStore` 在 config 里实现但不 import 它（structural typing），cli 负责注入。
   ≈ C#：接口在 Core，实现在 Infrastructure，Program.cs 注册。

## 3. Q4：key 存储与 default

- 位置：`~/.coachagent/auth.json`（secrets）与 `settings.json`（default provider/model、whisper 模型、sessions 目录）。
  分两个文件是为了 settings 可以随便贴出来排错。
- 为什么比 `.env` 好：在 repo **之外** → 不可能误 commit；换目录运行也找得到。和 aws/gh/Pi 同一做法。
- 安全级别：仍是明文（延续 09-06 不上 keyring 的决策）。要加密时写 `KeyringCredentialStore`，换注入即可。
- **default 优先级**：`--provider/--model` > 环境变量/.env > `settings.json` > 代码内置。
  首次运行写出一份默认 `settings.json` 并打印路径——"确认 default"就是用户打开这个文件看一眼。
- `.env` 仍然有效（优先级更高），你现有的 `.env` 不用动，可以等新方式跑通后再删 key。

**关于"不泄露"的一句实话**：本地文件靠 gitignore + `check-ignore` 兜住了；但 cleanup 和问答会把
transcript **文本**发给 OpenAI API。API 数据默认不用于训练，但确实离开了本机。你的录音已获授权
（09-07 回函），这里只是让你知情。纯本地方案是把 provider 换成 Ollama（`base_url` 指向本地，OpenAI 兼容协议）
——`ProviderSpec.base_url_env` 已经为此留了口。

## 4. 你的问题：调 ffmpeg 和 whisper 算不算 tool call？

**技术上可以包成 tool，但在当前 workflow 里不该算。** 判断标准只有一条：**这一步需不需要模型做决定？**

| | workflow 步骤 | tool |
|---|---|---|
| 谁决定调用 | 代码，固定顺序 | 模型，看情况 |
| 参数谁给 | 代码 | 模型生成 JSON |
| 例子 | mp4 进来 → 一定抽音频 → 一定转写 | "第 20 分钟附近讲了什么？" → 模型决定查哪段 |

现在 mp4 进来**永远**要抽音频再转写，没有任何决策空间。把它做成 tool，等于让模型花一次调用去做
`if ext == ".mp4"` 能做的事：更慢、更贵、还可能出错。这是常见的反模式：**为了用 agent 而用 agent**。

**什么时候 ffmpeg/whisper 会变成合理的 tool？** 当入口变成"和 agent 对话"时：
"把上周三那个录音转一下，然后跟今天这个比较"——这时模型需要决定转哪个文件、转几个。那就是
B4 之后的 `transcribe_media(path)` tool，而它的实现就是一行 `ingest_file(...)`。**模块设计不用改**，
这正是把 ingest 做成独立 package 的回报。

**B3 的第一个 tool：`search_transcript` + `read_transcript_window`**（骨架在 `tools/`）。理由：
只读、无副作用、毫秒级、好测试；模型真的要做决定（搜什么、看哪段、够没够）；直接复用 MVP 的
`Transcript.segments`；而且是 B7 RAG 的前身（把关键词搜索换成 embedding 搜索，接口不变）。
有了它，"有几个人"这类问题就从"整篇塞进去猜"变成"模型自己去翻证据"，答案的可信度和可解释性都会上一个台阶。

原 BACKLOG B3 的 `extract_meeting_notes` 是"借 tool call 做结构化抽取"，和 B5 的 structured output
有重叠，学的是同一种东西。建议 B3 改成 search 这个，`extract_meeting_notes` 并进 B5。（BACKLOG 我已改，你不同意就回滚）

## 5. MVP 写代码顺序（每步可独立运行验证）

1. `ingest/models.py` → `detect.py` → `audio.py`：手跑一次 mp4 → wav
2. `transcribers.py` + `service.py`：mp4 → `raw.transcript.md/json`（**最慢的一步，跑通后后面都用 `--from-json` 跳过**）
3. `config.py` + `model_providers/` 全部：替换 `hello.py` 的验收——`create_provider(...).ask("hi")` 能回话
4. `cleanup.py` → `chat.py` → `cli.py`：端到端，问"有几个不同的人"

r2 **无需 `uv add`**：用到的都是 stdlib 或 pyproject 里已有的包。

## 6. Q5：commit → 迁移（你在本机 PowerShell 里跑，sandbox 写不了 .git）

```powershell
cd C:\Users\qli\Documents\Playground\meeting-intelligence-pipeline
Remove-Item .git\index.lock -ErrorAction SilentlyContinue   # 我这边 git status 留下的空锁文件
git status                                                  # _archive 下那批 " M" 是 CRLF 换行差异，内容无改动（已用 --ignore-cr-at-eol 验证）
git add -A
git commit -m "snapshot before MVP: hello/client (B1), skeleton r1-r2, worklog"
# 然后开写。迁移时：transcribe.py → ingest/（git mv 保留历史），client.py 删，hello.py 改用 create_provider
```

如果不想让 CRLF 噪音进 commit：先 `git config core.autocrlf true`，再 `git add --renormalize .`。
