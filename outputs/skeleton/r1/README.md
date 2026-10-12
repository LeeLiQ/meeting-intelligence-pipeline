# Skeleton r1 — MVP：mp4 → transcript → 一次 AI 清理 → 终端问答

> 2026-10-11，第 1 轮。只有签名 + 注释 + `...`，没有实现。你对照自己的思路批注，我们再出 r2。
> 目录结构 = 将来 `coachagent/` 的真实结构；确认后你照着写进 `coachagent/`。

## 1. 数据流（一条直线，没有 agent loop）

```
cli.py
 │  input("路径> ")
 ▼
ingest.ingest_file(path)                       ← 模块 1：输入预处理（package）
 │  detect.detect_kind()      mp4? wav?
 │  audio.extract_audio()     视频 → 16kHz mono wav  (ffmpeg, subprocess)
 │  transcribers.Whisper...   音频 → Transcript(segments=[...])  (openai-whisper)
 ▼
Transcript (Pydantic model)  ── 写盘 raw.transcript.md
 ▼
cleanup.clean_transcript(transcript, provider) ← 唯一一次 LLM 调用（workflow 节点）
 ▼
cleaned.md
 ▼
chat.ChatSession(provider, context=cleaned)    ← 模块 2：provider 适配的"消费者"
 │  while True: input("问> ") → session.ask(q) → print
```

模块 2 `model_providers/` 被 `cleanup` 和 `chat` 共用——这正是要抽象它的理由：**同一个 client，两种用法**（单发 / 多轮）。

## 2. 两个模块的设计要点

### 模块 1 `ingest/` —— 为什么是 package 而不是一个文件
同意你的判断。再补一条硬约束：**模块内部不 `print`、不 `input`**。进度用 callback 往外报
（`on_progress: Callable[[str], None]`），这样将来 UI（FastAPI / Gradio / Streamlit 都行）直接调用
`ingest_file()`，换一个 callback 就能显示进度条，不用改模块。CLI 只是第一个调用方。

- `Transcriber` 用 `typing.Protocol`（≈ C# interface，但是 structural typing：不用继承，方法签名对上就算实现）。
  将来可换 `faster-whisper`（CPU 快 4x、不依赖 torch）或 OpenAI 的云端 transcribe API。
- `Transcript` 保留 **segments（带时间戳）**，不只是一整段 text——下面第 3 节说为什么。

### 模块 2 `model_providers/` —— 参考 Pi 的思路
Pi（`pi-ai`）的核心：**统一的 message 格式 + 每个 provider 一个 adapter 负责翻译**，调用方只认统一格式。
我们照搬这个形状：

| 层 | 文件 | 对应 C# 概念 |
|---|---|---|
| 统一数据结构 `Message` / `ChatResult` | `base.py` | DTO |
| 抽象基类 `LLMProvider` | `base.py` | `abstract class`（用 `abc.ABC`） |
| 具体 adapter `OpenAIProvider` | `openai_provider.py` | 实现类 |
| key 管理 `resolve_api_key()` | `credentials.py` | `IConfiguration` + 用户输入兜底 |
| 工厂 `create_provider("openai")` | `registry.py` | DI container / factory |

**为什么这里用 ABC 而 `Transcriber` 用 Protocol？** `LLMProvider` 有共享行为（`ask()` 是建在 `chat()`
之上的便捷方法，所有子类白拿），这正是 abstract class 比 interface 合适的场景，和 C# 一样。
`Transcriber` 只有一个方法、没有共享逻辑，Protocol 更轻。旧 v1 的 `LLMProvider` 用的是 Protocol + 单发
`generate(system, user)`——v2 改成 `chat(messages)` 是因为要多轮对话。

**对话历史放 client 端（`ChatSession` 里的 list），不用 OpenAI 的 `previous_response_id`**：
那是 OpenAI 独有的服务端状态，用了就没法泛化到 Gemini/Anthropic。provider-agnostic 的代价是每轮重发全部历史。

## 3. ⚠️ 一个必须先知道的坑：Whisper 不分说话人

你的验收问题是"这份文档里有几个不同的人"。**openai-whisper 只做 ASR，不做 speaker diarization**——
transcript 里没有"谁说的"信息。所以 MVP 下 LLM 只能从内容线索推断（称呼、"我同意 X 刚才说的"、问答交替），
答案是**猜测**，不是事实。可选方案：

| 方案 | lib | 代价 |
|---|---|---|
| A. MVP 接受推断，prompt 要求模型说明依据和置信度 | 无 | 答案可能错 |
| B. 本地 diarization | `pyannote.audio`（或打包好的 `whisperx`） | 需 HuggingFace token + 同意模型条款，依赖重，Windows 上装 torch 版本要对齐 |
| C. 云端带 diarization 的 transcribe | OpenAI `gpt-4o-transcribe-diarize`（需核实当前可用性/参数） | 音频离开本机——对公司会议要先过 guardrail |

**我的建议：r1 走 A**，但 `TranscriptSegment` 预留 `speaker: str | None` 字段，B/C 将来作为另一个
`Transcriber` 实现插进来，上游代码不动。这也是"为什么 Transcriber 要抽象"的第一个真实理由。

## 4. 用到的 lib / package（都已在 pyproject 里，r1 **无需 `uv add`**）

| 用途 | package | import | 备注 |
|---|---|---|---|
| 视频 → 音频 | 系统 `ffmpeg`（不是 Python 包） | `subprocess` | Windows 需在 PATH；`winget install ffmpeg` |
| 本地 ASR | `openai-whisper` | `import whisper` | 依赖 torch；**它内部也调 ffmpeg 读音频** |
| 调 LLM | `openai` (v2.x) | `from openai import OpenAI` | 用 `client.responses.create`，不用 `chat.completions` |
| 读 `.env` | `python-dotenv` | `from dotenv import load_dotenv` | |
| 数据模型 | `pydantic` v2 | `from pydantic import BaseModel, Field` | |
| 隐藏输入 key | stdlib | `from getpass import getpass` | 输入时不回显 |
| 抽象 | stdlib | `from abc import ABC, abstractmethod` / `from typing import Protocol` | |
| 路径 | stdlib | `from pathlib import Path` | |
| CLI 参数 | stdlib | `import argparse` | 交互用 `input()`，先不引 `rich`/`typer` |

现代 Python 点名（3.7 → 3.13 你会遇到的）：`X | None` 代替 `Optional[X]`（3.10）、`list[str]` 代替
`List[str]`（3.9）、`enum.StrEnum`（3.11）、`match` 语句（3.10）、Pydantic v2 的 `model_dump()` /
`model_validate()`（v1 是 `.dict()` / `.parse_obj()`）。

## 5. 需要你拍板的问题（对比时顺手回答）

1. **Q1 diarization**：r1 接受方案 A（推断）？
2. **Q2 清理的边界**：AI 清理只做"去口头禅、断句、修错别字、分段"，**不总结、不改写意思**？（我是这么写的 prompt 意图）
3. **Q3 长 transcript**：1 小时会议 ≈ 1–1.5 万 token，单次调用放得下；r1 不做 chunking，对吗？
4. **Q4 key 的兜底**：`.env` 没 key 时，`getpass` 让用户输入——要不要询问后写回 `.env`？（Pi 是存 `~/.pi/agent/auth.json`；我们 09-06 决定不上 keyring）
5. **Q5 迁移**：现有 `coachagent/transcribe.py` 拆进 `ingest/`，`model_providers/client.py` 被 `registry.py` 取代，`hello.py` 删除或改用新接口——同意？
6. **Q6 产出落点**：transcript 写在源文件旁边，还是统一写到 `coach/sessions/<date>-<slug>/`（已 gitignore）？我倾向后者。
