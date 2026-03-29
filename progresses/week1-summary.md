# Week 1 Summary — Software Engineering Review

## 1. Error Handling Audit

| Function | What's Covered | What's Missing |
|---|---|---|
| `_extract_audio_from_video` | ✅ `CalledProcessError` (ffmpeg fails) ✅ `FileNotFoundError` (ffmpeg not installed) | Good shape. |
| `convert_audio_to_transcript_markdown` | ✅ File existence/type validation ✅ Unsupported file type ✅ Video extraction `RuntimeError` | ❌ `whisper.load_model()` and `model.transcribe()` are completely unguarded. Invalid model name, corrupted audio, or OOM will crash with an unhandled exception. |
| `summarize_and_extract_core_info_from_markdown` | ✅ File existence/type validation ✅ Missing API key check | ❌ The `try/except` around the Responses API catches **all** exceptions (`except Exception`) as a fallback to Chat Completions. This silently swallows genuine errors like `AuthenticationError`, `RateLimitError`, or `ConnectionError`. The fallback will likely fail with the same error, but the original context is lost. |
| `_sync_env_from_template` | ✅ Handles missing `.env` and missing template gracefully. | Good shape. |
| `main()` | ✅ Transcription step is wrapped in `try/except` | ❌ The summary step has **zero** error handling — inconsistent with the transcription step. If the LLM call fails, the user gets a raw stack trace. |

**Verdict:** Error handling is **partially there** but inconsistent. The two biggest gaps are:
- The blanket `except Exception` on the LLM call that hides real API errors.
- The summary step in `main()` having no error handling at all.

---

## 2. Retry Mechanisms

| Location | Should Retry? | Rationale |
|---|---|---|
| **LLM API call** (OpenAI) | ✅ **Yes** | Network calls are the #1 candidate. `RateLimitError` (429), transient `APIConnectionError`, or `InternalServerError` (500/503) are all recoverable with a short wait. The `openai` Python SDK has built-in retry support via `max_retries` on the client constructor — just pass `OpenAI(max_retries=3)`. |
| **ffmpeg subprocess** | ❌ No | If ffmpeg fails, it's almost always deterministic (corrupt file, wrong codec). Retrying won't help. |
| **Whisper transcription** | ❌ No | Runs locally on CPU/GPU. Failures are due to bad files or resource issues — not transient. |
| **File I/O** | ❌ No | Local disk operations failing is not a transient condition. |

**Verdict:** Add retry logic **only** to the OpenAI API call. Simplest approach: `OpenAI(api_key=..., max_retries=3)`.

---

## 3. Pipeline → Agent Evolution

### Current Architecture: Linear Pipeline

```
Audio/Video → [Stage 1: Transcribe] → Markdown → [Stage 2: Summarize] → Summary Markdown
```

Each stage takes an input artifact, processes it, and produces an output artifact that feeds the next stage. This is a textbook data pipeline.

### Pipeline vs Agent

| | Pipeline | Agent |
|---|---|---|
| **Control Flow** | Fixed, linear, deterministic | Dynamic, decided at runtime by the LLM |
| **Decision Making** | Developer decides the steps at code time | The LLM decides what to do next based on context |
| **Tools** | Functions called in hardcoded order | Functions registered as "tools" the LLM can invoke |
| **Loop** | Run once, start to finish | Runs in a loop until a goal is satisfied |

### Steps to Evolve into an Agent

1. **Wrap each stage as a "Tool"**: `transcribe_audio`, `summarize_markdown`, and future capabilities (e.g., `extract_action_items`, `send_email_summary`, `search_calendar`) become tools the LLM can call.
2. **Add an orchestration loop**: Instead of `main()` calling Stage 1 → Stage 2 in order, an **agent loop** receives a goal (e.g., *"Process this meeting recording and email the action items to the team"*) and decides which tools to call, in what order, based on intermediate results.
3. **Add memory/state**: The agent needs to track what it has done and what remains. This is where frameworks like **LangGraph**, **CrewAI**, or the **OpenAI Agents SDK** come in.

### Practical Next Step

Keep the current pipeline as-is for the deterministic happy path, but add an agent layer on top that can:
- Decide whether to skip summarization if the transcript is too short.
- Choose different LLM models based on transcript length.
- Chain additional downstream steps (e.g., extract action items → create Jira tickets).
